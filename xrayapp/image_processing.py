from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

from django.core.files.base import ContentFile
from django.core.exceptions import ValidationError
from django.core.files.uploadedfile import UploadedFile
from django.utils.translation import gettext_lazy as _

try:
    import numpy as np
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut
    from PIL import Image
    from PIL.ExifTags import TAGS
    DICOM_AVAILABLE = True
except ImportError:
    DICOM_AVAILABLE = False

try:
    import magic
    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ImageMetadata:
    """Small internal DTO for extracted image metadata."""

    name: str
    format: str
    size: str
    resolution: str
    date_created: datetime | None

    def as_dict(self) -> dict[str, Any]:
        # Keep backward compatible structure for legacy code paths that expect dict-like access.
        return {
            "name": self.name,
            "format": self.format,
            "size": self.size,
            "resolution": self.resolution,
            "date_created": self.date_created,
        }


def looks_like_dicom(filename: str | None, header: bytes) -> bool:
    """Heuristic DICOM detection.

    We intentionally keep this lightweight:
    - Standard DICOM files have a 128-byte preamble followed by b"DICM".
    - Many uploads also use the conventional .dcm extension.
    """
    if len(header) >= 132 and header[128:132] == b"DICM":
        return True
    if filename and filename.lower().endswith((".dcm", ".dicom")):
        return True
    return False


def get_image_mime_type(header: bytes) -> str | None:
    """Detect MIME type from file header using python-magic."""
    if not MAGIC_AVAILABLE:
        return None
        
    try:
        return magic.from_buffer(header, mime=True)
    except Exception as exc:
        logger.warning("python-magic MIME detection failed: %s", exc)
        return None


def convert_dicom_to_png(uploaded: UploadedFile) -> ContentFile:
    """Convert a DICOM upload into a PNG `ContentFile`.

    Why convert?
    - The rest of MCADS expects an image that PIL/skimage can read.
    - Templates use `<img src="{{ image_url }}">`, which won't render a `.dcm`.

    We keep the original name stem, but always return a `.png` file.
    """
    if not DICOM_AVAILABLE:
        raise ValidationError(_("DICOM support is not available on this server."))

    uploaded.seek(0)
    try:
        ds = pydicom.dcmread(uploaded, force=True)
    except Exception as exc:
        raise ValidationError(_("Invalid DICOM file.")) from exc

    try:
        arr = np.asarray(ds.pixel_array)
    except Exception as exc:
        # This can happen for compressed pixel data without extra decoders.
        raise ValidationError(_("Unable to read DICOM pixel data.")) from exc

    # Handle common shapes:
    # - (H, W) grayscale
    # - (frames, H, W) multi-frame → take first frame
    # - (H, W, 3/4) RGB/RGBA → take first channel (rare for X-rays)
    if arr.ndim == 3:
        if arr.shape[-1] in (3, 4):
            arr = arr[..., 0]
        else:
            arr = arr[0]
    elif arr.ndim > 3:
        # Fall back to the last 2 dimensions as an image plane.
        arr = arr.reshape(arr.shape[-2], arr.shape[-1])

    arr = arr.astype("float32", copy=False)

    # Apply rescale slope/intercept when present.
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    arr = arr * slope + intercept

    # Apply VOI LUT / windowing when available.
    try:
        arr = apply_voi_lut(arr, ds)
    except Exception:
        pass

    # MONOCHROME1 means the grayscale is inverted.
    if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        arr = float(arr.max()) - arr

    # Robust normalization to 8-bit for downstream pipelines.
    finite_mask = np.isfinite(arr)
    if not finite_mask.any():
        raise ValidationError(_("DICOM pixel data is empty or invalid."))

    finite_vals = arr[finite_mask]

    # Percentile computation on very large arrays can be noticeably slow.
    # A light, deterministic subsample preserves robustness while improving latency.
    sample = finite_vals
    if sample.size > 512_000:
        step = max(1, sample.size // 512_000)
        sample = sample[::step]

    low, high = np.percentile(sample, [1, 99]).astype("float32")
    if not (high > low):
        low = float(finite_vals.min())
        high = float(finite_vals.max())
    if not (high > low):
        high = low + 1.0

    arr = np.clip(arr, low, high)
    arr = (arr - low) / (high - low)
    arr8 = (arr * 255.0).clip(0, 255).astype("uint8")

    img = Image.fromarray(arr8, mode="L")
    out = BytesIO()
    # `optimize=True` can add seconds for large PNGs; prefer fast compression.
    img.save(out, format="PNG", compress_level=1)

    # Keep name stable and safe.
    original_name = (uploaded.name or "dicom").rsplit("/", 1)[-1]
    stem = original_name.rsplit(".", 1)[0] or "dicom"
    return ContentFile(out.getvalue(), name=f"{stem}.png")


def extract_image_metadata(image_path: str | Path) -> ImageMetadata:
    """
    Extract metadata from an image file
    
    Args:
        image_path: Path to the image file
        
    Returns:
        ImageMetadata: strongly-typed metadata DTO
    """
    try:
        # Ensure image_path is a Path object for cross-platform compatibility
        img_path = Path(image_path)
        
        # Open the image with PIL
        with Image.open(img_path) as img:
            # Get format
            image_format = img.format
            
            # Get resolution
            width, height = img.size
            resolution = f"{width}x{height}"
            
            # Get file size
            file_size_bytes = img_path.stat().st_size
            # Convert to KB or MB as appropriate
            if file_size_bytes < 1024 * 1024:
                size = f"{file_size_bytes / 1024:.1f} KB"
            else:
                size = f"{file_size_bytes / (1024 * 1024):.1f} MB"
            
            # Try to get creation date from EXIF data using proper method
            date_created = None
            try:
                # Use getexif() method instead of _getexif()
                exif_data = img.getexif()
                if exif_data:
                    exif = {
                        TAGS.get(tag, tag): value
                        for tag, value in exif_data.items()
                    }
                    if 'DateTimeOriginal' in exif:
                        date_created = datetime.strptime(exif['DateTimeOriginal'], '%Y:%m:%d %H:%M:%S')
            except (AttributeError, ValueError, TypeError):
                # Fallback if EXIF extraction fails
                pass
            
            # If no EXIF data, use file creation/modification time
            if date_created is None:
                # Use cross-platform stats
                stats = img_path.stat()
                # Try creation time first, then modification time as fallback
                try:
                    # ctime is creation time on Windows, change time on Unix
                    date_created = datetime.fromtimestamp(stats.st_ctime)
                except (OSError, ValueError):
                    # If there's an error, use modification time
                    date_created = datetime.fromtimestamp(stats.st_mtime)
            
            return ImageMetadata(
                name=img_path.name,
                format=str(image_format or "Unknown"),
                size=str(size),
                resolution=str(resolution),
                date_created=date_created,
            )
    except Exception:
        logger.exception("Error extracting metadata for %s", image_path)
        return ImageMetadata(
            name="Unknown",
            format="Unknown",
            size="Unknown",
            resolution="Unknown",
            date_created=None,
        )
