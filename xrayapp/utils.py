from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import logging
import os
from pathlib import Path
from typing import Any, cast

import cv2
import matplotlib.pyplot as plt
import numpy as np

# CRITICAL FIX: PyTorch CPU backend configuration to prevent 75% stuck issue.
#
# These environment variables must be set *before* importing torch to affect
# the underlying CPU backends. Operators can override them externally if needed.
os.environ.setdefault('MKLDNN_ENABLED', '0')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import torch
import torchxrayvision as xrv
import torchvision
from PIL import Image
from PIL.ExifTags import TAGS

from .xrv_io import load_xrv_image as _load_xrv_image
from .model_loader import (
    load_autoencoder as cached_load_autoencoder,
    load_model as cached_load_model,
    load_segmentation_model,
)
from .models import XRayImage

logger = logging.getLogger(__name__)

def _env_int(name: str, default: int) -> int:
    """Parse an int environment variable with a safe fallback."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default

# Force PyTorch to use a safe CPU backend configuration.
if os.environ.get("MKLDNN_ENABLED", "0") == "0":
    torch.backends.mkldnn.enabled = False

torch.set_num_threads(_env_int("OMP_NUM_THREADS", 1))
try:
    torch.set_num_interop_threads(_env_int("OMP_NUM_THREADS", 1))
except Exception:
    # Some builds disallow changing interop threads after initialization.
    pass

# Additional memory optimizations to prevent model inference hanging
if hasattr(torch.backends, 'openmp') and os.environ.get("MCADS_DISABLE_OPENMP", "1") == "1":
    torch.backends.openmp.is_available = lambda: False
if hasattr(torch.backends, 'cudnn') and not torch.cuda.is_available():
    torch.backends.cudnn.enabled = False

logger.info("PyTorch optimized for MCADS - fixes applied to prevent 75% processing hang")


_model_cache: dict[str, Any] = {}
_calibration_cache_key = "calibration"


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

def load_model(model_type: str = 'densenet') -> tuple[torch.nn.Module, int]:
    """Compatibility wrapper: use shared cached loader."""
    return cached_load_model(model_type)


def clear_model_cache() -> None:
    """Deprecated; kept for backwards compatibility."""
    from .model_loader import clear_model_cache as _clear
    _clear()


def load_autoencoder() -> tuple[torch.nn.Module, int]:
    """Compatibility wrapper: use shared cached loader."""
    return cached_load_autoencoder()


def compute_ood_score(img_np: np.ndarray) -> dict[str, float | bool]:
    """Compute an OOD score via reconstruction error from the autoencoder.

    Args:
        img_np: numpy array after `xrv.datasets.normalize`, shape (H, W) or (1, H, W)

    Returns:
        dict with keys: { 'ood_score': float, 'is_ood': bool, 'threshold': float }
    """
    try:
        ae, ae_resize = load_autoencoder()
    except Exception:
        # If AE cannot be loaded, gracefully skip OOD gate
        return { 'ood_score': float('nan'), 'is_ood': False, 'threshold': float('inf') }

    # Ensure shape is (1, H, W)
    if img_np.ndim == 2:
        img_np = img_np[None, :, :]

    # Resize to AE input (center crop then resize like classifier preprocessing)
    transform = torchvision.transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(ae_resize)
    ])
    img_small = transform(img_np)

    # To tensor on CPU
    img_tensor = torch.from_numpy(img_small).unsqueeze(0)  # (1, 1, ae_resize, ae_resize)

    with torch.no_grad():
        # Encode and decode (AE API not typed; cast to Any for tooling)
        ae_mod: Any = ae
        z = ae_mod.encode(img_tensor)
        recon = ae_mod.decode(z)

    # Compute normalized MSE per pixel
    recon_err = (recon - img_tensor).pow(2).mean().item()

    # Threshold from env or default conservative value
    default_thr = 0.015  # empirically safe; adjustable via env
    thr = float(os.environ.get('XRV_AE_OOD_THRESHOLD', default_thr))
    is_ood = recon_err > thr

    return { 'ood_score': float(recon_err), 'is_ood': bool(is_ood), 'threshold': float(thr) }


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def _logit(p: float, eps: float = 1e-6) -> float:
    p = float(np.clip(p, eps, 1.0 - eps))
    return float(np.log(p / (1.0 - p)))


def load_calibration() -> dict[str, dict[str, float]]:
    """Load per-pathology temperature calibration and thresholds.

    Sources (in priority order):
    - Env var XRV_CALIBRATION_JSON: JSON string with keys {"temperature": {...}, "thresholds": {...}}
    - File at env var XRV_CALIBRATION_FILE
    - Default file at BASE_DIR/config/calibration.json if exists
    - Fallback to identity temps=1.0 and thresholds=0.5
    """
    if _calibration_cache_key in _model_cache:
        return _model_cache[_calibration_cache_key]

    # Defaults
    temps: dict[str, float] = {}
    thrs: dict[str, float] = {}

    # Try env JSON
    import json
    env_json = os.environ.get('XRV_CALIBRATION_JSON')
    if env_json:
        try:
            blob = json.loads(env_json)
            temps = {k: float(v) for k, v in blob.get('temperature', {}).items()}
            thrs = {k: float(v) for k, v in blob.get('thresholds', {}).items()}
        except Exception as e:
            logger.error(f"Failed to parse XRV_CALIBRATION_JSON: {e}")

    # Try file
    if not temps and not thrs:
        file_path = os.environ.get('XRV_CALIBRATION_FILE')
        if not file_path:
            # Default location under project base
            base_dir = Path(__file__).resolve().parent.parent
            file_path = str(base_dir / 'config' / 'calibration.json')
        try:
            if Path(file_path).exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    blob = json.load(f)
                    temps = {k: float(v) for k, v in blob.get('temperature', {}).items()}
                    thrs = {k: float(v) for k, v in blob.get('thresholds', {}).items()}
        except Exception as e:
            logger.error(f"Failed to load calibration file {file_path}: {e}")

    # Fill with identity defaults if missing based on known default pathologies
    default_pathologies = list(xrv.datasets.default_pathologies)
    for name in default_pathologies:
        temps.setdefault(name, 1.0)
        thrs.setdefault(name, 0.5)

    cfg = { 'temperature': temps, 'thresholds': thrs }
    _model_cache[_calibration_cache_key] = cfg
    return cfg


def apply_calibration_and_thresholds(results: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply temperature scaling (per label) and generate binary labels.

    Returns updated results and a metadata dictionary.
    """
    cfg = load_calibration()
    temps = cfg['temperature']
    thrs = cfg['thresholds']

    calibrated: dict[str, float] = {}
    binary: dict[str, bool] = {}
    for k, p in list(results.items()):
        # Skip non-pathology keys (added later) by checking value type
        if not isinstance(p, (float, int, np.floating)):
            continue
        T = max(1e-3, float(temps.get(k, 1.0)))
        # Temperature scaling in logit space
        logit = _logit(float(p))
        p_cal = _sigmoid(logit / T)
        calibrated[k] = p_cal
        thr = float(thrs.get(k, 0.5))
        binary[k] = bool(p_cal >= thr)

    # Merge back
    updated = dict(results)
    updated.update({k: float(v) for k, v in calibrated.items()})
    meta = {
        'calibration_temperatures': temps,
        'decision_thresholds': thrs,
        'predicted_labels': binary
    }
    return updated, meta


def get_memory_info() -> dict[str, float | int]:
    """Get current memory usage information"""
    import psutil
    process = psutil.Process()
    memory_info = process.memory_info()
    
    return {
        'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size in MB
        'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size in MB
        'percent': process.memory_percent(),
        'cached_models': len(_model_cache)
    }


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


def process_image(
    image_path: str | Path,
    xray_instance: XRayImage | None = None,
    model_type: str = 'densenet',
) -> dict[str, Any]:
    """
    Process an X-ray image and return predictions
    If xray_instance is provided, will update progress
    
    Args:
        image_path: Path to the image
        xray_instance: Database instance for progress tracking
        model_type (str): 'densenet' or 'resnet'
    """
    # Update status to processing
    if xray_instance:
        xray_instance.processing_status = 'processing'
        xray_instance.progress = 5
        xray_instance.save(update_fields=['processing_status', 'progress'])
    
    # Extract and save image metadata
    if xray_instance:
        metadata = extract_image_metadata(image_path)
        # If the upload was DICOM (converted to PNG for processing), keep the
        # user-facing format as DICOM instead of overwriting with "PNG".
        if not (xray_instance.image_format and str(xray_instance.image_format).upper() == "DICOM"):
            xray_instance.image_format = metadata.format
        xray_instance.image_size = metadata.size
        xray_instance.image_resolution = metadata.resolution
        xray_instance.image_date_created = metadata.date_created
        xray_instance.save(update_fields=['image_format', 'image_size', 'image_resolution', 'image_date_created'])
    
    # Load and preprocess the image
    # Update progress to 10%
    if xray_instance:
        xray_instance.progress = 10
        xray_instance.save(update_fields=['progress'])
    
    # Ensure image_path is a Path object.
    image_path = Path(image_path)
    
    # Normalize the image
    # Update progress to 20%
    if xray_instance:
        xray_instance.progress = 20
        xray_instance.save(update_fields=['progress'])
    
    # TorchXRayVision helper: supports DICOM + normal images and returns the
    # expected model input (normalized, shape (1, H, W)).
    img = _load_xrv_image(image_path)

    # OOD gate (reconstruction error)
    ood = compute_ood_score(img[0])  # pass (H, W)
    if xray_instance:
        # Flag for review if likely OOD
        xray_instance.requires_expert_review = bool(ood.get('is_ood', False))
        xray_instance.save(update_fields=['requires_expert_review'])
    
    # Load model and get resize dimension
    # Update progress to 40%
    if xray_instance:
        xray_instance.progress = 40
        xray_instance.save(update_fields=['progress'])
    
    model, resize_dim = load_model(model_type)
    model = cast(torch.nn.Module, model)
    
    # Apply transforms for model: center-crop then resize.
    transform = torchvision.transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(resize_dim),
    ])
    img = transform(img)
    
    # Convert to tensor
    img_tensor = torch.from_numpy(img)
    
    # Add batch dimension
    if len(img_tensor.shape) < 3:
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
    elif len(img_tensor.shape) == 3:
        img_tensor = img_tensor.unsqueeze(0)
    
    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_tensor = img_tensor.to(device)
    model = model.to(device)
    
    # Update progress to 60%
    if xray_instance:
        xray_instance.progress = 60
        xray_instance.save(update_fields=['progress'])
    
    # Get predictions
    # Update progress to 75%
    if xray_instance:
        xray_instance.progress = 75
        xray_instance.save(update_fields=['progress'])
    
    with torch.inference_mode():
        # Use the model's forward method for both model types
        preds = model(img_tensor).cpu()
    
    # Create a dictionary of pathology predictions
    # For ResNet, ALWAYS use default_pathologies for correct mapping
    if model_type == 'resnet':
        # ResNet model outputs 18 values in the order of default_pathologies
        results = dict(zip(list(xrv.datasets.default_pathologies), preds[0].detach().numpy()))
    else:
        # For DenseNet, we can use the model's pathologies directly (access dynamically)
        model_pathologies: list[str] = list(getattr(model, 'pathologies', list(xrv.datasets.default_pathologies)))
        results = dict(zip(model_pathologies, preds[0].detach().numpy()))
    
    # Filter out specific classes for ResNet if needed
    # Note: These classes will always output 0.5 for ResNet as they're not trained
    if model_type == 'resnet':
        excluded_classes = ["Enlarged Cardiomediastinum", "Lung Lesion"]
        results = {k: v for k, v in results.items() if k not in excluded_classes}
    
    # Apply calibration and thresholds
    results, calib_meta = apply_calibration_and_thresholds(results)

    # If we have an XRay instance, report "almost done".
    #
    # NOTE: We deliberately do NOT set `processing_status='completed'` here,
    # because prediction fields are persisted by the caller after this function
    # returns. Marking "completed" early can briefly expose an inconsistent row.
    if xray_instance:
        xray_instance.progress = 90
        xray_instance.save(update_fields=['progress'])
    
    # Attach OOD + calibration metadata (not persisted to DB fields)
    results["OOD_Score"] = ood.get('ood_score')
    results["OOD_Threshold"] = ood.get('threshold')
    results["Is_OOD"] = ood.get('is_ood')
    results["Calibration"] = {
        'temperatures': calib_meta.get('calibration_temperatures'),
        'thresholds': calib_meta.get('decision_thresholds'),
        'labels': calib_meta.get('predicted_labels'),
    }

    return results 


def save_interpretability_visualization(
    interpretation_results: dict[str, Any],
    output_path: Path,
    format: str = 'png',
) -> Path:
    """
    Save interpretability visualization to a file
    
    Args:
        interpretation_results: Results from process_image_with_interpretability
        output_path: Path to save the visualization
        format: Image format to save (png, jpg, etc.)
        
    Returns:
        Path to saved file
    """
    # Create figure with subplots
    plt.figure(figsize=(12, 4))
    
    # Plot original image
    plt.subplot(1, 3, 1)
    plt.imshow(interpretation_results['original'], cmap='gray')
    plt.title('Original X-ray')
    plt.axis('off')
    
    # Plot method-specific visualizations
    if interpretation_results.get('method') == 'gradcam':
        # Plot heatmap
        plt.subplot(1, 3, 2)
        plt.imshow(interpretation_results['heatmap'], cmap='jet')
        plt.title(f'Heatmap\n{interpretation_results["target_class"]}')
        plt.axis('off')
        
        # Plot overlay
        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(interpretation_results['overlay'], cv2.COLOR_BGR2RGB))
        plt.title('Grad-CAM Overlay')
        plt.axis('off')
    
    elif interpretation_results.get('method') == 'combined_gradcam':
        # Plot combined heatmap
        plt.subplot(1, 3, 2)
        plt.imshow(interpretation_results['heatmap'], cmap='jet')
        plt.title(f'Combined Heatmap\n{len(interpretation_results["selected_pathologies"])} pathologies > {interpretation_results["threshold"]}')
        plt.axis('off')
        
        # Plot overlay
        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(interpretation_results['overlay'], cv2.COLOR_BGR2RGB))
        plt.title('Combined Overlay')
        plt.axis('off')
    
    elif interpretation_results.get('method') == 'combined_pli':
        # Plot combined saliency map
        plt.subplot(1, 3, 2)
        plt.imshow(interpretation_results['saliency_map'], cmap='jet')
        plt.title(f'Combined PLI Saliency\n{len(interpretation_results["selected_pathologies"])} pathologies > {interpretation_results["threshold"]}')
        plt.axis('off')
        
        # Plot overlay
        plt.subplot(1, 3, 3)
        plt.imshow(interpretation_results['overlay'])
        plt.title('Combined PLI Overlay')
        plt.axis('off')
    
    elif interpretation_results.get('method') == 'pli':
        # Plot saliency map
        plt.subplot(1, 3, 2)
        plt.imshow(interpretation_results['saliency_map'], cmap='jet')
        plt.title(f'Saliency Map\n{interpretation_results["target_class"]}')
        plt.axis('off')
        
        # Plot overlay rather than colored saliency
        plt.subplot(1, 3, 3)
        plt.imshow(interpretation_results['overlay'])
        plt.title('Pixel-Level Overlay')
        plt.axis('off')
    
    # Add image metadata as text if available
    if 'metadata' in interpretation_results:
        metadata = interpretation_results['metadata']
        metadata_text = f"Image: {metadata.get('name', 'Unknown')} | Format: {metadata.get('format', 'Unknown')} | Size: {metadata.get('size', 'Unknown')} | Resolution: {metadata.get('resolution', 'Unknown')}"
        plt.figtext(0.5, 0.01, metadata_text, ha='center', fontsize=8, bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
    
    # Save the figure with metadata preserved
    plt.tight_layout()
    plt.savefig(output_path, format=format, dpi=300, bbox_inches='tight', metadata={'Interpretation': interpretation_results.get('method', 'Unknown'), 
                                                                                    'Target': interpretation_results.get('target_class', 'Unknown')})
    plt.close()
    
    return output_path


def save_overlay_visualization(
    interpretation_results: dict[str, Any],
    output_path: Path,
    format: str = 'png',
) -> Path:
    """
    Save only the overlay visualization to a file for pixel-level interpretability without white spaces
    
    Args:
        interpretation_results: Results from apply_pixel_interpretability
        output_path: Path to save the visualization
        format: Image format to save (png, jpg, etc.)
        
    Returns:
        Path to saved file
    """
    if 'overlay' in interpretation_results:
        # Get the overlay data (should already be in proper RGB format)
        overlay = interpretation_results['overlay']
        
        # Save directly as image without matplotlib padding
        from PIL import Image
        Image.fromarray(overlay).save(output_path, format=format.upper())
    
    return output_path


def save_saliency_map(
    interpretation_results: dict[str, Any],
    output_path: Path,
    format: str = 'png',
) -> Path:
    """
    Save only the saliency map to a file for pixel-level interpretability without white spaces
    
    Args:
        interpretation_results: Results from apply_pixel_interpretability
        output_path: Path to save the visualization
        format: Image format to save (png, jpg, etc.)
        
    Returns:
        Path to saved file
    """
    if 'saliency_map' in interpretation_results:
        # Get the saliency map data
        saliency_map = interpretation_results['saliency_map']
        
        # Apply colormap to saliency map (convert to 0-255 range)
        saliency_colored = cv2.applyColorMap(np.uint8(255 * saliency_map), cv2.COLORMAP_JET)  # type: ignore
        
        # Convert BGR to RGB for proper color display
        saliency_rgb = cv2.cvtColor(saliency_colored, cv2.COLOR_BGR2RGB)
        
        # Save directly as image without matplotlib padding
        from PIL import Image
        Image.fromarray(saliency_rgb).save(output_path, format=format.upper())
    
    return output_path


def save_heatmap(
    interpretation_results: dict[str, Any],
    output_path: Path,
    format: str = 'png',
) -> Path:
    """
    Save only the heatmap to a file without white spaces
    
    Args:
        interpretation_results: Results from interpretability analysis
        output_path: Path to save the visualization
        format: Image format to save (png, jpg, etc.)
        
    Returns:
        Path to saved file
    """
    if 'heatmap' in interpretation_results and 'original' in interpretation_results:
        # Get the heatmap data and original image dimensions
        heatmap = interpretation_results['heatmap']
        original_shape = interpretation_results['original'].shape
        
        # Resize heatmap to match original image dimensions
        heatmap_resized = cv2.resize(heatmap, (original_shape[1], original_shape[0]))
        
        # Apply colormap to heatmap (convert to 0-255 range)
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)  # type: ignore
        
        # Convert BGR to RGB for proper color display
        heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Save directly as image without matplotlib padding
        from PIL import Image
        Image.fromarray(heatmap_rgb).save(output_path, format=format.upper())
    
    return output_path


def save_overlay(
    interpretation_results: dict[str, Any],
    output_path: Path,
    format: str = 'png',
) -> Path:
    """
    Save only the overlay to a file without white spaces
    
    Args:
        interpretation_results: Results from interpretability analysis
        output_path: Path to save the visualization
        format: Image format to save (png, jpg, etc.)
        
    Returns:
        Path to saved file
    """
    if 'overlay' in interpretation_results:
        # Get the overlay data (already in RGB format from overlay_heatmap method)
        overlay = interpretation_results['overlay']
        
        # Convert BGR to RGB if needed (overlay_heatmap returns BGR format)
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        
        # Save directly as image without matplotlib padding
        from PIL import Image
        Image.fromarray(overlay_rgb).save(output_path, format=format.upper())
    
    return output_path


def apply_segmentation(image_path: str) -> dict[str, Any]:
    """
    Apply anatomical segmentation to a chest X-ray image using PSPNet
    
    Args:
        image_path: Path to the X-ray image
        
    Returns:
        Dictionary containing segmentation results
    """
    logger.info(f"Applying segmentation to image: {image_path}")
    
    # Load the segmentation model
    seg_model = load_segmentation_model()
    
    # Load + normalize image (supports DICOM via pydicom).
    img = _load_xrv_image(image_path)
    
    # Preserve original image for visualization (2D)
    original_img = img[0].copy()
    
    # Resize to 512x512 for PSPNet
    transform = torchvision.transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(512)
    ])
    
    # Apply transform to numpy array first
    img_transformed = transform(img)
    
    # Convert to tensor and add batch dimension
    img_tensor = torch.from_numpy(img_transformed).float().unsqueeze(0)
    
    # Perform segmentation
    with torch.no_grad():
        output = seg_model(img_tensor)
    
    # Output shape is [1, 14, 512, 512]
    # 14 channels for different anatomical structures
    segmentation_masks = output.squeeze(0).cpu().numpy()
    
    # Define the anatomical structures
    anatomical_structures = [
        'Left Clavicle', 'Right Clavicle', 'Left Scapula', 'Right Scapula',
        'Left Lung', 'Right Lung', 'Left Hilus Pulmonis', 'Right Hilus Pulmonis',
        'Heart', 'Aorta', 'Facies Diaphragmatica', 'Mediastinum',
        'Weasand', 'Spine'
    ]
    
    # Apply sigmoid to get probabilities
    segmentation_probs = torch.sigmoid(output).squeeze(0).cpu().numpy()
    
    # Create binary masks with threshold
    threshold = 0.5
    binary_masks = (segmentation_probs > threshold).astype(np.uint8)
    
    # Store results
    results = {
        'segmentation_masks': segmentation_masks,
        'segmentation_probs': segmentation_probs,
        'binary_masks': binary_masks,
        'anatomical_structures': anatomical_structures,
        'original': original_img,
        'num_structures': len(anatomical_structures)
    }
    
    logger.info("Segmentation completed successfully")
    return results


def save_segmentation_visualization(
    segmentation_results: dict[str, Any],
    output_path: Path,
    structures_to_show: list[int] | None = None,
) -> Path:
    """
    Save segmentation visualization with colored overlays for each anatomical structure
    
    Args:
        segmentation_results: Results from apply_segmentation
        output_path: Path to save the visualization
        structures_to_show: List of structure indices to show (None = show all)
        
    Returns:
        Path to saved file
    """
    original = segmentation_results['original']
    binary_masks = segmentation_results['binary_masks']
    anatomical_structures = segmentation_results['anatomical_structures']
    
    # `original` is stored in TorchXRayVision's normalized space (~[-1024, 1024]).
    # Convert it to a displayable 8-bit image before compositing overlays.
    #
    # This fixed mapping is intentional (fast + consistent across images):
    #   -1024 -> 0, 0 -> 127/128, 1024 -> 255
    if len(original.shape) == 2:
        orig01 = (original.astype("float32") / 1024.0 + 1.0) * 0.5
        orig01 = np.clip(orig01, 0.0, 1.0)
        original_u8 = (orig01 * 255.0).astype(np.uint8)
        # Keep OpenCV images in BGR because our overlay colors are defined in BGR.
        original_bgr = cv2.cvtColor(original_u8, cv2.COLOR_GRAY2BGR)
    else:
        # Fallback for unexpected 3-channel inputs (assume already in [0, 1]).
        original_bgr = (np.clip(original, 0.0, 1.0) * 255.0).astype(np.uint8)
    
    # Resize original to match mask dimensions (512x512)
    original_resized = cv2.resize(original_bgr, (512, 512))
    
    # Create overlay image
    overlay = original_resized.copy()
    
    # Define colors for each anatomical structure (BGR format)
    colors = [
        (255, 0, 0),     # Blue - Left Clavicle
        (0, 255, 0),     # Green - Right Clavicle
        (0, 0, 255),     # Red - Left Scapula
        (255, 255, 0),   # Cyan - Right Scapula
        (255, 0, 255),   # Magenta - Left Lung
        (0, 255, 255),   # Yellow - Right Lung
        (128, 0, 255),   # Purple - Left Hilus Pulmonis
        (255, 128, 0),   # Orange - Right Hilus Pulmonis
        (0, 128, 255),   # Light Blue - Heart
        (255, 0, 128),   # Pink - Aorta
        (128, 255, 0),   # Lime - Facies Diaphragmatica
        (0, 255, 128),   # Spring Green - Mediastinum
        (128, 128, 255), # Light Purple - Weasand
        (255, 128, 128)  # Light Red - Spine
    ]
    
    # Apply masks
    if structures_to_show is None:
        structures_to_show = list(range(len(anatomical_structures)))
    
    for idx in structures_to_show:
        if idx < len(binary_masks):
            mask = binary_masks[idx]
            color = colors[idx % len(colors)]
            
            # Create colored mask
            colored_mask = np.zeros_like(overlay)
            colored_mask[:, :] = color
            
            # Apply mask with transparency
            # Multiply colored mask by the binary mask and alpha value
            masked_color = (colored_mask * mask[:, :, np.newaxis] * 0.3).astype(overlay.dtype)
            overlay = cv2.addWeighted(overlay, 1.0, masked_color, 1.0, 0)
    
    # Convert BGR to RGB for saving
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    
    # Save the visualization
    Image.fromarray(overlay_rgb).save(output_path, format='PNG')
    
    return output_path


def save_individual_segmentation_masks(
    segmentation_results: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    """
    Save individual binary masks for each anatomical structure
    
    Args:
        segmentation_results: Results from apply_segmentation
        output_dir: Directory to save individual masks
        
    Returns:
        Dictionary mapping structure names to file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    binary_masks = segmentation_results['binary_masks']
    anatomical_structures = segmentation_results['anatomical_structures']
    
    saved_paths = {}
    
    for idx, structure_name in enumerate(anatomical_structures):
        if idx < len(binary_masks):
            mask = binary_masks[idx]
            
            # Convert to uint8 (0 or 255)
            mask_img = (mask * 255).astype(np.uint8)
            
            # Clean filename
            filename = f"{structure_name.lower().replace(' ', '_')}_mask.png"
            filepath = output_dir / filename
            
            # Save mask
            Image.fromarray(mask_img, mode='L').save(filepath)
            saved_paths[structure_name] = str(filepath)
    
    return saved_paths 