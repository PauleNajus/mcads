from __future__ import annotations

"""
Tiny shared TorchXRayVision I/O helpers.

Why this exists:
- `xrayapp.utils` and `xrayapp.interpretability` both need the exact same
  "load image (supports DICOM) -> normalized (1, H, W) numpy array" behavior.
- Keeping the logic here avoids duplicated code and keeps imports lightweight
  (no OpenCV/matplotlib side-effects at import time).
"""

from pathlib import Path

import numpy as np
import torchxrayvision as xrv


def load_xrv_image(path: str | Path) -> np.ndarray:
    """Load and normalize an X-ray image for TorchXRayVision.

    Returns:
        Numpy array shaped (1, H, W) in TorchXRayVision's normalized space.

    Notes:
        Some DICOM exports omit the 128-byte preamble ("DICM" marker). If the file
        extension suggests DICOM and `xrv.utils.load_image()` fails, fall back to
        `xrv.utils.read_xray_dcm()`.
    """
    p = Path(path)
    s = str(p)
    try:
        return xrv.utils.load_image(s)
    except Exception:
        if p.suffix.lower() in (".dcm", ".dicom"):
            return xrv.utils.read_xray_dcm(s)[None, ...]
        raise

