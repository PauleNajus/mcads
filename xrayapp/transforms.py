from __future__ import annotations

import torchvision
import torchxrayvision as xrv

def get_xrv_transform(resize_dim: int) -> torchvision.transforms.Compose:
    """Return the standard TorchXRayVision preprocessing pipeline.
    
    This ensures consistent preprocessing (center crop + resize) across
    inference and interpretability modules.
    """
    return torchvision.transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(resize_dim),
    ])


def get_center_crop_bounds(h: int, w: int) -> tuple[int, int, int, int]:
    """Return (y0, y1, x0, x1) bounds for TorchXRayVision-style center crop.

    TorchXRayVision's `XRayCenterCrop()` produces a centered square crop with side
    length `min(h, w)`. We replicate that here so heatmaps (computed on
    the cropped+resized model input) can be mapped back onto the *original*
    rectangular image without stretching.
    """
    size = min(int(h), int(w))
    y0 = max(0, (int(h) - size) // 2)
    x0 = max(0, (int(w) - size) // 2)
    return y0, y0 + size, x0, x0 + size
