import os
from pathlib import Path
from typing import Dict, Tuple

import torch
import torchxrayvision as xrv

# Centralized cache for all heavy models used across the app
_model_cache: Dict[str, tuple] = {}

_AE_CACHE_KEY = "autoencoder"
_CALIBRATION_CACHE_KEY = "calibration"
_SEGMENTATION_CACHE_KEY = "segmentation_pspnet"


def _ensure_cache_dirs() -> str:
    """Ensure a writable cache directory is set for Torch/XRV.

    Returns:
        The selected cache directory path as a string.
    """
    default_cache = str((Path(__file__).resolve().parent.parent / '.torchxrayvision').resolve())
    cache_dir = os.environ.get('TORCHXRAYVISION_CACHE_DIR', default_cache)
    os.makedirs(cache_dir, exist_ok=True)
    os.environ['XRV_DATA_DIR'] = cache_dir
    os.environ['TORCHXRAYVISION_CACHE_DIR'] = cache_dir
    os.environ['TORCH_HOME'] = cache_dir
    return cache_dir


def load_model(model_type: str = 'densenet') -> Tuple[torch.nn.Module, int]:
    """Load and cache TorchXRayVision classifier models.

    Args:
        model_type: 'densenet' or 'resnet'

    Returns:
        (model, resize_dim)
    """
    if model_type in _model_cache:
        return _model_cache[model_type]  # type: ignore[return-value]

    device = torch.device('cpu')
    _ensure_cache_dirs()

    if model_type == 'resnet':
        weights = os.environ.get('XRV_RESNET_WEIGHTS', 'resnet50-res512-all')
        model = xrv.models.ResNet(weights=weights)
        resize_dim = 512
    else:
        weights = os.environ.get('XRV_DENSENET_WEIGHTS', 'densenet121-res224-all')
        model = xrv.models.DenseNet(weights=weights)
        resize_dim = 224

    model.to(device)
    model.eval()
    _model_cache[model_type] = (model, resize_dim)
    return model, resize_dim


def load_autoencoder() -> Tuple[torch.nn.Module, int]:
    """Load and cache TorchXRayVision autoencoder for OOD gating.

    Returns:
        (autoencoder, input_resize_dim)
    """
    if _AE_CACHE_KEY in _model_cache:
        return _model_cache[_AE_CACHE_KEY]  # type: ignore[return-value]

    device = torch.device('cpu')
    _ensure_cache_dirs()
    ae_weights = os.environ.get('XRV_AE_WEIGHTS', '').strip() or '101-elastic'

    ae = xrv.autoencoders.ResNetAE(weights=ae_weights)
    ae.to(device)
    ae.eval()
    ae_resize = int(os.environ.get('XRV_AE_INPUT_SIZE', '64'))
    _model_cache[_AE_CACHE_KEY] = (ae, ae_resize)
    return ae, ae_resize


def clear_model_cache() -> None:
    """Clear cached models to free memory."""
    _model_cache.clear()
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_segmentation_model() -> torch.nn.Module:
    """Load and cache TorchXRayVision PSPNet segmentation model.
    
    Returns:
        PSPNet model for anatomical structure segmentation
    """
    if _SEGMENTATION_CACHE_KEY in _model_cache:
        return _model_cache[_SEGMENTATION_CACHE_KEY][0]  # Return just the model, no resize dim
    
    device = torch.device('cpu')
    _ensure_cache_dirs()
    
    import logging
    logger = logging.getLogger(__name__)
    logger.info("Loading PSPNet segmentation model...")
    
    try:
        # Load the PSPNet model
        seg_model = xrv.baseline_models.chestx_det.PSPNet()
        seg_model.to(device)
        seg_model.eval()
        
        # Cache the model (store as tuple for consistency)
        _model_cache[_SEGMENTATION_CACHE_KEY] = (seg_model, 512)  # PSPNet uses 512x512 input
        logger.info("PSPNet segmentation model loaded successfully")
        
        return seg_model
    except Exception as e:
        logger.error(f"Failed to load segmentation model: {e}")
        raise


def get_cached_model_count() -> int:
    """Return number of cached models (classifier + AE entries)."""
    return len(_model_cache)


