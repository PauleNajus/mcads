import os
import logging
import threading
import gc
from pathlib import Path
from typing import Dict, Tuple

import torch
import torchxrayvision as xrv

# Centralized cache for all heavy models used across the app
_model_cache: Dict[str, tuple] = {}

_AE_CACHE_KEY = "autoencoder"
_SEGMENTATION_CACHE_KEY = "segmentation_pspnet"

logger = logging.getLogger(__name__)
_cache_lock = threading.Lock()


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
    # Thread-safe cache: the web "thread fallback" can run inference concurrently.
    with _cache_lock:
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
        logger.info("Loaded model type: %s", model_type)
        return model, resize_dim


def load_autoencoder() -> Tuple[torch.nn.Module, int]:
    """Load and cache TorchXRayVision autoencoder for OOD gating.

    Returns:
        (autoencoder, input_resize_dim)
    """
    with _cache_lock:
        if _AE_CACHE_KEY in _model_cache:
            return _model_cache[_AE_CACHE_KEY]  # type: ignore[return-value]

        device = torch.device('cpu')
        _ensure_cache_dirs()
        ae_weights = os.environ.get('XRV_AE_WEIGHTS', '').strip() or '101-elastic'

        ae = xrv.autoencoders.ResNetAE(weights=ae_weights)
        ae.to(device)
        ae.eval()
        ae_resize_default = 224
        raw_resize = os.environ.get('XRV_AE_INPUT_SIZE')
        if raw_resize is None:
            ae_resize = ae_resize_default
        else:
            try:
                ae_resize = int(raw_resize)
            except (TypeError, ValueError):
                logger.warning(
                    "Invalid XRV_AE_INPUT_SIZE=%r; using default %d",
                    raw_resize,
                    ae_resize_default,
                )
                ae_resize = ae_resize_default
        _model_cache[_AE_CACHE_KEY] = (ae, ae_resize)
        logger.info("Loaded autoencoder: %s", ae_weights)
        return ae, ae_resize


def clear_model_cache() -> None:
    """Clear cached models to free memory."""
    with _cache_lock:
        _model_cache.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_segmentation_model() -> Tuple[torch.nn.Module, int]:
    """Load and cache TorchXRayVision PSPNet segmentation model.
    
    Returns:
        (model, resize_dim) - PSPNet model and its input resize dimension (512)
    """
    with _cache_lock:
        if _SEGMENTATION_CACHE_KEY in _model_cache:
            return _model_cache[_SEGMENTATION_CACHE_KEY]  # type: ignore[return-value]

        device = torch.device('cpu')
        _ensure_cache_dirs()

        logger.info("Loading PSPNet segmentation model...")

        try:
            # Load the PSPNet model
            seg_model = xrv.baseline_models.chestx_det.PSPNet()
            seg_model.to(device)
            seg_model.eval()

            # Cache the model (store as tuple for consistency)
            resize_dim = 512  # PSPNet uses 512x512 input
            _model_cache[_SEGMENTATION_CACHE_KEY] = (seg_model, resize_dim)
            logger.info("PSPNet segmentation model loaded successfully")

            return seg_model, resize_dim
        except Exception:
            # Include full traceback for operational debugging.
            logger.exception("Failed to load segmentation model")
            raise


def get_cached_model_count() -> int:
    """Return number of cached models (classifier + AE entries)."""
    with _cache_lock:
        return len(_model_cache)
