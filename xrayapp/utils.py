from __future__ import annotations

import logging
from typing import Any

# Configuration helpers
from .model_loader import load_model, load_autoencoder, load_segmentation_model, clear_model_cache

logger = logging.getLogger(__name__)

# PyTorch configuration moved to xrayapp/__init__.py to ensure early execution

_model_cache: dict[str, Any] = {}

def get_memory_info() -> dict[str, float | int]:
    """Get current memory usage information"""
    import psutil
    process = psutil.Process()
    memory_info = process.memory_info()
    
    # We can access _model_cache from model_loader via the new functions if needed, 
    # but since we're in utils, we might not have direct access to model_loader's private cache.
    # Actually, model_loader exposes `get_cached_model_count`.
    from .model_loader import get_cached_model_count
    
    return {
        'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size in MB
        'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size in MB
        'percent': process.memory_percent(),
        'cached_models': get_cached_model_count()
    }

# Re-export key functions for backward compatibility/ease of use
# (Though callers should ideally import from inference_logic directly)
from .inference_logic import (
    process_image,
    apply_segmentation,
    save_segmentation_results,
    save_and_record_visualization,
)
from .visualization_utils import (
    save_interpretability_visualization,
    save_heatmap,
    save_overlay,
    save_saliency_map,
    save_overlay_visualization,
    save_segmentation_visualization,
    save_individual_segmentation_masks,
)
