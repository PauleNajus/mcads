import os
import logging

# CRITICAL FIX: PyTorch CPU backend configuration to prevent 75% stuck issue.
# This must run before torch is imported anywhere in the app.
# These environment variables must be set *before* importing torch to affect
# the underlying CPU backends. Operators can override them externally if needed.
_default_threads = os.environ.get("MCADS_CPU_THREADS")
if not _default_threads:
    _default_threads = str(max(1, min(os.cpu_count() or 1, 4)))

os.environ.setdefault("MKLDNN_ENABLED", "0")
os.environ.setdefault("MKL_NUM_THREADS", _default_threads)
os.environ.setdefault("OMP_NUM_THREADS", _default_threads)
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

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

# Try to configure PyTorch if available
try:
    import torch
    
    # Force PyTorch to use a safe CPU backend configuration.
    if os.environ.get("MKLDNN_ENABLED", "0") == "0":
        if hasattr(torch.backends, 'mkldnn'):
            torch.backends.mkldnn.enabled = False  # type: ignore[assignment]
    
    intra_threads = _env_int("OMP_NUM_THREADS", 1)
    interop_threads = _env_int("TORCH_NUM_INTEROP_THREADS", 1)
    torch.set_num_threads(intra_threads)
    try:
        # Inter-op threads should usually be small (1) to avoid oversubscription.
        torch.set_num_interop_threads(interop_threads)
    except Exception:
        # Some builds disallow changing interop threads after initialization.
        pass
    
    # Additional backend toggles (operators can override via env vars).
    if hasattr(torch.backends, 'openmp') and os.environ.get("MCADS_DISABLE_OPENMP", "0") == "1":
        torch.backends.openmp.is_available = lambda: False  # type: ignore[assignment]
    if hasattr(torch.backends, 'cudnn') and not torch.cuda.is_available():
        torch.backends.cudnn.enabled = False
        
    logger.info(
        "PyTorch configured for MCADS (intra_threads=%s, interop_threads=%s, mkldnn=%s)",
        intra_threads,
        interop_threads,
        "enabled" if os.environ.get("MKLDNN_ENABLED", "0") != "0" else "disabled",
    )
    
except ImportError:
    # PyTorch not installed or not available
    pass

default_app_config = 'xrayapp.apps.XrayappConfig'
