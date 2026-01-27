import os
import logging

# CRITICAL FIX: PyTorch CPU backend configuration to prevent 75% stuck issue.
# This must run before torch is imported anywhere in the app.
# These environment variables must be set *before* importing torch to affect
# the underlying CPU backends. Operators can override them externally if needed.
os.environ.setdefault('MKLDNN_ENABLED', '0')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
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
    
    torch.set_num_threads(_env_int("OMP_NUM_THREADS", 1))
    try:
        torch.set_num_interop_threads(_env_int("OMP_NUM_THREADS", 1))
    except Exception:
        # Some builds disallow changing interop threads after initialization.
        pass
    
    # Additional memory optimizations to prevent model inference hanging
    if hasattr(torch.backends, 'openmp') and os.environ.get("MCADS_DISABLE_OPENMP", "1") == "1":
        torch.backends.openmp.is_available = lambda: False  # type: ignore[assignment]
    if hasattr(torch.backends, 'cudnn') and not torch.cuda.is_available():
        torch.backends.cudnn.enabled = False
        
    logger.info("PyTorch optimized for MCADS - fixes applied to prevent 75% processing hang")
    
except ImportError:
    # PyTorch not installed or not available
    pass

default_app_config = 'xrayapp.apps.XrayappConfig'
