from __future__ import annotations

# Gunicorn configuration for MCADS Django application

# Server socket
bind = "0.0.0.0:8000"
backlog = 2048

# Worker processes - Reduced for memory optimization
workers = 1  # Single worker to prevent OOM on 2GB RAM server
worker_class = "uvicorn.workers.UvicornWorker"  # ASGI worker for better async performance
worker_connections = 100  # Very reduced connections for memory constrained system
timeout = 60  # Increased timeout for ML processing
keepalive = 2

# Restart workers periodically to prevent memory leaks on constrained system
max_requests = 1000  # Restart after 1000 requests to balance memory and performance
max_requests_jitter = 50

# Logging
# In containers, prefer stdout/stderr so logs are captured by Docker.
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Process naming
proc_name = 'mcads_gunicorn'

# Server mechanics
daemon = False
pidfile = "/tmp/gunicorn.pid"
# user and group are handled by Docker container user context
tmp_upload_dir = None

# SSL (if needed later)
# keyfile = "/path/to/ssl.key"
# certfile = "/path/to/ssl.crt"

# Security
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190

# Worker hooks to preserve environment variables
import os
from typing import Any

def pre_fork(server: Any, worker: Any) -> None:
    """Called just before a worker is forked."""
    # Ensure critical environment variables are set
    if 'USE_CELERY' not in os.environ:
        os.environ['USE_CELERY'] = '1'

def post_fork(server: Any, worker: Any) -> None:
    """Called just after a worker has been forked."""
    # Verify environment variables are still set
    server.log.info(f"Worker {worker.pid} started with USE_CELERY={os.environ.get('USE_CELERY', 'NOT SET')}") 