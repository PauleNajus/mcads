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

# Restart workers very frequently to prevent memory leaks on constrained system
max_requests = 20  # Very frequent restarts to prevent OOM
max_requests_jitter = 2

# Logging
accesslog = "/opt/mcads/app/logs/gunicorn_access.log"
errorlog = "/opt/mcads/app/logs/gunicorn_error.log"
loglevel = "info"

# Process naming
proc_name = 'mcads_gunicorn'

# Server mechanics
daemon = False
pidfile = "/opt/mcads/app/gunicorn.pid"
user = "paubun"
group = "paubun"
tmp_upload_dir = None

# SSL (if needed later)
# keyfile = "/path/to/ssl.key"
# certfile = "/path/to/ssl.crt"

# Security
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190 