# Gunicorn configuration for MCADS Django application

# Server socket
bind = "0.0.0.0:8000"
backlog = 2048

# Worker processes - Reduced for memory optimization
workers = 1  # Single worker to prevent OOM on 2GB RAM server
worker_class = "uvicorn.workers.UvicornWorker"  # ASGI worker for better async performance
worker_connections = 500  # Reduced connections
timeout = 60  # Increased timeout for ML processing
keepalive = 2

# Restart workers more frequently to prevent memory leaks
max_requests = 100  # More frequent restarts
max_requests_jitter = 10

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