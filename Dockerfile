# Multi-stage build for MCADS X-ray Analysis System
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MKLDNN_ENABLED=0 \
    MKL_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    DEBIAN_FRONTEND=noninteractive

# Create app user
RUN groupadd -r mcads && useradd -r -g mcads mcads

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    libmagic1 \
    libmagic-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    libsm6 \
    libxext6 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
# Install PyTorch and torchvision from the official CPU index first to ensure wheels resolve,
# then install the remaining dependencies from requirements.txt (already satisfied pins are skipped).
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.7.0 torchvision==0.22.0 && \
    pip install --no-cache-dir -r requirements.txt

# Install PostgreSQL client for pg_isready
RUN apt-get update && apt-get install -y postgresql-client redis-tools && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Copy entrypoint script
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Create necessary directories
RUN mkdir -p /app/logs /app/media /app/staticfiles /app/data_exports /app/backups /app/.matplotlib /app/.torch /app/.torchxrayvision /home/mcads/.torchxrayvision /home/mcads/.torchxrayvision/models_data

# Change ownership to app user (including the entrypoint script)
RUN chown -R mcads:mcads /app && chown -R mcads:mcads /home/mcads && chown mcads:mcads /usr/local/bin/docker-entrypoint.sh

# Switch to app user
USER mcads

# Expose port
EXPOSE 8000

# Health check (root path responds without auth)
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Set entrypoint
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]

# Default command
CMD ["gunicorn", "--config", "gunicorn_config.py", "mcads_project.asgi:application"]