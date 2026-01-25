# syntax=docker/dockerfile:1.6

# Debian 13 (trixie) + Python 3.11.9 runtime.
#
# Note: Debian 13 ships Python 3.13 by default, so we install a pinned CPython
# 3.11.9 using `uv` (fast, reproducible, and avoids compiling from source).
FROM ghcr.io/astral-sh/uv:0.9.26 AS uv

FROM debian:trixie-slim

ARG PYTHON_VERSION=3.11.9

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Keep the interpreter + venv in stable locations
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    # PyTorch CPU backend fixes (also set in Django settings for safety)
    MKLDNN_ENABLED=0 \
    MKL_NUM_THREADS=1 \
    OMP_NUM_THREADS=1

# OS runtime dependencies for this repo:
# - libmagic1: python-magic
# - libgl1 + libglib2.0-0 (+ a couple X libs): opencv-python import/runtime
# - libpq5: psycopg2-binary compatibility with libpq symbols (safe, small)
# - bash: docker-entrypoint.sh uses bash
RUN apt-get update && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        gosu \
        libmagic1 \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libxext6 \
        libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Copy the single-file uv binary from the pinned uv image (entrypoint is /uv).
COPY --from=uv /uv /usr/local/bin/uv

# Install pinned CPython and create a seeded venv.
RUN uv python install "${PYTHON_VERSION}" && \
    uv venv --seed --python "${PYTHON_VERSION}" "${VIRTUAL_ENV}"

WORKDIR /app

# Install Python deps first for better layer caching.
COPY requirements.txt /app/requirements.txt

# Use the CPU-only PyTorch wheel index as an extra index so `torch==2.7.0`
# resolves to `2.7.0+cpu` wheels.
RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/venv/bin/python -m pip install -r /app/requirements.txt \
        --extra-index-url https://download.pytorch.org/whl/cpu

# Copy the application code last.
COPY . /app

# Runtime directories must be writable by the app user.
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin mcads && \
    mkdir -p /app/logs /app/media /app/staticfiles /app/data_exports /app/backups /app/.torchxrayvision /app/.matplotlib && \
    chmod +x /app/docker-entrypoint.sh && \
    chown -R mcads:mcads /app /opt/uv-python /opt/venv

EXPOSE 8000

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["gunicorn", "--config", "gunicorn_config.py", "mcads_project.asgi:application"]
