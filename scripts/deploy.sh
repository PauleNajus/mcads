#!/bin/bash
# MCADS Docker Deployment Script
set -euo pipefail

echo "MCADS Docker Deployment Setup"
echo "================================"

# Check if Docker is installed
if ! command -v docker >/dev/null 2>&1; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Prefer the modern Docker Compose plugin (`docker compose`). Fall back to the legacy binary if present.
if command -v docker-compose >/dev/null 2>&1; then
    COMPOSE=(docker-compose)
elif docker compose version >/dev/null 2>&1; then
    COMPOSE=(docker compose)
else
    echo "❌ Docker Compose is not available. Install Docker Compose plugin or docker-compose."
    exit 1
fi

echo "✅ Docker and Docker Compose are available (${COMPOSE[*]})"

# Speed + correctness:
# - BuildKit is required for our Dockerfile cache mounts (`RUN --mount=type=cache,...`).
# - Enabling these env vars also makes `docker-compose` (legacy) use the Docker CLI builder.
export DOCKER_BUILDKIT="${DOCKER_BUILDKIT:-1}"
export COMPOSE_DOCKER_CLI_BUILD="${COMPOSE_DOCKER_CLI_BUILD:-1}"

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p logs media staticfiles data_exports backups

# Set permissions
echo "Setting proper permissions..."
chmod 755 logs media staticfiles data_exports backups
chmod +x scripts/*.sh docker-entrypoint.sh

# Compose file selection:
# - Always use docker-compose.yml
# - If TLS certs exist, automatically enable docker-compose.prod.yml (binds 443 + TLS config)
COMPOSE_FILES=(-f docker-compose.yml)
TLS_ENABLED=0
if [[ -f docker-compose.prod.yml && -f ssl/fullchain.pem && -f ssl/privkey.pem ]]; then
    COMPOSE_FILES+=(-f docker-compose.prod.yml)
    TLS_ENABLED=1
    echo "TLS detected (ssl/fullchain.pem + ssl/privkey.pem). Enabling production TLS override."
else
    echo "TLS certs not found in ./ssl. Starting HTTP-only on port 80."
    echo "NOTE: If DEBUG=0, Django redirects HTTP -> HTTPS; provide TLS certs or set DEBUG=1 for HTTP-only."
fi

# Determine the health endpoint (used for fast readiness waiting below).
if [[ "${TLS_ENABLED}" == "1" ]]; then
    # Cert is usually for the real domain, not "localhost", so skip verification here.
    HEALTH_URL="https://localhost/health/"
    CURL_FLAGS=(-fsSk)
else
    HEALTH_URL="http://localhost/health/"
    CURL_FLAGS=(-fsS)
fi

# Build (if needed) and start services.
# `--build` ensures the running containers pick up fresh code changes.
# `--remove-orphans` avoids stale services after compose file edits.
echo "Building (if needed) and starting services..."
"${COMPOSE[@]}" "${COMPOSE_FILES[@]}" up -d --build --remove-orphans

# Nginx reads config only on start/reload. Since our config is mounted from the host
# (and Compose doesn't detect content-only changes), explicitly restart Nginx so
# updates like `client_max_body_size` take effect.
#
# Prefer a zero-downtime reload; fall back to a restart if reload isn't supported.
nginx_cid="$("${COMPOSE[@]}" "${COMPOSE_FILES[@]}" ps -q nginx 2>/dev/null | head -n1 || true)"
if [[ -n "${nginx_cid}" ]]; then
    echo "Reloading nginx to apply config..."
    "${COMPOSE[@]}" "${COMPOSE_FILES[@]}" exec -T nginx nginx -s reload \
      || "${COMPOSE[@]}" "${COMPOSE_FILES[@]}" restart nginx
else
    echo "Nginx container not created yet; skipping early reload."
fi

# Wait for services to be ready (faster than a fixed sleep; returns as soon as ready).
echo "Waiting for application to become ready..."
HEALTH_TIMEOUT_SECONDS="${MCADS_DEPLOY_HEALTH_TIMEOUT_SECONDS:-120}"
HEALTH_INTERVAL_SECONDS="${MCADS_DEPLOY_HEALTH_INTERVAL_SECONDS:-2}"
health_deadline=$((SECONDS + HEALTH_TIMEOUT_SECONDS))
until curl "${CURL_FLAGS[@]}" "${HEALTH_URL}" > /dev/null 2>&1; do
    if (( SECONDS >= health_deadline )); then
        echo "⚠️  Application health check timed out after ${HEALTH_TIMEOUT_SECONDS}s - checking logs..."
        "${COMPOSE[@]}" "${COMPOSE_FILES[@]}" logs --tail=50 web || true
        "${COMPOSE[@]}" "${COMPOSE_FILES[@]}" logs --tail=50 nginx || true
        break
    fi
    sleep "${HEALTH_INTERVAL_SECONDS}"
done

# Check service health
echo "Checking service health..."

services=("db" "redis" "web" "nginx")
for service in "${services[@]}"; do
    if "${COMPOSE[@]}" "${COMPOSE_FILES[@]}" ps | grep -q "${service}.*Up"; then
        echo "✅ $service is running"
    else
        echo "❌ $service is not running"
    fi
done

# Test application response
echo "Testing application..."
if curl "${CURL_FLAGS[@]}" "${HEALTH_URL}" > /dev/null 2>&1; then
    echo "✅ Application is responding"
else
    echo "⚠️  Application health check failed - checking logs..."
    "${COMPOSE[@]}" "${COMPOSE_FILES[@]}" logs --tail=20 web
fi

echo ""
echo "MCADS Docker deployment completed!"
echo ""
echo "📋 Access Information:"
if [[ "${TLS_ENABLED}" == "1" ]]; then
    echo "   Web Application: https://<your-domain>"
else
    echo "   Web Application: http://<your-domain-or-ip>"
fi
echo "   Admin Panel: /secure-admin-mcads-2024/"
echo ""
echo "📚 Management Commands:"
echo "   📁 Create backup: ./scripts/backup.sh"
echo "   📁 Restore backup: ./scripts/restore.sh <backup_name>"
echo "   🔍 View logs: ${COMPOSE[*]} ${COMPOSE_FILES[*]} logs"
echo "   🛑 Stop services: ${COMPOSE[*]} ${COMPOSE_FILES[*]} down"
echo "   🔄 Restart services: ${COMPOSE[*]} ${COMPOSE_FILES[*]} restart"