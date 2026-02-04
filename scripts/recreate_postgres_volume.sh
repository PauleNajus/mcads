#!/bin/bash
# Recreate the Postgres data volume (major version upgrades)
#
# When upgrading PostgreSQL across *major* versions (e.g. 15 -> 18), the on-disk
# data directory format changes. The official `postgres` Docker image will refuse
# to start if the existing volume was initialized by an older major version.
#
# This script deletes ONLY the Postgres named volume from this Compose project
# (it does NOT touch media/static/backups/etc.), then starts a fresh Postgres
# instance that will initialize a new empty cluster.
#
# WARNING: This permanently deletes ALL database data in the volume.
set -euo pipefail

# Prefer the modern Docker Compose plugin (`docker compose`). Fall back to legacy `docker-compose` if present.
if command -v docker-compose >/dev/null 2>&1; then
  COMPOSE=(docker-compose)
elif docker compose version >/dev/null 2>&1; then
  COMPOSE=(docker compose)
else
  echo "❌ Docker Compose is not available."
  exit 1
fi

# Compose file selection:
# - Always use docker-compose.yml
# - If TLS certs exist, enable docker-compose.prod.yml (binds 443 + TLS config)
COMPOSE_FILES=(-f docker-compose.yml)
if [[ -f docker-compose.prod.yml && -f ssl/fullchain.pem && -f ssl/privkey.pem ]]; then
  COMPOSE_FILES+=(-f docker-compose.prod.yml)
fi

# Load local env overrides if present (kept out of git).
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

PROJECT_NAME="${COMPOSE_PROJECT_NAME:-$(basename "$(pwd)")}"
VOLUME_NAME="${PROJECT_NAME}_postgres_data"

echo "⚠️  This will DELETE the Docker volume '${VOLUME_NAME}' (ALL DB DATA)."
echo "Stopping app services that use the DB..."

# Avoid noisy errors from services that poll the DB (celery beat, web, workers)
# while the DB is being recreated.
"${COMPOSE[@]}" "${COMPOSE_FILES[@]}" stop web celery-worker celery-beat >/dev/null 2>&1 || true

echo "Stopping and removing the DB container (if any)..."

# Stop/remove only the db service container (does not touch other services/volumes).
"${COMPOSE[@]}" "${COMPOSE_FILES[@]}" stop db >/dev/null 2>&1 || true
"${COMPOSE[@]}" "${COMPOSE_FILES[@]}" rm -f db >/dev/null 2>&1 || true

echo "Removing volume: ${VOLUME_NAME}"
docker volume rm "${VOLUME_NAME}"

echo "Starting a fresh Postgres container (will initialize a new cluster)..."
"${COMPOSE[@]}" "${COMPOSE_FILES[@]}" up -d db

echo "Waiting for Postgres health check..."
HEALTH_TIMEOUT_SECONDS="${MCADS_PG_HEALTH_TIMEOUT_SECONDS:-120}"
deadline=$((SECONDS + HEALTH_TIMEOUT_SECONDS))

db_cid="$("${COMPOSE[@]}" "${COMPOSE_FILES[@]}" ps -q db 2>/dev/null | head -n1 || true)"
if [[ -z "${db_cid}" ]]; then
  echo "❌ Could not resolve DB container id."
  exit 1
fi

while true; do
  status="$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' "${db_cid}" 2>/dev/null || echo "unknown")"
  if [[ "${status}" == "healthy" ]]; then
    break
  fi
  if [[ "${status}" == "unhealthy" ]]; then
    echo "❌ Postgres is unhealthy. Recent logs:"
    docker logs --tail=200 "${db_cid}" || true
    exit 1
  fi
  if (( SECONDS >= deadline )); then
    echo "❌ Timed out waiting for Postgres to become healthy after ${HEALTH_TIMEOUT_SECONDS}s. Recent logs:"
    docker logs --tail=200 "${db_cid}" || true
    exit 1
  fi
  sleep 2
done

echo "✅ Postgres volume recreated."
echo "Next, recreate tables by running:"
echo "  ${COMPOSE[*]} ${COMPOSE_FILES[*]} exec -T web python manage.py migrate --noinput"

