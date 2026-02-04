#!/bin/bash
# MCADS Database Reset Script
#
# Deletes ALL PostgreSQL data by recreating the `public` schema.
#
# Why this approach?
# - Fast: avoids deleting the Docker volume (no re-init cost).
# - Safe: does NOT touch other named volumes (media/static/backups/etc.).
#
# For major PostgreSQL upgrades (e.g. 15 -> 18), you must recreate the data
# volume instead (old data dirs are incompatible). Use:
#   ./scripts/recreate_postgres_volume.sh
#
# After running, re-apply Django migrations to recreate tables.
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

# Resolve DB identity with sensible defaults (matches docker-compose.yml).
DB_NAME="${DB_NAME:-${POSTGRES_DB:-mcads_db}}"
DB_USER="${DB_USER:-${POSTGRES_USER:-mcads_user}}"

echo "⚠️  This will DELETE ALL DATA in database '${DB_NAME}'."
echo "Resetting PostgreSQL schema..."

# Avoid noisy errors from services that poll the DB (celery beat, web, workers)
# while the schema is being wiped.
"${COMPOSE[@]}" "${COMPOSE_FILES[@]}" stop web celery-worker celery-beat >/dev/null 2>&1 || true

# Ensure the DB container is running.
"${COMPOSE[@]}" "${COMPOSE_FILES[@]}" up -d db

# Drop and recreate the schema (clears all tables, including Django migration history).
"${COMPOSE[@]}" "${COMPOSE_FILES[@]}" exec -T db psql -v ON_ERROR_STOP=1 -U "${DB_USER}" -d "${DB_NAME}" -c \
  "DROP SCHEMA public CASCADE; CREATE SCHEMA public; GRANT ALL ON SCHEMA public TO ${DB_USER}; GRANT ALL ON SCHEMA public TO public;"

echo "✅ Database cleared."
echo "Recreate tables by running:"
echo "  ${COMPOSE[*]} ${COMPOSE_FILES[*]} exec -T web python manage.py migrate --noinput"

