#!/usr/bin/env bash
set -euo pipefail

# Minimal entrypoint for the MCADS Docker stack.
#
# Goals:
# - Keep containers reproducible (no hardcoded credentials).
# - Wait for DB/Redis without requiring extra OS tools (no pg_isready/redis-cli).
# - Run migrations/collectstatic only for the web role (configurable).

role="${MCADS_ROLE:-web}"

echo "MCADS entrypoint starting (role=${role})"

# Ensure runtime directories exist (may be bind-mounted / empty).
mkdir -p /app/logs /app/media /app/staticfiles /app/data_exports /app/backups /app/.torchxrayvision /app/.matplotlib

# If we start as root (recommended for correct volume permissions), fix ownership
# then re-exec as the unprivileged app user.
if [[ "$(id -u)" == "0" && "${MCADS_ENTRYPOINT_REEXEC:-0}" != "1" ]]; then
  chown -R mcads:mcads /app/logs /app/media /app/staticfiles /app/data_exports /app/backups /app/.torchxrayvision /app/.matplotlib
  export MCADS_ENTRYPOINT_REEXEC=1
  exec gosu mcads "$0" "$@"
fi

# Defaults mirror settings.py logic; only required when using Postgres.
export DB_PORT="${DB_PORT:-5432}"
export DB_NAME="${DB_NAME:-mcads_db}"
export DB_USER="${DB_USER:-mcads_user}"

wait_seconds="${MCADS_WAIT_TIMEOUT_SECONDS:-60}"

if [[ -n "${DB_HOST:-}" ]]; then
  echo "Waiting for Postgres (${DB_HOST}:${DB_PORT})..."
  python - <<'PY'
import os, sys, time
import psycopg2

host = os.environ["DB_HOST"]
port = int(os.environ.get("DB_PORT", "5432"))
dbname = os.environ.get("DB_NAME", "mcads_db")
user = os.environ.get("DB_USER", "mcads_user")
password = os.environ.get("DB_PASSWORD", "")
timeout = int(os.environ.get("MCADS_WAIT_TIMEOUT_SECONDS", "60"))

deadline = time.time() + timeout
last_err: Exception | None = None
while time.time() < deadline:
    try:
        conn = psycopg2.connect(host=host, port=port, dbname=dbname, user=user, password=password, connect_timeout=3)
        conn.close()
        print("Postgres is ready.")
        sys.exit(0)
    except Exception as e:  # noqa: BLE001 - best effort readiness loop
        last_err = e
        time.sleep(2)
print(f"Postgres not ready after {timeout}s: {last_err}", file=sys.stderr)
sys.exit(1)
PY
fi

if [[ -n "${REDIS_URL:-}" ]]; then
  echo "Waiting for Redis..."
  python - <<'PY'
import os, sys, time
import redis

url = os.environ["REDIS_URL"]
timeout = int(os.environ.get("MCADS_WAIT_TIMEOUT_SECONDS", "60"))

deadline = time.time() + timeout
last_err: Exception | None = None
client = redis.Redis.from_url(url)
while time.time() < deadline:
    try:
        client.ping()
        print("Redis is ready.")
        sys.exit(0)
    except Exception as e:  # noqa: BLE001 - best effort readiness loop
        last_err = e
        time.sleep(2)
print(f"Redis not ready after {timeout}s: {last_err}", file=sys.stderr)
sys.exit(1)
PY
fi

if [[ "${role}" == "web" ]]; then
  if [[ "${MCADS_RUN_MIGRATIONS:-1}" == "1" ]]; then
    echo "Running migrations..."
    python manage.py migrate --noinput
  fi

  if [[ "${MCADS_COLLECTSTATIC:-1}" == "1" ]]; then
    echo "Collecting static files..."
    python manage.py collectstatic --noinput --clear
  fi
fi

echo "Starting process: $*"
exec "$@"