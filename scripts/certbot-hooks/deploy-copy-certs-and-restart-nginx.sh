#!/usr/bin/env bash
set -euo pipefail

# Deploy hook: run only when a cert is actually renewed.
#
# We copy the renewed cert/key from /etc/letsencrypt into the project-local
# ./ssl/ directory that the Nginx container mounts.
#
# NOTE: This keeps the Docker stack self-contained (it only needs ./ssl),
# while letting Certbot manage the actual issuance/renewal on the host.

DOMAIN="mcads.cloud"
SRC="/etc/letsencrypt/live/${DOMAIN}"
DST="/home/debian/mcads/ssl"

mkdir -p "${DST}"

cp "${SRC}/fullchain.pem" "${DST}/fullchain.pem"
cp "${SRC}/privkey.pem" "${DST}/privkey.pem"

# Keep permissions tight (key readable only by root; nginx master runs as root in container).
chmod 644 "${DST}/fullchain.pem"
chmod 600 "${DST}/privkey.pem"

cd /home/debian/mcads

# Restart Nginx so it picks up the renewed certificate.
sudo docker compose -f docker-compose.yml -f docker-compose.prod.yml restart nginx

