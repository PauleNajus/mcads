#!/usr/bin/env bash
set -euo pipefail

# Certbot standalone uses port 80 for HTTP-01 challenges.
# Our production Nginx container also binds port 80, so we must stop it
# before Certbot starts its temporary standalone server.

cd /home/debian/mcads

# Don't fail if it's already stopped.
sudo docker compose -f docker-compose.yml -f docker-compose.prod.yml stop nginx || true

