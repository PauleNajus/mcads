#!/usr/bin/env bash
set -euo pipefail

# After Certbot finishes (success or failure), ensure the Nginx container is running.

cd /home/debian/mcads

sudo docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d nginx

