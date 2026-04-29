#!/bin/bash
# GEO-Digest: Setup Docker Access for Hermes Agent
# Run this ONCE on the HOST as root:
#   bash /root/.hermes/geo_digest/setup-docker-access.sh
#
# What it does:
# 1. Adds docker socket mount to Hermes WebUI compose
# 2. Restarts Hermes WebUI container with new mount
# 3. Verifies docker access from inside the container

set -e

HERMES_COMPOSE="/root/hermes-webui/docker-compose.yml"

echo "=== GEO-Digest Docker Access Setup ==="

# 1. Patch Hermes WebUI compose — add docker socket mount
if grep -q "docker.sock" "$HERMES_COMPOSE" 2>/dev/null; then
    echo "[OK] docker.sock already in $HERMES_COMPOSE"
else
    echo "[PATCH] Adding docker.sock mount to $HERMES_COMPOSE"
    sed -i '/ \/root\/workspace:\/root\/workspace/a\      # Docker socket — access to docker CLI for managing geo-digest containers\n      - /var/run/docker.sock:/var/run/docker.sock' "$HERMES_COMPOSE"
    echo "[OK] Patched"
fi

# 2. Restart Hermes WebUI
echo "[RESTART] Restarting Hermes WebUI container..."
cd /root/hermes-webui
docker compose down
docker compose up -d

# 3. Wait for container to start
echo "[WAIT] Waiting for container to start..."
sleep 5

# 4. Verify
echo "[VERIFY] Checking docker access..."
docker exec $(docker ps -q --filter "name=hermes") docker ps --format "table {{.Names}}\t{{.Status}}" 2>/dev/null && \
    echo "[OK] Docker access works!" || \
    echo "[WARN] Could not verify docker access — check manually"

echo ""
echo "=== Done! Hermes Agent now has docker access ==="
echo "Available containers:"
docker ps --format "  {{.Names}}\t{{.Status}}\t{{.Ports}}"
