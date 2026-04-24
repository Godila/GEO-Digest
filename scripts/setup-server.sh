#!/bin/bash
# GEO-Digest: Server Setup Script
# Run once on fresh VPS to prepare for Docker deployment.
#
# Usage:
#   sudo bash scripts/setup-server.sh
#
set -euo pipefail

echo "═══ GEO-Digest Server Setup ═══"
echo ""

# ── 1. Swap file ────────────────────────────────
SWAP_SIZE_GB=${1:-4}
SWAPFILE="/swapfile"

if [ -f "$SWAPFILE" ] && swapon --show | grep -q "$SWAPFILE"; then
    echo "✅ Swap already active:"
    free -h | grep Swap
else
    echo "[$] Creating ${SWAP_SIZE_GB}GB swap file..."
    if [ ! -f "$SWAPFILE" ]; then
        fallocate -l "${SWAP_SIZE_GB}G" "$SWAPFILE"
        chmod 600 "$SWAPFILE"
        mkswap "$SWAPFILE"
    fi
    swapon "$SWAPFILE"

    # Persist in fstab
    if ! grep -q "$SWAPFILE" /etc/fstab; then
        echo "$SWAPFILE none swap sw 0 0" >> /etc/fstab
    fi

    echo "✅ Swap enabled:"
    free -h | grep Swap
fi

# ── 2. Swappiness ────────────────────────────────
echo ""
echo "[$] Setting vm.swappiness=10..."
sysctl -w vm.swappiness=10 > /dev/null

if ! grep -q "vm.swappiness" /etc/sysctl.conf; then
    echo "vm.swappiness = 10" >> /etc/sysctl.conf
else
    sed -i 's/vm.swappiness.*/vm.swappiness = 10/' /etc/sysctl.conf
fi

echo "✅ Swappiness set to 10 (prefer RAM, use swap under memory pressure)"

# ── 3. Docker ───────────────────────────────────
if ! command -v docker &>/dev/null; then
    echo ""
    echo "[$] Installing Docker..."
    curl -fsSL https://get.docker.com | sh
    systemctl enable docker
    systemctl start docker
    echo "✅ Docker installed"
else
    echo "✅ Docker already installed: $(docker --version)"
fi

# ── 4. Summary ──────────────────────────────────
echo ""
echo "═══ SUMMARY ═══"
free -h
echo ""
echo "Swap file: $SWAPFILE (${SWAP_SIZE_GB}GB)"
echo "Swappiness: $(cat /proc/sys/vm/swappiness)"
echo ""
echo "Next step: cd geo-digest && docker compose up --build -d"
