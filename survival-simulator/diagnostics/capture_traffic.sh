#!/usr/bin/env bash
set -euo pipefail

PORT="${SURVIVAL_AGENT_PORT:-9052}"
OUT="${1:-diagnostics/validation_9052.pcap}"
mkdir -p "$(dirname "$OUT")"

echo "Capturing TCP port $PORT to $OUT"
echo "Stop with Ctrl-C after validation."
exec sudo tcpdump -i any -nn -s 160 -w "$OUT" "tcp port $PORT"
