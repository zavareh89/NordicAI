#!/usr/bin/env bash
set -euo pipefail

HOST="${SURVIVAL_AGENT_HOST:-0.0.0.0}"
PORT="${SURVIVAL_AGENT_PORT:-9052}"

mkdir -p diagnostics

exec .venv/bin/python -m uvicorn diagnostic_agent_server:app \
  --host "$HOST" \
  --port "$PORT" \
  --workers 1 \
  --no-access-log \
  --log-level warning
