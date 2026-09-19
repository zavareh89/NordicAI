#!/usr/bin/env bash
set -euo pipefail

# One worker is intentional: SurvivalController contains per-agent state that
# must remain in one process for the full episode.
export LOG_LEVEL="${LOG_LEVEL:-warning}"
export SURVIVAL_AGENT_HOST="${SURVIVAL_AGENT_HOST:-0.0.0.0}"
export SURVIVAL_AGENT_PORT="${SURVIVAL_AGENT_PORT:-9052}"

exec uvicorn agent_server_fast:app \
  --host "$SURVIVAL_AGENT_HOST" \
  --port "$SURVIVAL_AGENT_PORT" \
  --workers 1 \
  --loop uvloop \
  --http httptools \
  --no-access-log \
  --log-level "$LOG_LEVEL" \
  --timeout-keep-alive 75 \
  --backlog 64
