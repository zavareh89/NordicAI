#!/usr/bin/env bash
set -euo pipefail

URL="${1:-http://127.0.0.1:9052}"
OUT="${2:-diagnostics/perf_after_validation.json}"
mkdir -p "$(dirname "$OUT")"

ARGS=(-s)
if [[ -n "${SURVIVAL_DIAG_TOKEN:-}" ]]; then
  ARGS+=(-H "x-diagnostic-token: ${SURVIVAL_DIAG_TOKEN}")
fi
curl "${ARGS[@]}" "$URL/diag/perf" > "$OUT"
python -m json.tool "$OUT"
