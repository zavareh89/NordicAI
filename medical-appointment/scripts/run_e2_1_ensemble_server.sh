#!/usr/bin/env bash
set -euo pipefail

export E2_MODE=e2_1_ensemble
export E1_CONFIG="${E1_CONFIG:-config/e1_evidence_tuned.json}"
export E2_1B_CONFIG="${E2_1B_CONFIG:-config/e2_1b_default.json}"
export E2_1_ENSEMBLE_CONFIG="${E2_1_ENSEMBLE_CONFIG:-config/e2_1_ensemble_default.json}"
export E1_DEVICE="${E1_DEVICE:-cuda}"

exec uvicorn api:app --host 0.0.0.0 --port "${PORT:-8000}"
