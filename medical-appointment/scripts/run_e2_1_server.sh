python -m unittest \
  tests/test_e2_1b_word_evidence.py \
  tests/test_e2_1b_pipeline_mock.py#!/usr/bin/env bash
set -euo pipefail

# Keep your final/tuned E1 config. E2.1 adds only evidence reranking.
export E2_1_ENABLED=1
export E1_CONFIG="${E1_CONFIG:-config/e1_evidence_tuned.json}"
export E2_1_CONFIG="${E2_1_CONFIG:-config/e2_1_default.json}"
export E1_DEVICE="${E1_DEVICE:-cuda}"

exec uvicorn api:app --host 0.0.0.0 --port "${PORT:-8000}"
