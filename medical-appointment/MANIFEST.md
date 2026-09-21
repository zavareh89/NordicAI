# Patch manifest

## Step 1 - direct word-index selection

- `config/e2_1b_default.json`
- `e1/e2_1b_config.py`
- `e1/llm_word_evidence.py`
- `e1/e2_1b_pipeline.py`
- `example.py`
- `scripts/run_e2_1b_server.sh`
- `tests/test_e2_1b_word_evidence.py`
- `tests/test_e2_1b_pipeline_mock.py`

## Step 2 - boundary calibration

- `config/e2_1_boundary_calibration.json`
- `e1/boundary_calibration.py`
- `e1/e2_1c_pipeline.py`
- `dev/boundary_tuning.py`
- `dev/generate_e2_1b_boundary_cache.py`
- `dev/tune_boundary_calibration.py`
- `example.py`
- `scripts/run_e2_1c_server.sh`
- `tests/test_boundary_calibration.py`

## Step 3 - learned proposal ranker

- `config/e2_1_ranker_default.json`
- `e1/e2_1_ranker_config.py`
- `e1/evidence_ranker.py`
- `e1/e2_1_ranker_pipeline.py`
- `dev/ranker_training.py`
- `dev/train_evidence_ranker.py`
- `example.py`
- `scripts/run_e2_1_ranker_server.sh`
- `tests/test_evidence_ranker.py`

## Step 4 - ensemble

- `config/e2_1_ensemble_default.json`
- `e1/e2_1_ensemble_config.py`
- `e1/e2_1_ensemble_pipeline.py`
- `example.py`
- `scripts/run_e2_1_ensemble_server.sh`
- `tests/test_e2_1_ensemble.py`

`FINAL_ALL_UPDATED_FILES/` contains the union of all four stages.
