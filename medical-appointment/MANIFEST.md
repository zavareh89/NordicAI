# File replacement manifest

## Step 1 — threshold search
Add:
- `dev/__init__.py`
- `dev/common.py`
- `dev/generate_dev_cache.py`
- `dev/threshold_logic.py`
- `dev/threshold_search.py`
- `tests/test_dev_threshold_logic.py`

## Step 2 — false-negative breakdown
Add:
- `dev/fn_breakdown.py`

## Step 3 — evidence oracle
Add:
- `dev/evidence_oracle.py`
- `tests/test_evidence_oracle.py`

## Step 4 — richer evidence candidates
Replace:
- `config/e1_default.json`
- `e1/config.py`
- `e1/schemas.py`
- `e1/evidence.py`

Add:
- `tests/test_e1_evidence_candidates.py`

## Step 5 — evidence CV tuning
Add:
- `dev/evidence_cv_tune.py`
- `tests/test_evidence_cv_tune.py`

No model packages or requirements files change.
