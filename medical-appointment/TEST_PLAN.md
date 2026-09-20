# Dual-ASR E1 test plan

The package contains tests but they were intentionally **not executed during package creation**, per the request not to execute solution/test code in the assistant environment.

## Pure-logic tests (`tests/test_e1_logic.py`)

Covers:

1. English number-word normalization (`five hundred` -> `500`).
2. Cross-unit dose canonicalization (`0.5 g` equals `500 mg`).
3. Deterministic hard-negative contradiction (`200 mg` vs `100 mg`).
4. Retrieval of the correct medical event even when the queried numeric value is deliberately wrong.
5. Cross-ASR temporal alignment.
6. Fact guard overriding a falsely high NLI entailment score.
7. Dual-ASR YES consensus.
8. Evidence tightening remaining below the configured maximum span size.

## Mock end-to-end tests (`tests/test_e1_pipeline_mock.py`)

Uses fake ASR and fake NLI components; no weights or GPU are required. Covers:

1. Positive question with agreement from both ASRs.
2. Hard negative where mock NLI is deliberately fooled but the exact-value guard must return NO.
3. ASR disagreement (`50 mg` vs `500 mg`) where the exact-fact supporting transcript should be selected for YES evidence.

## Server validation to run on the A10

After the unit tests:

```bash
python api.py
```

In a second shell:

```bash
python local_evaluator.py --verbose
```

Record at least:

- total score;
- positive accuracy;
- hard-negative accuracy;
- off-topic accuracy;
- mean tIoU;
- missing evidence spans;
- mean/worst conversation latency.

The next iteration should tune only `config/e1_default.json` until a model/design change is intentionally introduced.
