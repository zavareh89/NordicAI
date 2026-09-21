# E2.1 evidence-localization improvement roadmap

This package starts from your **current E2.1 baseline**:

- accuracy: **0.910**
- positives: **170 / 195**
- hard negatives: **133 / 142**
- off-topic: **52 / 53**
- mean tIoU over all positives: **0.446**
- tIoU when answered YES: **0.511**
- final score: **0.632**
- worst observed round trip: **13.718 s**

The classification path is deliberately frozen throughout these steps. The goal is to increase tIoU for the 170 positives already detected without changing the E1/E2.1 YES/NO decisions.

The stages are incremental. Apply and evaluate them in order.

---

## Overview

```text
Current E2.1
   |
   v
Step 1: E2.1-B direct word-index LLM selection
   |
   v
Step 2: cross-validated boundary calibration
   |
   v
Step 3: lightweight learned proposal ranker
   |
   v
Step 4: optional LLM + ranker ensemble
```

The main comparison metric is always:

```text
tIoU when answered YES
```

Because classification is frozen, accuracy should remain **0.910** in every evidence-only experiment.

---

# Before starting

You should already have the current E1/E2.1 code and these files from the previous stages:

```text
config/e1_evidence_tuned.json
config/e2_1_default.json

e1/pipeline.py
e1/evidence.py
e1/schemas.py
e1/e2_1_pipeline.py
e1/llm_evidence.py
```

You should also already have the Qwen evidence model downloaded:

```text
Qwen/Qwen3-4B-Instruct-2507
```

No new model weights are required by Steps 1-4.

Keep your final E1 config fixed while comparing the evidence experiments:

```bash
export E1_CONFIG=config/e1_evidence_tuned.json
```

---

# Step 1 - E2.1-B: direct word-index selection

## Why

Current E2.1 forces Qwen to choose one pre-generated proposal. The oracle analysis showed substantial ranking headroom, and an ideal annotation boundary can fall between two proposals.

E2.1-B gives Qwen the local selected event as indexed words:

```text
[142] I
[143] would
[144] like
[145] you
[146] to
[147] start
[148] metformin
[149] 500
[150] mg
[151] twice
[152] daily
```

The model returns only:

```json
{
  "0": {
    "source": "medasr",
    "start_word": 142,
    "end_word": 152
  }
}
```

It never generates timestamps. Code converts word indices back to timestamps.

## Files to apply

From:

```text
step1_e2_1b_word_indices/APPLY_THESE_FILES/
```

copy:

```text
config/e2_1b_default.json
e1/e2_1b_config.py
e1/llm_word_evidence.py
e1/e2_1b_pipeline.py
example.py
scripts/run_e2_1b_server.sh
tests/test_e2_1b_word_evidence.py
tests/test_e2_1b_pipeline_mock.py
```

## Tests

```bash
python -m unittest \
  tests/test_e2_1b_word_evidence.py \
  tests/test_e2_1b_pipeline_mock.py
```

Then rerun the existing E1/E2 tests if desired:

```bash
python -m unittest \
  tests/test_e1_logic.py \
  tests/test_e1_pipeline_mock.py \
  tests/test_e1_evidence_candidates.py \
  tests/test_e2_1_llm_evidence.py
```

## Run E2.1-B

```bash
export E2_MODE=e2_1b
export E1_CONFIG=config/e1_evidence_tuned.json
export E2_1B_CONFIG=config/e2_1b_default.json
python api.py
```

or:

```bash
bash scripts/run_e2_1b_server.sh
```

Then in another shell:

```bash
python local_evaluator.py --verbose
```

## What must remain unchanged

```text
questions/correct classification counts
positive count
hard-negative count
off-topic count
accuracy
```

In particular, target:

```text
accuracy = 0.910
positives = 170 / 195
```

If classification changes, stop and inspect the integration before continuing.

## Decision after Step 1

Compare:

```text
Current E2.1 tIoU when YES: 0.511
E2.1-B tIoU when YES:      ?
```

If E2.1-B improves tIoU, use it as the main LLM boundary system.

If it is slightly worse, do **not** discard the rest of the roadmap. Step 3's learned ranker is independent and may still beat both systems.

---

# Step 2 - CV-tuned boundary calibration

## Why

Even a semantically correct word span may systematically start a little late or end a little early relative to the annotation style.

Step 2 learns small shifts using conversation-level cross-validation:

```text
calibrated_start = raw_start + start_shift
calibrated_end   = raw_end   + end_shift
```

Examples:

```text
start_shift = -0.15  -> expand 150 ms to the left
end_shift   = +0.10  -> expand 100 ms to the right
```

The tuner allows both expansion and shrinking.

## Files to apply

From:

```text
step2_boundary_calibration/APPLY_THESE_FILES/
```

copy:

```text
config/e2_1_boundary_calibration.json
e1/boundary_calibration.py
e1/e2_1c_pipeline.py
dev/boundary_tuning.py
dev/generate_e2_1b_boundary_cache.py
dev/tune_boundary_calibration.py
example.py
scripts/run_e2_1c_server.sh
tests/test_boundary_calibration.py
```

## Test

```bash
python -m unittest tests/test_boundary_calibration.py
```

## Generate raw E2.1-B boundary cache

This is an **offline training/development cache**. Do not ship it with the submission.

```bash
python -m dev.generate_e2_1b_boundary_cache \
  --e1-config config/e1_evidence_tuned.json \
  --e2-config config/e2_1b_default.json \
  --output dev_cache/e2_1b_boundary_cache.json.gz
```

The cache generation temporarily uses zero padding so it records the raw ASR word boundaries.

## Tune calibration

```bash
python -m dev.tune_boundary_calibration \
  --cache dev_cache/e2_1b_boundary_cache.json.gz \
  --output config/e2_1_boundary_calibration.json \
  --folds 5
```

Inspect:

```text
raw_mean_tiou
cv_mean_tiou
fit_mean_tiou
global shifts
per-source shifts
```

The **cross-validated** value matters more than the fit value.

## Run calibrated E2.1-C

```bash
export E2_MODE=e2_1c
export E1_CONFIG=config/e1_evidence_tuned.json
export E2_1B_CONFIG=config/e2_1b_default.json
export E2_1_CALIBRATION_CONFIG=config/e2_1_boundary_calibration.json
python api.py
```

or:

```bash
bash scripts/run_e2_1c_server.sh
```

Then:

```bash
python local_evaluator.py --verbose
```

## Decision after Step 2

Keep calibration if both are true:

1. conversation-level CV improves tIoU;
2. the full evaluator confirms the improvement.

If CV gain is approximately zero or negative, use identity shifts and continue to Step 3.

---

# Step 3 - lightweight learned proposal ranker

## Why

The earlier oracle experiment showed the proposal set often already contains a much better span than the handcrafted evidence score chooses.

This step trains a very small **ridge regression ranker** to predict proposal tIoU from features such as:

```text
NLI entailment / neutral / contradiction
exact-fact coverage
topic coverage
proposal word count and duration
proposal family
MedASR vs Parakeet
parent retrieval score
parent E1 yes/no score
parent relevance
pause threshold
```

No new neural model or dependency is required. It uses NumPy only.

## Files to apply

From:

```text
step3_learned_ranker/APPLY_THESE_FILES/
```

copy:

```text
config/e2_1_ranker_default.json
e1/e2_1_ranker_config.py
e1/evidence_ranker.py
e1/e2_1_ranker_pipeline.py
dev/ranker_training.py
dev/train_evidence_ranker.py
example.py
scripts/run_e2_1_ranker_server.sh
tests/test_evidence_ranker.py
```

## Test

```bash
python -m unittest tests/test_evidence_ranker.py
```

## Important: regenerate the E1 dev cache if needed

The ranker uses the proposal cache produced by the previous E1 development tools.
It must reflect your **final Stage-4/5 proposal generator and final E1 config**.

If you are unsure, regenerate it:

```bash
python -m dev.generate_dev_cache \
  --config config/e1_evidence_tuned.json \
  --output dev_cache/e1_dev_cache.json.gz
```

## Train with conversation-level CV

```bash
python -m dev.train_evidence_ranker \
  --cache dev_cache/e1_dev_cache.json.gz \
  --calibration config/e2_1_boundary_calibration.json \
  --output config/e2_1_evidence_ranker.json \
  --folds 5
```

The script reports:

```text
selected ridge alpha
CV mean tIoU
training-fit mean tIoU
proposal oracle tIoU
```

The ranker JSON is tiny and is the only learned artifact needed at runtime.

## Run ranker-only evidence system

```bash
export E2_MODE=e2_1_ranker
export E1_CONFIG=config/e1_evidence_tuned.json
export E2_1_RANKER_CONFIG=config/e2_1_ranker_default.json
python api.py
```

or:

```bash
bash scripts/run_e2_1_ranker_server.sh
```

Then:

```bash
python local_evaluator.py --verbose
```

## Compare three systems

Record:

```text
                    accuracy    tIoU YES    mean tIoU    score    worst ms
E2.1 proposal LLM     0.910       0.511       0.446      0.632     13718
E2.1-B word LLM       0.910       ?           ?          ?         ?
E2.1-C + calibration  0.910       ?           ?          ?         ?
Ranker + calibration  0.910       ?           ?          ?         ?
```

The ranker may be useful even if its tIoU is slightly lower because it should be much faster than a Qwen generation.

---

# Step 4 - optional LLM + ranker ensemble

## Why

The two approaches have different strengths:

- the learned ranker knows which proposal structures historically align with annotations;
- the LLM can choose a boundary between predefined proposals.

The ensemble uses the learned ranker as a conservative structural reference.
It keeps the direct LLM boundary when:

1. LLM and ranker spans overlap substantially; or
2. the LLM span has materially stronger NLI entailment.

Otherwise it uses the learned ranker proposal.

The final span is then boundary-calibrated.

## Files to apply

From:

```text
step4_ensemble/APPLY_THESE_FILES/
```

copy:

```text
config/e2_1_ensemble_default.json
e1/e2_1_ensemble_config.py
e1/e2_1_ensemble_pipeline.py
example.py
scripts/run_e2_1_ensemble_server.sh
tests/test_e2_1_ensemble.py
```

This step assumes these already exist:

```text
config/e2_1_evidence_ranker.json
config/e2_1_boundary_calibration.json
config/e2_1b_default.json
```

## Test

```bash
python -m unittest tests/test_e2_1_ensemble.py
```

## Run ensemble

```bash
export E2_MODE=e2_1_ensemble
export E1_CONFIG=config/e1_evidence_tuned.json
export E2_1B_CONFIG=config/e2_1b_default.json
export E2_1_ENSEMBLE_CONFIG=config/e2_1_ensemble_default.json
python api.py
```

or:

```bash
bash scripts/run_e2_1_ensemble_server.sh
```

Then:

```bash
python local_evaluator.py --verbose
```

---

# Final selection rule

Do not assume the most complex system is best.

Choose the runtime configuration with the best **validated challenge score** while keeping enough timeout margin.

Suggested acceptance priorities:

1. accuracy remains exactly at the frozen E1 value;
2. tIoU when YES improves;
3. mean tIoU improves;
4. final score improves;
5. worst-case latency remains comfortably below 60 s.

With your present baseline, a useful target is:

```text
tIoU when YES >= 0.56
```

A strong target is:

```text
tIoU when YES >= 0.60
```

At unchanged classification, every evidence improvement goes almost directly into the challenge score.

---

# Recommended experiment order

Run exactly this sequence:

```text
A. Current E2.1 proposal selector             baseline = 0.632
B. E2.1-B direct word-index selector
C. E2.1-C direct word selector + calibration
D. learned proposal ranker + calibration
E. LLM + ranker ensemble + calibration
```

Do not tune classification thresholds during these experiments. That would make it impossible to attribute score changes to evidence localization.

---

# Full test set after all files are applied

```bash
python -m unittest \
  tests/test_e1_logic.py \
  tests/test_e1_pipeline_mock.py \
  tests/test_e1_evidence_candidates.py \
  tests/test_e2_1_llm_evidence.py \
  tests/test_e2_1_pipeline_mock.py \
  tests/test_e2_1b_word_evidence.py \
  tests/test_e2_1b_pipeline_mock.py \
  tests/test_boundary_calibration.py \
  tests/test_evidence_ranker.py \
  tests/test_e2_1_ensemble.py
```

I did not execute these tests or model inference while creating this package.
