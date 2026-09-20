# Dual-ASR E1 — staged improvement plan

This package is designed to be applied **on top of the Dual-ASR E1 v2 code**
(the version that scored about **0.538** on the supplied 390 questions).

The work is intentionally split into five stages. Do **not** copy every folder
into the project at once if you want clean attribution of gains. Apply one stage,
run its diagnostics/evaluator, record the result, then continue.

> The files in this package were written but **not executed or tested in the
> authoring environment**. Run the listed unit tests and smoke commands on your
> A10 environment before trusting each stage.

## Directory layout

Each stage contains an `APPLY_THESE_FILES/` directory. Copy the contents of that
directory into the root of `medical-appointment/`, preserving relative paths.

```text
step1_threshold_search/APPLY_THESE_FILES/dev/...    -> medical-appointment/dev/...
step4_evidence_candidates/APPLY_THESE_FILES/e1/... -> medical-appointment/e1/...
```

No model/runtime dependency changes are required for these stages.

---

# Stage 1 — threshold search optimized for the official score

## Why

The current model has high YES precision and lower YES recall. Manually lowering
one threshold is risky because the consensus contains several different paths.
Stage 1 makes threshold tuning model-free and evaluates the exact challenge
objective:

```text
score = 0.4 * accuracy + 0.6 * mean positive tIoU
```

It first creates a reusable cache containing:

- every retrieved temporal event;
- MedASR/Parakeet NLI scores;
- deterministic fact matches/mismatches;
- every evidence proposal and its NLI score;
- gold labels/evidence for offline development.

After the cache exists, threshold sweeps require **no ASR or NLI inference**.

## Files to add

```text
dev/__init__.py
dev/common.py
dev/generate_dev_cache.py
dev/threshold_logic.py
dev/threshold_search.py
tests/test_dev_threshold_logic.py
```

No production E1 file is replaced in this stage.

## Apply

```bash
cp -r step1_threshold_search/APPLY_THESE_FILES/* .
```

## Test

```bash
python -m unittest tests/test_dev_threshold_logic.py
```

## Generate the expensive cache once

Use the config that produced your current 0.538 result:

```bash
python -m dev.generate_dev_cache \
  --config config/e1_default.json \
  --device cuda \
  --output dev_cache/e1_dev_cache.json.gz
```

Optional two-conversation smoke run:

```bash
python dev/generate_dev_cache.py \
  --config config/e1_default.json \
  --device cuda \
  --limit-conversations 2 \
  --output dev_cache/e1_dev_cache_smoke.json.gz
```

The full cache is development-only and can be large because it stores all
counterfactual evidence proposals. Do not include it in the competition image.

## Search thresholds

```bash
python -m dev.threshold_search \
  --cache dev_cache/e1_dev_cache.json.gz \
  --config config/e1_default.json \
  --folds 5 \
  --passes 3 \
  --output-dir dev_results/threshold_search \
  --write-config config/e1_threshold_tuned.json
```

Outputs:

```text
dev_results/threshold_search/summary.json
dev_results/threshold_search/cv_folds.csv
dev_results/threshold_search/search_history.csv
config/e1_threshold_tuned.json
```

### What number to trust

Use the **cross-validated mean score** to judge whether calibration generalizes.
The full-training-set score is optimistic. `e1_threshold_tuned.json` is the
candidate config to try in the actual local evaluator.

## Validate on the endpoint

```bash
E1_CONFIG=config/e1_threshold_tuned.json python api.py
```

Then in another shell:

```bash
python local_evaluator.py --verbose
```

Record:

- total score;
- positive accuracy;
- hard-negative accuracy;
- off-topic accuracy;
- mean tIoU;
- tIoU when answered YES;
- worst latency.

### Proceed to Stage 2 when

The cache and threshold replay complete successfully. Even if the tuned config
is not better, keep the cache; Stages 2 and 3 use it to identify why.

---

# Stage 2 — false-negative stage breakdown

## Why

The remaining positive errors should not all be treated as the same problem.
This stage categorizes every positive question that the selected config answers
NO into one of:

```text
retrieval_miss
fact_guard_conflict
asr_disagreement
nli_or_semantic_support_weak
wrong_event_ranking
consensus_threshold_rejection
```

## Files to add

```text
dev/fn_breakdown.py
```

It depends on the `dev/` files from Stage 1.

## Apply

```bash
cp -r step2_false_negative_breakdown/APPLY_THESE_FILES/* .
```

## Run

Prefer the Stage-1 tuned config if it improved CV/evaluator performance:

```bash
python -m dev.fn_breakdown \
  --cache dev_cache/e1_dev_cache.json.gz \
  --config config/e1_threshold_tuned.json \
  --output-dir dev_results/fn_breakdown
```

If you kept the original config:

```bash
python -m dev.fn_breakdown \
  --cache dev_cache/e1_dev_cache.json.gz \
  --config config/e1_default.json
```

Outputs:

```text
dev_results/fn_breakdown/summary.json
dev_results/fn_breakdown/false_negatives.csv
```

## How to interpret

- **retrieval_miss dominates** -> improve retrieval before touching NLI.
- **nli_or_semantic_support_weak dominates** -> stronger/rephrased reasoner is the next classification target.
- **consensus_threshold_rejection dominates** -> threshold calibration still has room.
- **wrong_event_ranking dominates** -> event/reranking logic is the issue.
- **fact_guard_conflict dominates** -> inspect fact extraction for false contradictions before relaxing the guard globally.
- **asr_disagreement dominates** -> later ambiguity resolver is justified.

### Proceed to Stage 3

Stage 3 is independent of which FN category dominates; it diagnoses the second
major bottleneck, evidence localization.

---

# Stage 3 — evidence proposal oracle-tIoU analysis

## Why

Current tIoU when answered YES is about 0.40 while ASR timestamp ceilings are
about 0.86–0.88. We need to distinguish:

1. **proposal generation is bad** — the correct boundary is never proposed;
2. **proposal ranking is bad** — a good span exists but the scorer chooses the wrong one;
3. **event selection is bad** — good proposals exist under another event.

## Files to add

```text
dev/evidence_oracle.py
tests/test_evidence_oracle.py
```

## Apply

```bash
cp -r step3_evidence_oracle/APPLY_THESE_FILES/* .
```

## Test

```bash
python -m unittest tests/test_evidence_oracle.py
```

## Run

```bash
python -m dev.evidence_oracle \
  --cache dev_cache/e1_dev_cache.json.gz \
  --config config/e1_threshold_tuned.json \
  --output-dir dev_results/evidence_oracle
```

Outputs:

```text
dev_results/evidence_oracle/summary.json
dev_results/evidence_oracle/positive_questions.csv
```

Focus on these values:

```text
mean_selected_tiou_when_answered_yes
mean_selected_event_proposal_oracle_when_answered_yes
mean_gold_event_proposal_oracle
mean_all_proposal_oracle
```

Interpretation:

- selected ~0.40, selected-event oracle ~0.70 -> **ranking is bad**;
- selected-event oracle low but all-proposal oracle high -> **event selection is bad**;
- all-proposal oracle itself low -> **proposal boundaries are bad**.

Stage 4 directly addresses proposal-boundary coverage.

---

# Stage 4 — pause/clause/multi-scale evidence candidates

## Why

E1 v2 generates mostly compact spans. The gold annotations often represent a
whole spoken clause/utterance, not the smallest logically sufficient phrase.
This stage expands the proposal family without changing the ASR or decision
architecture.

New proposal types include:

```text
anchor_cluster
fixed_5 / fixed_8 / fixed_12 / fixed_16 / fixed_24 / fixed_32
pause_0.30 / pause_0.45 / pause_0.60
clause
retrieval_16 / retrieval_32
scan_N       # semantic fallback when no lexical anchor exists
```

Pause proposals use actual ASR word gaps. Clause proposals use punctuation and
basic discourse boundaries. Fixed-width/retrieval proposals ensure the proposal
set contains annotation-like longer alternatives.

## Files to replace

```text
config/e1_default.json
e1/config.py
e1/schemas.py
e1/evidence.py
```

## Files to add

```text
tests/test_e1_evidence_candidates.py
```

Nothing else in the production pipeline needs to change because the existing
second batched NLI pass already consumes `build_evidence_proposals()`.

## Important config handling

If Stage 1 produced useful threshold values, do **not** lose them when replacing
`config/e1_default.json`. Either:

1. copy the tuned threshold values into the new Stage-4 config; or
2. generate the Stage-4 config first and rerun Stage-1 threshold search later.

The safest workflow is to keep named configs:

```text
config/e1_default.json
config/e1_threshold_tuned.json
```

and manually carry the Stage-4 evidence keys into the threshold-tuned file.

## Apply

```bash
cp -r step4_evidence_candidates/APPLY_THESE_FILES/* .
```

## Test

Run all E1 logic tests plus the new proposal test:

```bash
python -m unittest tests/test_e1_logic.py
python -m unittest tests/test_e1_pipeline_mock.py
python -m unittest tests/test_e1_evidence_candidates.py
```

## First endpoint check

Before tuning evidence weights, see whether simply increasing proposal coverage
changes the evaluator:

```bash
E1_CONFIG=config/e1_threshold_tuned.json python api.py
```

```bash
python local_evaluator.py --verbose
```

Do not reject Stage 4 merely because this immediate score is flat. The main
purpose is to increase the **proposal oracle**, after which Stage 5 learns how
to rank the richer proposal family.

## Regenerate the development cache

This is mandatory because the proposal set has changed:

```bash
python -m dev.generate_dev_cache \
  --config config/e1_threshold_tuned.json \
  --device cuda \
  --output dev_cache/e1_dev_cache_step4.json.gz
```

## Re-run the oracle

```bash
python -m dev.evidence_oracle \
  --cache dev_cache/e1_dev_cache_step4.json.gz \
  --config config/e1_threshold_tuned.json \
  --output-dir dev_results/evidence_oracle_step4
```

### Proceed to Stage 5 when

The Stage-4 proposal oracle is materially above the selected tIoU. Ideally the
`mean_all_proposal_oracle` and `mean_gold_event_proposal_oracle` rise versus
Stage 3. That means the richer candidates created real headroom for ranking.

---

# Stage 5 — cross-validated evidence-boundary tuning

## Why

Stage 4 creates many plausible boundaries. Stage 5 tunes the ranking function
against tIoU **by conversation-level cross-validation**, without rerunning any
model.

It tunes:

```text
evidence_min_entailment
evidence_entailment_weight
evidence_fact_weight
evidence_topic_weight
evidence_compactness_weight
evidence_medasr_tie_bonus
```

and proposal-family biases for:

```text
pause
clause
retrieval
anchor
fixed_short
fixed_long
scan
```

Positive compactness weights prefer shorter spans; negative values allow the
search to prefer longer annotation-like spans when that generalizes.

## Files to add

```text
dev/evidence_cv_tune.py
tests/test_evidence_cv_tune.py
```

Stage 4 already made production `e1/evidence.py` and `e1/config.py` understand
`evidence_kind_bias`, so no additional production source replacement is needed.

## Apply

```bash
cp -r step5_evidence_cv_tuning/APPLY_THESE_FILES/* .
```

## Test

```bash
python -m unittest tests/test_evidence_cv_tune.py
```

## Tune

Use the **Stage-4 cache**:

```bash
python -m dev.evidence_cv_tune \
  --cache dev_cache/e1_dev_cache_step4.json.gz \
  --config config/e1_threshold_tuned.json \
  --folds 5 \
  --passes 3 \
  --output-dir dev_results/evidence_cv \
  --write-config config/e1_evidence_tuned.json
```

Outputs:

```text
dev_results/evidence_cv/summary.json
dev_results/evidence_cv/cv_folds.csv
dev_results/evidence_cv/search_history.csv
config/e1_evidence_tuned.json
```

Again, judge generalization by **CV tIoU**, not the full-training recommendation.

## Final endpoint evaluation

```bash
E1_CONFIG=config/e1_evidence_tuned.json python api.py
```

Then:

```bash
python local_evaluator.py --verbose
```

Compare against the 0.538 baseline and every intermediate stage.

---

# Final calibration cycle

Evidence tuning changes the official score associated with a newly recovered
YES, so after Stage 5 it is worth performing **one final threshold search** using
the Stage-4 cache and the evidence-tuned config:

```bash
python dev/threshold_search.py \
  --cache dev_cache/e1_dev_cache_step4.json.gz \
  --config config/e1_evidence_tuned.json \
  --folds 5 \
  --passes 3 \
  --output-dir dev_results/final_threshold_search \
  --write-config config/e1_final_tuned.json
```

This does not rerun any model. `threshold_search.py` dynamically re-ranks the
cached evidence proposals using the evidence weights in the supplied config.

Final local evaluation:

```bash
E1_CONFIG=config/e1_final_tuned.json python api.py
```

```bash
python local_evaluator.py --verbose
```

---

# Suggested experiment log

Keep one row per stage:

| Stage | Positive acc | Hard-neg acc | Off-topic acc | Overall acc | Mean tIoU | tIoU when YES | Score | Worst ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E1 v2 baseline | 0.774 | 0.972 | 1.000 | 0.877 | 0.312 | 0.402 | 0.538 | 11798 |
| Stage 1 thresholds | | | | | | | | |
| Stage 4 proposals | | | | | | | | |
| Stage 5 evidence tune | | | | | | | | |
| Final joint tune | | | | | | | | |

Do not select a configuration only from the 39-conversation full-training
number. Prefer changes that improve the conversation-level CV diagnostics and
then verify that the endpoint behavior is consistent.

# What is and is not changed

These five stages do **not** change:

- MedASR;
- Parakeet v3;
- audio decoding;
- DeBERTa NLI model;
- HTTP/API contract;
- challenge evaluator.

They focus only on the two bottlenecks exposed by the 0.538 run:

1. decision calibration / false-negative attribution;
2. evidence boundary proposal generation and ranking.
