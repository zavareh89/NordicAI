# Medical Appointment — Dual-ASR E1

This package implements the first end-to-end solution after ASR selection:

```text
audio
  ├── MedASR ──────────────┐
  └── Parakeet v3 ─────────┤
                           ▼
                independent transcripts
                           │
                normalization + facts
                           │
              retrieval from each ASR
                           │
                 temporal alignment
                           │
              batched 3-way local NLI
                           │
          deterministic exact-fact guard
                           │
                dual-ASR consensus
                           │
              word-index evidence trim
                           │
                   YES/NO + span
```

**The two ASR transcripts are never merged.** Their disagreement is retained as information.

## Models used by E1

- `google/medasr`
- `nvidia/parakeet-tdt-0.6b-v3`
- `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`

Whisper is not used by E1. The old Whisper benchmark remains in the package only so previous ASR experiments are reproducible.

## Environment

Keep the environment already used for the successful ASR benchmark:

```text
Python        3.11
PyTorch       2.5.1+cu121
CUDA          12.1
Transformers  5.13.1
GPU           NVIDIA A10 24 GB
```

Install system FFmpeg if needed:

```bash
sudo apt update
sudo apt install -y ffmpeg
```

Install PyTorch first:

```bash
pip install \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

Then:

```bash
pip install -r requirements_e1.txt
```

If your Hugging Face cache is on `/data`, keep:

```bash
export HF_HOME=/data/huggingface
export HF_HUB_CACHE=/data/huggingface/hub
```

## Download/check E1 weights

```bash
python prepare_e1_models.py
```

MedASR is gated, so its terms must already be accepted and Hugging Face authentication must be active.

## Integration with the official challenge repo

Copy this package into `medical-appointment/` and replace only the official baseline `example.py` with the provided `example.py`.

Do **not** change:

- `api.py`
- `dtos.py`
- `utils.py`
- `local_evaluator.py`

The production server eagerly loads E1 by default so the endpoint is not declared ready before weights are resident.

Optional environment variables:

```bash
export E1_DEVICE=cuda
export E1_CONFIG=config/e1_default.json
```

`E1_LAZY_LOAD=1` exists only for lightweight imports/tests. Do not use it for the competition server unless you intentionally want the first request to pay model-loading latency.

## Run tests on your server

The package includes two new test files. They use synthetic transcripts and mock ASR/NLI objects, so they do not require model downloads or inference:

```bash
python -m unittest tests/test_e1_logic.py
python -m unittest tests/test_e1_pipeline_mock.py
```

The previous ASR tests are also retained:

```bash
python -m unittest tests/test_asr_benchmark.py
```

## Run the endpoint

Use the unchanged official API:

```bash
python api.py
```

Then in another shell:

```bash
python local_evaluator.py --verbose
```

This is the main E1 experiment. It gives the official local accuracy/tIoU/score decomposition.

## Optional trace tool

To see retrieval candidates and E1 decisions for one conversation:

```bash
python e1_trace.py path/to/conversation.mp3 questions.json
```

where `questions.json` is a JSON list of question strings.

## Architecture details

### 1. Dual ASR

Each request is decoded only once to mono 16 kHz float32. MedASR and Parakeet each transcribe the complete conversation once. All ten questions share the resulting transcripts.

If one ASR crashes, E1 continues with the other. If both fail, `example.py` protects the HTTP contract with a catastrophic fallback rather than losing all ten questions through an exception.

### 2. Normalization

Raw ASR words are preserved. A separate normalized representation converts common forms such as:

```text
five hundred milligrams  -> 500 mg
0.5 grams                 -> 500 mg (fact representation)
two weeks                 -> 14 days (fact representation)
135 over 88               -> blood-pressure fact (135, 88)
twice daily               -> 2/day
```

Raw text is always available for NLI and diagnostics.

### 3. Deterministic facts

E1 extracts strict facts including:

- mass dose
- unit dose
- volume
- duration
- frequency
- blood pressure
- temperature
- common numeric laboratory units
- percentages
- left/right/bilateral side

A candidate is only called a deterministic contradiction when the question and candidate contain a comparable strict fact type and the candidate has no matching value for the question-side value.

This is deliberately cautious. It prevents a passage containing both `500 mg` and `250 mg` from being rejected merely because one of its values differs.

### 4. Hybrid retrieval

Retrieval runs independently on each transcript. Windows default to 28 words with stride 8.

The score combines:

- BM25 using the full normalized question;
- BM25 with numbers/units removed so a hard-negative value still retrieves the underlying event;
- exact topic-token overlap;
- fuzzy token overlap for ASR spelling errors;
- strict fact-type overlap;
- exact fact match as a bonus, never a requirement.

Top 4 windows from each ASR are retained.

### 5. Temporal alignment

MedASR and Parakeet candidates are considered the same event when either:

```text
temporal IoU >= 0.15
```

or their centers differ by at most:

```text
2.5 seconds
```

Consensus uses this alignment when deciding whether two YES votes support the same real utterance.

### 6. Batched NLI

All candidate/question pairs for all ten questions are sent through the NLI model in batches rather than invoking it separately per question.

For every candidate it produces:

```text
ENTAILMENT
NEUTRAL
CONTRADICTION
```

The hypothesis template is configurable and defaults to:

```text
The correct answer is yes to this question: <question>
```

### 7. Fact guard

NLI cannot override a strict exact-value contradiction.

Example:

```text
question:   Was the dose 200 mg?
candidate:  The dose was 100 mg.
```

Even if NLI assigns high entailment because the sentences are semantically similar, the deterministic guard forces a strong contradiction score.

Exact fact agreement gives only a modest entailment bonus; it does not by itself produce a YES.

### 8. Consensus

Each ASR gets its own best YES and NO evidence vote.

Main paths:

- YES + YES on aligned evidence -> YES
- NO + NO -> NO
- strong YES + neutral other ASR -> YES if no deterministic contradiction exists
- YES + NO -> normally NO, except when the YES transcript exactly matches the questioned strict fact and the NO transcript is contradictory specifically because the ASRs disagree on that fact
- neutral/uncertain -> NO

This is E1's deterministic replacement for the future LLM ambiguity resolver.

### 9. Evidence localization

The NLI model never creates timestamps.

After a YES decision, E1 chooses a winning ASR candidate and anchors evidence using:

- matching topic words;
- fuzzy topic matches;
- exact strict-fact n-grams.

It then returns a tight word-index span, adds at most two words of context on each side, caps the evidence at 18 words, and converts the selected immutable word IDs to timestamps.

MedASR is preferred for timing when its support score is close to Parakeet because it had the higher timestamp ceiling in the ASR benchmark. Stronger Parakeet evidence still wins when the score difference is meaningful.

## Configuration

Everything intended for tuning is in:

```text
config/e1_default.json
```

Do not hard-code HPO changes in the Python modules. Important future tuning dimensions include:

- retrieval window/stride/top-k;
- temporal pairing thresholds;
- NLI thresholds;
- single-source YES threshold;
- fact-match bonus;
- MedASR timestamp preference tolerance;
- evidence context words/max words/padding.

## Runtime design

The two ASRs and NLI model are loaded once and kept resident. ASR runs once per conversation, not once per question. NLI is batched across all candidate windows from all ten questions.

There is no LLM in E1 and no cloud API call.
