# Medical Appointment ASR benchmark — v3

This revision fixes the three runtime failures found on the A10.

## Fixed

### Parakeet

Previous failure:

```text
Input type (float) and bias type (c10::Half) should be the same
```

The model was FP16 while the processor output remained FP32.

v3 casts floating processor tensors to `model.dtype` while preserving integer
attention masks.

### MedASR

Previous failure:

```text
string indices must be integers, not 'str'
```

The generic Transformers CTC word-timestamp postprocessor assumes a
character-offset tokenizer. MedASR uses a SentencePiece tokenizer, so v3 does
not ask the generic pipeline for word timestamps.

Instead v3:

1. runs `AutoModelForCTC` directly;
2. takes greedy argmax CTC frame IDs;
3. collapses blank/repeated frames;
4. groups SentencePiece subwords into words;
5. converts frame positions into timestamps;
6. uses 20-second chunks with 2-second overlap.

This is consistent with Google's use of greedy decoding for the base MedASR
evaluation while retaining timestamp information from the CTC frames.

### Whisper Large-v3

Previous failure:

```text
libcudnn_ops_infer.so.8: cannot open shared object file
```

That came from `ctranslate2==4.4.0`, which introduced a second cuDNN ABI
requirement.

v3 removes `faster-whisper` and CTranslate2 entirely.

Whisper Large-v3 now runs as:

```text
openai/whisper-large-v3
Transformers
PyTorch 2.5.1
CUDA 12.1
FP16
```

and gets word timestamps through Transformers' Whisper DTW timestamp support.

This gives all three ASR systems one CUDA/PyTorch stack.

## Environment

```text
Python             3.11
PyTorch            2.5.1+cu121
CUDA runtime       12.1
Transformers       5.13.1
huggingface-hub    1.5.0
GPU                NVIDIA A10 24 GB
```

## Update your environment

Remove the old Whisper runtime packages:

```bash
pip uninstall -y faster-whisper ctranslate2
```

Install/refresh the v3 requirements:

```bash
pip install -r requirements_asr_benchmark.txt
```

No PyAV is used. Make sure FFmpeg exists:

```bash
sudo apt update
sudo apt install -y ffmpeg
```

## Hugging Face cache

If you already moved the cache to `/data`, keep:

```bash
export HF_HOME=/data/huggingface
export HF_HUB_CACHE=/data/huggingface/hub
```

## Download the new Whisper weights

v2 used:

```text
Systran/faster-whisper-large-v3
```

v3 uses:

```text
openai/whisper-large-v3
```

Therefore run:

```bash
python prepare_asr_models.py \
  --models parakeet_v3 medasr whisper_large_v3
```

Parakeet and MedASR already in your Hugging Face cache will be reused.

## Clear failed benchmark caches

The previous benchmark wrote failed transcript JSON files with an older cache
schema. v3 uses a new schema, so they will normally be ignored automatically.

For a completely clean smoke test:

```bash
rm -rf benchmark_results/asr
```

## Run unit tests

```bash
python -m unittest tests/test_asr_benchmark.py
```

These tests do not download weights or run ASR inference.

## Two-conversation smoke benchmark

Run each model separately first. This makes debugging much easier.

```bash
python benchmark_asr.py \
  --models parakeet_v3 \
  --device cuda \
  --limit 2 \
  --force
```

Then:

```bash
python benchmark_asr.py \
  --models medasr \
  --device cuda \
  --limit 2 \
  --force
```

Then:

```bash
python benchmark_asr.py \
  --models whisper_large_v3 \
  --device cuda \
  --limit 2 \
  --force
```

Once all three pass:

```bash
python benchmark_asr.py \
  --models parakeet_v3 medasr whisper_large_v3 \
  --device cuda \
  --limit 2 \
  --force
```

Then run the full 39 conversations without `--limit`.

## Outputs

```text
benchmark_results/asr/
├── environment.json
├── report.md
├── model_summary.csv
├── model_summary.json
├── evidence_details_all_models.csv
├── parakeet_v3/
├── medasr/
└── whisper_large_v3/
```

## Note on comparability

The benchmark now compares:

- Parakeet native TDT timing;
- MedASR CTC-frame-derived timing;
- Whisper DTW word timing.

These are different timestamping mechanisms, but all ultimately produce the
same normalized `word/start/end` schema and are evaluated against the same gold
evidence spans.
