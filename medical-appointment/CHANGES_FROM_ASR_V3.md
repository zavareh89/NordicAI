# Changes from the previous ASR v3 package

## Existing files retained unchanged

These are kept so the ASR benchmark remains reproducible:

- `asr_backends.py`
- `benchmark_asr.py`
- `prepare_asr_models.py`
- `requirements_asr_benchmark.txt`
- `ASR_BENCHMARK.md`
- `tests/test_asr_benchmark.py`

## New files for Dual-ASR E1

- `example.py` — drop-in replacement for the official baseline; keeps `api.py`, `dtos.py`, and `utils.py` unchanged.
- `prepare_e1_models.py` — downloads MedASR, Parakeet v3, and the NLI model.
- `requirements_e1.txt` — E1 dependencies for the Torch 2.5.1 + cu121 environment.
- `config/e1_default.json` — every retrieval/NLI/consensus/evidence threshold in one tunable file.
- `e1/audio.py` — FFmpeg byte/path decoding to mono 16 kHz float32.
- `e1/config.py` — typed E1 configuration.
- `e1/schemas.py` — normalized transcript, candidate, fact, NLI, decision, evidence schemas.
- `e1/textnorm.py` — number/unit normalization and topic-token extraction.
- `e1/facts.py` — deterministic dose/duration/frequency/BP/value/side extraction and contradiction guard.
- `e1/retrieval.py` — independent hybrid retrieval from MedASR and Parakeet.
- `e1/pairing.py` — temporal pairing/alignment of candidates across ASRs.
- `e1/nli.py` — batched local DeBERTa 3-way NLI.
- `e1/consensus.py` — source voting, exact-fact guard, dual-ASR consensus and disagreement handling.
- `e1/evidence.py` — deterministic word-index evidence tightening and timestamp conversion.
- `e1/pipeline.py` — full dual-ASR E1 orchestration; ASR once per request, all ten questions batched through NLI.
- `e1_trace.py` — optional per-question diagnostic tool.
- `tests/test_e1_logic.py` — pure logic tests.
- `tests/test_e1_pipeline_mock.py` — end-to-end mock pipeline tests without model downloads.
- `E1_README.md` — setup, architecture, run and evaluation instructions.

No changes are required to the official `api.py`, `dtos.py`, `utils.py`, or `local_evaluator.py`.
