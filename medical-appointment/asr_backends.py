"""ASR backends for the Nordic AI Cup Medical Appointment benchmark.

Runtime target:
- Python 3.11
- PyTorch 2.5.1 + CUDA 12.1
- Transformers 5.13.1
- NVIDIA A10 24 GB

All three ASRs use PyTorch/Transformers in this version.  This intentionally
removes CTranslate2/faster-whisper so Whisper shares the same CUDA/cuDNN stack
as Parakeet and MedASR.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import re
from typing import Any, Iterable

SCHEMA_VERSION = 4

MODEL_SPECS = {
    "parakeet_v3": {
        "model_id": "nvidia/parakeet-tdt-0.6b-v3",
        "page": "https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3",
        "runtime": "transformers",
        "timestamp_source": "native TDT token durations grouped into words",
    },
    "medasr": {
        "model_id": "google/medasr",
        "page": "https://huggingface.co/google/medasr",
        "runtime": "transformers-direct-ctc",
        "timestamp_source": "greedy CTC frame spans grouped from SentencePiece tokens",
    },
    "whisper_large_v3": {
        "model_id": "openai/whisper-large-v3",
        "page": "https://huggingface.co/openai/whisper-large-v3",
        "runtime": "transformers-whisper",
        "timestamp_source": "Transformers Whisper DTW word timestamps",
    },
}


@dataclass(frozen=True)
class WordTimestamp:
    text: str
    start: float
    end: float
    confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TranscriptResult:
    text: str
    words: list[WordTimestamp]
    segments: list[dict[str, Any]]
    language: str = "en"
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "words": [w.to_dict() for w in self.words],
            "segments": self.segments,
            "language": self.language,
            "metadata": self.metadata or {},
        }


_WORD_RE = re.compile(
    r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+(?:[/'’\-.][A-Za-zÀ-ÖØ-öø-ÿ0-9]+)*",
    re.UNICODE,
)


def _cuda_index(device: str) -> int:
    if device == "cuda":
        return 0
    if device.startswith("cuda:"):
        return int(device.split(":", 1)[1])
    raise ValueError(f"Not a CUDA device: {device}")


def _move_inputs(inputs: Any, device: Any, dtype: Any | None = None) -> dict[str, Any]:
    """Move BatchFeature/dict tensors without converting integer masks to FP16."""
    moved: dict[str, Any] = {}
    for key, value in dict(inputs).items():
        if not hasattr(value, "to"):
            moved[key] = value
            continue
        if dtype is not None and getattr(value, "is_floating_point", lambda: False)():
            moved[key] = value.to(device=device, dtype=dtype)
        else:
            moved[key] = value.to(device=device)
    return moved


def _canonical_chars(text: str) -> str:
    return "".join(ch.lower() for ch in text if ch.isalnum())


def parakeet_tokens_to_words(
    transcript: str,
    token_timestamps: Iterable[dict[str, Any]],
) -> list[WordTimestamp]:
    token_ranges: list[tuple[int, int, float, float]] = []
    char_stream_parts: list[str] = []
    cursor = 0

    for item in token_timestamps:
        token = str(item.get("token", ""))
        norm = _canonical_chars(token)
        if not norm:
            continue

        start = float(item["start"])
        end = float(item["end"])
        if not (math.isfinite(start) and math.isfinite(end)) or end < start:
            continue

        left = cursor
        cursor += len(norm)
        token_ranges.append((left, cursor, start, end))
        char_stream_parts.append(norm)

    char_stream = "".join(char_stream_parts)
    if not char_stream:
        return []

    words: list[WordTimestamp] = []
    search_from = 0

    for match in _WORD_RE.finditer(transcript):
        raw_word = match.group(0)
        norm_word = _canonical_chars(raw_word)
        if not norm_word:
            continue

        pos = char_stream.find(norm_word, search_from)
        if pos < 0:
            pos = char_stream.find(norm_word, max(0, search_from - 6))
        if pos < 0:
            continue

        right = pos + len(norm_word)
        overlapping = [r for r in token_ranges if r[1] > pos and r[0] < right]

        if overlapping:
            words.append(
                WordTimestamp(
                    text=raw_word,
                    start=min(r[2] for r in overlapping),
                    end=max(r[3] for r in overlapping),
                )
            )
        search_from = right

    return words


def ctc_collapse_runs(
    frame_ids: list[int],
    blank_id: int,
    special_ids: set[int] | None = None,
) -> list[tuple[int, int, int]]:
    """Collapse CTC frame IDs into token runs.

    Returns (token_id, start_frame, end_frame_exclusive). Repeated IDs separated
    by a blank correctly remain separate token emissions.
    """
    special_ids = special_ids or set()
    runs: list[tuple[int, int, int]] = []
    if not frame_ids:
        return runs

    run_id = frame_ids[0]
    run_start = 0

    def maybe_add(token_id: int, start: int, end: int) -> None:
        if token_id == blank_id or token_id in special_ids:
            return
        runs.append((token_id, start, end))

    for i in range(1, len(frame_ids) + 1):
        if i == len(frame_ids) or frame_ids[i] != run_id:
            maybe_add(run_id, run_start, i)
            if i < len(frame_ids):
                run_id = frame_ids[i]
                run_start = i

    return runs


def sentencepiece_run_groups(
    token_strings: list[str],
) -> list[tuple[int, int]]:
    """Group SentencePiece token strings into lexical-word token ranges.

    Returns ranges into token_strings: (start_index, end_index_exclusive).
    """
    if not token_strings:
        return []

    groups: list[tuple[int, int]] = []
    start = 0

    for i, token in enumerate(token_strings):
        if i > 0 and token.startswith("▁"):
            groups.append((start, i))
            start = i

    groups.append((start, len(token_strings)))
    return groups


def _words_to_sentence_segments(words: list[WordTimestamp]) -> list[dict[str, Any]]:
    if not words:
        return []

    segments: list[dict[str, Any]] = []
    bucket: list[WordTimestamp] = []

    for word in words:
        bucket.append(word)
        if word.text.endswith((".", "?", "!")) or len(bucket) >= 32:
            segments.append(_segment_from_words(bucket))
            bucket = []

    if bucket:
        segments.append(_segment_from_words(bucket))

    return segments


def _segment_from_words(words: list[WordTimestamp]) -> dict[str, Any]:
    return {
        "start": words[0].start,
        "end": words[-1].end,
        "text": " ".join(w.text for w in words),
    }


class ParakeetV3Backend:
    name = "parakeet_v3"
    model_id = MODEL_SPECS[name]["model_id"]

    def __init__(self, device: str = "cuda") -> None:
        import torch
        from transformers import AutoModelForTDT, AutoProcessor

        self.device = device
        self.processor = AutoProcessor.from_pretrained(self.model_id)

        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.model = AutoModelForTDT.from_pretrained(
            self.model_id,
            dtype=dtype,
        )
        self.model.to(device)
        self.model.eval()

    def transcribe(self, audio: Any, sample_rate: int = 16000) -> TranscriptResult:
        import torch

        inputs = self.processor(
            [audio],
            sampling_rate=sample_rate,
            return_tensors="pt",
        )
        inputs = _move_inputs(
            inputs,
            self.model.device,
            dtype=self.model.dtype,
        )

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                return_dict_in_generate=True,
            )

        decoded_text, decoded_timestamps = self.processor.decode(
            output.sequences,
            durations=output.durations,
            skip_special_tokens=True,
        )

        text = (
            decoded_text[0]
            if isinstance(decoded_text, (list, tuple))
            else str(decoded_text)
        )

        token_stamps = decoded_timestamps
        if (
            isinstance(decoded_timestamps, (list, tuple))
            and decoded_timestamps
            and isinstance(decoded_timestamps[0], list)
        ):
            token_stamps = decoded_timestamps[0]

        words = parakeet_tokens_to_words(
            text,
            token_stamps or [],
        )

        if not words:
            raise RuntimeError(
                "Parakeet produced no usable word timestamps."
            )

        return TranscriptResult(
            text=text,
            words=words,
            segments=_words_to_sentence_segments(words),
            metadata={
                "model_id": self.model_id,
                "dtype": str(self.model.dtype),
                "runtime": MODEL_SPECS[self.name]["runtime"],
                "timestamp_source": MODEL_SPECS[self.name]["timestamp_source"],
            },
        )


class MedASRBackend:
    """MedASR with direct greedy CTC decoding and frame-derived timestamps.

    The generic Transformers ASR pipeline currently fails on MedASR word
    timestamps because MedASR uses a SentencePiece tokenizer rather than the
    character tokenizer assumed by the generic CTC timestamp postprocessor.
    """

    name = "medasr"
    model_id = MODEL_SPECS[name]["model_id"]

    def __init__(
        self,
        device: str = "cuda",
        chunk_length_s: float = 20.0,
        stride_length_s: float = 2.0,
    ) -> None:
        import torch
        from transformers import AutoModelForCTC, AutoProcessor

        self.device = device
        self.chunk_length_s = float(chunk_length_s)
        self.stride_length_s = float(stride_length_s)

        if self.chunk_length_s <= 0:
            raise ValueError("chunk_length_s must be positive")
        if not 0 <= self.stride_length_s < self.chunk_length_s:
            raise ValueError("stride_length_s must be >=0 and < chunk_length_s")

        try:
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.model = AutoModelForCTC.from_pretrained(
                self.model_id,
                dtype=torch.float32,
            )
        except Exception as exc:
            msg = str(exc).lower()
            if any(
                key in msg
                for key in (
                    "gated",
                    "401",
                    "403",
                    "access",
                    "authorized",
                    "authorization",
                )
            ):
                raise RuntimeError(
                    "MedASR is gated on Hugging Face. Accept the model terms at "
                    f"{MODEL_SPECS[self.name]['page']}, then run `hf auth login` "
                    "or set HF_TOKEN and retry."
                ) from exc
            raise

        self.model.to(device)
        self.model.eval()

        tokenizer = self.processor.tokenizer
        blank_id = getattr(self.model.config, "pad_token_id", None)
        if blank_id is None:
            blank_id = getattr(tokenizer, "pad_token_id", None)
        if blank_id is None:
            raise RuntimeError("Could not determine MedASR CTC blank token ID.")

        self.blank_id = int(blank_id)
        self.special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
        self.special_ids.discard(self.blank_id)

    def _transcribe_chunk(
        self,
        chunk: Any,
        sample_rate: int,
        global_start_s: float,
    ) -> list[WordTimestamp]:
        import torch

        inputs = self.processor(
            chunk,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs = _move_inputs(
            inputs,
            self.model.device,
            dtype=self.model.dtype,
        )

        with torch.inference_mode():
            outputs = self.model(**inputs)

        logits = outputs.logits[0]
        frame_ids = logits.argmax(dim=-1).detach().cpu().tolist()
        num_frames = len(frame_ids)

        if num_frames == 0:
            return []

        chunk_duration_s = float(len(chunk)) / float(sample_rate)
        seconds_per_frame = chunk_duration_s / float(num_frames)

        runs = ctc_collapse_runs(
            frame_ids,
            blank_id=self.blank_id,
            special_ids=self.special_ids,
        )
        if not runs:
            return []

        tokenizer = self.processor.tokenizer
        token_ids = [run[0] for run in runs]
        token_strings = tokenizer.convert_ids_to_tokens(token_ids)

        groups = sentencepiece_run_groups(token_strings)
        words: list[WordTimestamp] = []

        for left, right in groups:
            group_ids = token_ids[left:right]
            group_runs = runs[left:right]

            text = tokenizer.decode(
                group_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            ).strip()

            if not text:
                raw = "".join(token_strings[left:right])
                text = raw.replace("▁", " ").strip()

            if not text:
                continue

            start_frame = group_runs[0][1]
            end_frame = group_runs[-1][2]

            words.append(
                WordTimestamp(
                    text=text,
                    start=global_start_s + start_frame * seconds_per_frame,
                    end=global_start_s + end_frame * seconds_per_frame,
                )
            )

        return words

    def transcribe(self, audio: Any, sample_rate: int = 16000) -> TranscriptResult:
        total_samples = len(audio)
        if total_samples == 0:
            raise RuntimeError("Empty audio.")

        chunk_samples = max(1, int(round(self.chunk_length_s * sample_rate)))
        overlap_samples = max(0, int(round(self.stride_length_s * sample_rate)))
        step_samples = chunk_samples - overlap_samples

        all_words: list[WordTimestamp] = []
        chunk_index = 0
        start_sample = 0

        while start_sample < total_samples:
            end_sample = min(total_samples, start_sample + chunk_samples)
            chunk = audio[start_sample:end_sample]

            global_start_s = start_sample / float(sample_rate)
            global_end_s = end_sample / float(sample_rate)
            is_first = chunk_index == 0
            is_last = end_sample >= total_samples

            words = self._transcribe_chunk(
                chunk,
                sample_rate,
                global_start_s,
            )

            # Keep the center of overlapping regions from only one chunk.
            # For a 2 s overlap, each neighboring chunk owns 1 s.
            half_overlap_s = self.stride_length_s / 2.0
            trusted_start = (
                global_start_s
                if is_first
                else global_start_s + half_overlap_s
            )
            trusted_end = (
                global_end_s
                if is_last
                else global_end_s - half_overlap_s
            )

            for word in words:
                midpoint = 0.5 * (word.start + word.end)
                if trusted_start <= midpoint <= trusted_end:
                    all_words.append(word)

            if is_last:
                break

            start_sample += step_samples
            chunk_index += 1

        all_words.sort(key=lambda w: (w.start, w.end))

        if not all_words:
            raise RuntimeError("MedASR produced no timestamped words.")

        text = " ".join(word.text for word in all_words)

        return TranscriptResult(
            text=text,
            words=all_words,
            segments=_words_to_sentence_segments(all_words),
            metadata={
                "model_id": self.model_id,
                "dtype": str(self.model.dtype),
                "chunk_length_s": self.chunk_length_s,
                "stride_length_s": self.stride_length_s,
                "runtime": MODEL_SPECS[self.name]["runtime"],
                "timestamp_source": MODEL_SPECS[self.name]["timestamp_source"],
            },
        )


class WhisperLargeV3Backend:
    """Whisper Large-v3 via native Transformers/PyTorch.

    This intentionally avoids faster-whisper/CTranslate2 so there is no second
    cuDNN ABI requirement beside the PyTorch 2.5.1 CUDA stack.
    """

    name = "whisper_large_v3"
    model_id = MODEL_SPECS[name]["model_id"]

    def __init__(
        self,
        device: str = "cuda",
        beam_size: int = 5,
    ) -> None:
        import torch
        from transformers import (
            AutoModelForSpeechSeq2Seq,
            AutoProcessor,
            pipeline,
        )

        self.device = device
        self.beam_size = int(beam_size)
        self.dtype = torch.float16 if device.startswith("cuda") else torch.float32

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id,
            dtype=self.dtype,
        )
        self.model.to(device)
        self.model.eval()

        pipeline_device = _cuda_index(device) if device.startswith("cuda") else -1

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            device=pipeline_device,
            dtype=self.dtype,
        )

    def transcribe(self, audio: Any, sample_rate: int = 16000) -> TranscriptResult:
        result = self.pipe(
            {"array": audio, "sampling_rate": sample_rate},
            return_timestamps="word",
            batch_size=1,
            generate_kwargs={
                "language": "english",
                "task": "transcribe",
                "num_beams": self.beam_size,
            },
        )

        chunks = result.get("chunks") or []
        words: list[WordTimestamp] = []

        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue

            stamp = chunk.get("timestamp")
            if not stamp or stamp[0] is None or stamp[1] is None:
                continue

            text = str(chunk.get("text", "")).strip()
            if not text:
                continue

            words.append(
                WordTimestamp(
                    text=text,
                    start=float(stamp[0]),
                    end=float(stamp[1]),
                )
            )

        if not words:
            raise RuntimeError("Whisper returned no word timestamps.")

        return TranscriptResult(
            text=str(result.get("text", "")).strip(),
            words=words,
            segments=_words_to_sentence_segments(words),
            metadata={
                "model_id": self.model_id,
                "dtype": str(self.model.dtype),
                "beam_size": self.beam_size,
                "runtime": MODEL_SPECS[self.name]["runtime"],
                "timestamp_source": MODEL_SPECS[self.name]["timestamp_source"],
            },
        )


def create_backend(name: str, device: str = "cuda", **kwargs: Any) -> Any:
    if name == "parakeet_v3":
        return ParakeetV3Backend(device=device)

    if name == "medasr":
        return MedASRBackend(
            device=device,
            chunk_length_s=kwargs.get("medasr_chunk_length_s", 20.0),
            stride_length_s=kwargs.get("medasr_stride_length_s", 2.0),
        )

    if name == "whisper_large_v3":
        return WhisperLargeV3Backend(
            device=device,
            beam_size=kwargs.get("whisper_beam_size", 5),
        )

    raise ValueError(
        f"Unknown ASR backend: {name}. Choose from {sorted(MODEL_SPECS)}"
    )
