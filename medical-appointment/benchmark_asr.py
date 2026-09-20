#!/usr/bin/env python3
"""ASR-only benchmark for Nordic AI Cup Medical Appointment."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import threading
import time
from typing import Any, Iterable

from asr_backends import MODEL_SPECS, SCHEMA_VERSION, create_backend

DEFAULT_MODELS = ["parakeet_v3", "medasr", "whisper_large_v3"]

NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
    "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
    "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    "hundred", "thousand", "million", "first", "second", "third",
    "once", "twice",
}

STOPWORDS = {
    "a", "about", "after", "again", "all", "also", "am", "an", "and",
    "any", "are", "as", "at", "be", "been", "before", "being", "both",
    "but", "by", "can", "did", "do", "does", "for", "from", "had", "has",
    "have", "he", "her", "here", "him", "his", "how", "i", "in", "into",
    "is", "it", "its", "me", "more", "no", "not", "of", "on", "or", "our",
    "out", "patient", "right", "she", "should", "so", "some", "than",
    "that", "the", "their", "them", "then", "there", "they", "this", "to",
    "was", "we", "were", "what", "when", "where", "which", "who", "will",
    "with", "would", "yes", "you", "your",
}

TERM_RE = re.compile(
    r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+(?:[/.'’-][A-Za-zÀ-ÖØ-öø-ÿ0-9]+)*"
)


def environment_snapshot() -> dict[str, Any]:
    snap: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "ffmpeg": shutil.which("ffmpeg"),
    }

    try:
        import torch
        snap["torch"] = torch.__version__
        snap["torch_cuda"] = torch.version.cuda
        snap["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            snap["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        snap["torch_error"] = repr(exc)

    for key, module_name in [
        ("transformers", "transformers"),
        ("huggingface_hub", "huggingface_hub"),
        ("numpy", "numpy"),
    ]:
        try:
            module = __import__(module_name)
            snap[key] = getattr(module, "__version__", "unknown")
        except Exception:
            pass

    return snap


def preflight(device: str) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg is not installed. Run: "
            "sudo apt update && sudo apt install -y ffmpeg"
        )

    if device.startswith("cuda"):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but torch.cuda.is_available() is False."
            )


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return xs[lo]
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def temporal_iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    overlap = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return overlap / union if union > 0 else 0.0


def cache_signature(model: str, options: dict[str, Any]) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "model": model,
        "model_id": MODEL_SPECS[model]["model_id"],
        "options": options,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def decode_audio_16k(path: Path) -> tuple[Any, int, float]:
    import numpy as np

    cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(path),
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "-ac", "1",
        "-ar", "16000",
        "pipe:1",
    ]

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed for {path}: {stderr}")

    audio = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    if audio.size == 0:
        raise RuntimeError(f"Decoded zero audio samples: {path}")

    sample_rate = 16000
    return audio, sample_rate, float(audio.size) / sample_rate


class GpuMemoryMonitor:
    def __init__(self, device: str, interval_s: float = 0.02) -> None:
        self.device = device
        self.interval_s = interval_s
        self.peak_mb: float | None = None
        self.baseline_mb: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._reader = None

    def _build_reader(self):
        if not self.device.startswith("cuda"):
            return None

        try:
            import pynvml
            pynvml.nvmlInit()
            index = int(self.device.split(":", 1)[1]) if ":" in self.device else 0
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            pid = os.getpid()

            fns = [
                getattr(pynvml, "nvmlDeviceGetComputeRunningProcesses_v3", None),
                getattr(pynvml, "nvmlDeviceGetComputeRunningProcesses_v2", None),
                getattr(pynvml, "nvmlDeviceGetComputeRunningProcesses", None),
            ]
            fns = [fn for fn in fns if fn is not None]

            def read_mb() -> float:
                for fn in fns:
                    try:
                        values = [
                            float(p.usedGpuMemory) / (1024.0 * 1024.0)
                            for p in fn(handle)
                            if p.pid == pid
                            and getattr(p, "usedGpuMemory", None) not in (None, 0)
                        ]
                        if values:
                            return sum(values)
                    except Exception:
                        continue
                return float(pynvml.nvmlDeviceGetMemoryInfo(handle).used) / (1024.0 * 1024.0)

            return read_mb
        except Exception:
            return None

    def __enter__(self):
        self._reader = self._build_reader()
        if self._reader is None:
            return self

        self.baseline_mb = self._reader()
        self.peak_mb = self.baseline_mb

        def poll() -> None:
            while not self._stop.wait(self.interval_s):
                try:
                    self.peak_mb = max(self.peak_mb or 0.0, self._reader())
                except Exception:
                    pass

        self._thread = threading.Thread(target=poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._reader is not None:
            try:
                self.peak_mb = max(self.peak_mb or 0.0, self._reader())
            except Exception:
                pass

        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    @property
    def incremental_peak_mb(self) -> float | None:
        if self.peak_mb is None or self.baseline_mb is None:
            return None
        return max(0.0, self.peak_mb - self.baseline_mb)


def normalize_terms(text: str, drop_stopwords: bool = True) -> list[str]:
    terms = [m.group(0).lower().replace("’", "'") for m in TERM_RE.finditer(text)]
    return [t for t in terms if t not in STOPWORDS] if drop_stopwords else terms


def numeric_terms(text: str) -> set[str]:
    return {
        token
        for token in normalize_terms(text, drop_stopwords=False)
        if any(ch.isdigit() for ch in token) or token in NUMBER_WORDS
    }


def read_questions(csv_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            item: dict[str, Any] = dict(row)
            item["label"] = int(row["label"])
            item["evidence_start"] = (
                float(row["evidence_start"]) if row["evidence_start"] else None
            )
            item["evidence_end"] = (
                float(row["evidence_end"]) if row["evidence_end"] else None
            )
            rows.append(item)

    return rows


def sample_sort_key(transcript_id: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", transcript_id)
    return (int(match.group(1)) if match else 10**9, transcript_id)


def discover_audio(data_dir: Path, rows: list[dict[str, Any]]) -> list[tuple[str, Path]]:
    ids = sorted({r["transcript_id"] for r in rows}, key=sample_sort_key)
    items: list[tuple[str, Path]] = []

    for transcript_id in ids:
        path = data_dir / "audio" / f"conversation_{transcript_id}.mp3"
        if not path.exists():
            raise FileNotFoundError(f"Missing audio for {transcript_id}: {path}")
        items.append((transcript_id, path))

    return items


def validate_transcript_payload(payload: dict[str, Any]) -> None:
    words = payload.get("transcript", {}).get("words")

    if not isinstance(words, list):
        raise ValueError("Transcript payload is missing a words list")

    previous_start = -1.0

    for i, word in enumerate(words):
        start = float(word["start"])
        end = float(word["end"])

        if not (math.isfinite(start) and math.isfinite(end)) or end < start:
            raise ValueError(f"Invalid timestamp at word {i}: {start}-{end}")

        if start + 0.5 < previous_start:
            raise ValueError(f"Word timestamps are badly out of order at index {i}")

        previous_start = start


def transcript_cache_path(output_dir: Path, model: str, transcript_id: str) -> Path:
    return output_dir / model / "transcripts" / f"{transcript_id}.json"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_windows(words: list[dict[str, Any]], size: int, stride: int) -> list[dict[str, Any]]:
    if not words:
        return []

    starts = list(range(0, len(words), stride))
    if starts and starts[-1] + size < len(words):
        starts.append(max(0, len(words) - size))

    windows: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()

    for left in starts:
        right = min(len(words), left + size)
        if right <= left:
            continue

        key = (left, right)
        if key in seen:
            continue
        seen.add(key)

        chunk = words[left:right]
        windows.append(
            {
                "left": left,
                "right": right,
                "start": float(chunk[0]["start"]),
                "end": float(chunk[-1]["end"]),
                "text": " ".join(str(w["text"]) for w in chunk),
            }
        )

        if right == len(words):
            break

    return windows


def bm25_rank(question: str, windows: list[dict[str, Any]]) -> list[tuple[int, float]]:
    if not windows:
        return []

    query = normalize_terms(question)
    docs = [normalize_terms(w["text"]) for w in windows]
    n_docs = len(docs)
    avgdl = sum(len(doc) for doc in docs) / max(1, n_docs)

    dfs: dict[str, int] = {}
    for doc in docs:
        for term in set(doc):
            dfs[term] = dfs.get(term, 0) + 1

    query_numeric = numeric_terms(question)
    scores: list[tuple[int, float]] = []
    k1, b = 1.5, 0.75

    for i, doc in enumerate(docs):
        tf: dict[str, int] = {}
        for term in doc:
            tf[term] = tf.get(term, 0) + 1

        dl = len(doc)
        score = 0.0

        for term in query:
            if term not in tf:
                continue

            df = dfs.get(term, 0)
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            freq = tf[term]
            denom = freq + k1 * (1.0 - b + b * dl / max(avgdl, 1e-9))
            score += idf * (freq * (k1 + 1.0)) / denom

        score += 1.5 * len(query_numeric & numeric_terms(windows[i]["text"]))
        scores.append((i, score))

    return sorted(scores, key=lambda x: (-x[1], x[0]))


def evidence_detail(
    model: str,
    row: dict[str, Any],
    transcript: dict[str, Any],
    window_words: int,
    window_stride: int,
    context_s: float,
) -> dict[str, Any]:
    gold_start = float(row["evidence_start"])
    gold_end = float(row["evidence_end"])
    words = transcript["transcript"]["words"]

    overlap_words = [
        w
        for w in words
        if min(float(w["end"]), gold_end) > max(float(w["start"]), gold_start)
    ]

    context_words = [
        w
        for w in words
        if min(float(w["end"]), gold_end + context_s)
        > max(float(w["start"]), gold_start - context_s)
    ]

    asr_evidence_text = " ".join(str(w["text"]) for w in overlap_words)
    context_text = " ".join(str(w["text"]) for w in context_words)

    if overlap_words:
        envelope_start = float(overlap_words[0]["start"])
        envelope_end = float(overlap_words[-1]["end"])
        timestamp_ceiling = temporal_iou(
            envelope_start,
            envelope_end,
            gold_start,
            gold_end,
        )
    else:
        envelope_start = None
        envelope_end = None
        timestamp_ceiling = 0.0

    query_terms = set(normalize_terms(row["question"]))
    evidence_terms = set(normalize_terms(asr_evidence_text, drop_stopwords=False))
    content_recall = (
        len(query_terms & evidence_terms) / len(query_terms) if query_terms else None
    )

    q_num = numeric_terms(row["question"])
    e_num = numeric_terms(asr_evidence_text)
    numeric_recall = len(q_num & e_num) / len(q_num) if q_num else None

    windows = build_windows(words, window_words, window_stride)
    ranking = bm25_rank(row["question"], windows)

    gold_ranks: list[int] = []
    best_gold_tiou = 0.0

    for rank, (idx, _) in enumerate(ranking, start=1):
        window = windows[idx]
        overlap = temporal_iou(
            float(window["start"]),
            float(window["end"]),
            gold_start,
            gold_end,
        )
        if overlap > 0:
            gold_ranks.append(rank)
            best_gold_tiou = max(best_gold_tiou, overlap)

    retrieval_rank = min(gold_ranks) if gold_ranks else None

    return {
        "model": model,
        "question_id": row["question_id"],
        "transcript_id": row["transcript_id"],
        "question": row["question"],
        "gold_start": gold_start,
        "gold_end": gold_end,
        "gold_duration_s": gold_end - gold_start,
        "asr_word_count_in_gold": len(overlap_words),
        "asr_envelope_start": envelope_start,
        "asr_envelope_end": envelope_end,
        "timestamp_ceiling_tiou": timestamp_ceiling,
        "content_token_recall": content_recall,
        "numeric_token_count": len(q_num),
        "numeric_token_recall": numeric_recall,
        "retrieval_rank": retrieval_rank,
        "retrieval_hit_at_1": int(retrieval_rank is not None and retrieval_rank <= 1),
        "retrieval_hit_at_3": int(retrieval_rank is not None and retrieval_rank <= 3),
        "retrieval_hit_at_5": int(retrieval_rank is not None and retrieval_rank <= 5),
        "best_gold_window_tiou": best_gold_tiou,
        "asr_evidence_text": asr_evidence_text,
        "context_text": context_text,
    }


def mean_non_null(values: Iterable[float | None]) -> float | None:
    values = [float(v) for v in values if v is not None]
    return sum(values) / len(values) if values else None


def summarize_model(
    model: str,
    transcript_payloads: list[dict[str, Any]],
    details: list[dict[str, Any]],
    load_info: dict[str, Any] | None,
) -> dict[str, Any]:
    ok = [p for p in transcript_payloads if not p.get("error")]
    failed = [p for p in transcript_payloads if p.get("error")]

    inf = [float(p["timing"]["inference_s"]) for p in ok]
    e2e = [float(p["timing"]["decode_plus_inference_s"]) for p in ok]
    rtf = [float(p["timing"]["rtf"]) for p in ok]
    peaks = [
        float(p["memory"]["peak_process_vram_mb"])
        for p in ok
        if p.get("memory", {}).get("peak_process_vram_mb") is not None
    ]

    numeric_details = [d for d in details if d["numeric_token_recall"] is not None]

    return {
        "model": model,
        "model_id": MODEL_SPECS[model]["model_id"],
        "conversations_ok": len(ok),
        "conversations_failed": len(failed),
        "load_time_s": (load_info or {}).get("load_time_s"),
        "load_peak_vram_mb": (load_info or {}).get("load_peak_vram_mb"),
        "warmup_time_s": (load_info or {}).get("warmup_time_s"),
        "inference_p50_s": percentile(inf, 0.50),
        "inference_p95_s": percentile(inf, 0.95),
        "inference_max_s": max(inf) if inf else None,
        "decode_plus_inference_p95_s": percentile(e2e, 0.95),
        "rtf_p50": percentile(rtf, 0.50),
        "rtf_p95": percentile(rtf, 0.95),
        "peak_process_vram_mb": max(peaks) if peaks else None,
        "positive_spans": len(details),
        "missing_gold_span_fraction": (
            sum(d["asr_word_count_in_gold"] == 0 for d in details) / len(details)
            if details else None
        ),
        "mean_timestamp_ceiling_tiou": mean_non_null(
            d["timestamp_ceiling_tiou"] for d in details
        ),
        "mean_content_token_recall": mean_non_null(
            d["content_token_recall"] for d in details
        ),
        "numeric_positive_spans": len(numeric_details),
        "mean_numeric_token_recall": mean_non_null(
            d["numeric_token_recall"] for d in numeric_details
        ),
        "retrieval_hit_at_1": mean_non_null(d["retrieval_hit_at_1"] for d in details),
        "retrieval_hit_at_3": mean_non_null(d["retrieval_hit_at_3"] for d in details),
        "retrieval_hit_at_5": mean_non_null(d["retrieval_hit_at_5"] for d in details),
        "mean_best_gold_window_tiou": mean_non_null(
            d["best_gold_window_tiou"] for d in details
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields: list[str] = []
    seen: set[str] = set()

    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def write_report(path: Path, summaries: list[dict[str, Any]], env: dict[str, Any]) -> None:
    lines = [
        "# ASR benchmark summary",
        "",
        "ASR-only comparison; no QA/NLI model is used.",
        "",
        "## Environment",
        "",
        "```json",
        json.dumps(env, indent=2, ensure_ascii=False),
        "```",
        "",
        "| Model | OK/Fail | P95 inference (s) | P95 RTF | Peak VRAM (MB) | Timestamp ceiling tIoU | Numeric recall | Retrieval H@1 | H@3 | H@5 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for s in summaries:
        lines.append(
            "| {model} | {ok}/{fail} | {p95} | {rtf} | {vram} | {tiou} | {num} | {h1} | {h3} | {h5} |".format(
                model=s["model"],
                ok=s["conversations_ok"],
                fail=s["conversations_failed"],
                p95=fmt(s["inference_p95_s"]),
                rtf=fmt(s["rtf_p95"]),
                vram=fmt(s["peak_process_vram_mb"], 0),
                tiou=fmt(s["mean_timestamp_ceiling_tiou"]),
                num=fmt(s["mean_numeric_token_recall"]),
                h1=fmt(s["retrieval_hit_at_1"]),
                h3=fmt(s["retrieval_hit_at_3"]),
                h5=fmt(s["retrieval_hit_at_5"]),
            )
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    preflight(args.device)

    data_dir = Path(args.data_dir)
    csv_path = Path(args.questions_csv) if args.questions_csv else data_dir / "question_train.csv"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = environment_snapshot()
    write_json(output_dir / "environment.json", env)

    rows = read_questions(csv_path)
    audio_items = discover_audio(data_dir, rows)

    if args.limit:
        audio_items = audio_items[: args.limit]

    selected_ids = {transcript_id for transcript_id, _ in audio_items}
    positive_rows = [
        row
        for row in rows
        if row["label"] == 1
        and row["evidence_start"] is not None
        and row["transcript_id"] in selected_ids
    ]

    options = {
        "medasr_chunk_length_s": args.medasr_chunk_length_s,
        "medasr_stride_length_s": args.medasr_stride_length_s,
        "whisper_beam_size": args.whisper_beam_size,
    }

    all_summaries: list[dict[str, Any]] = []
    all_details: list[dict[str, Any]] = []

    for model_name in args.models:
        print(
            f"\n=== {model_name} ({MODEL_SPECS[model_name]['model_id']}) ===",
            flush=True,
        )

        signature = cache_signature(model_name, options)
        model_dir = output_dir / model_name
        load_info_path = model_dir / "load_info.json"
        backend = None
        load_info: dict[str, Any] = {}

        usable_cache = True
        for transcript_id, _ in audio_items:
            cache_path = transcript_cache_path(output_dir, model_name, transcript_id)

            if args.force or not cache_path.exists():
                usable_cache = False
                break

            try:
                cached = load_json(cache_path)
                if cached.get("cache_signature") != signature:
                    usable_cache = False
                    break
            except Exception:
                usable_cache = False
                break

        if not usable_cache:
            try:
                with GpuMemoryMonitor(args.device) as monitor:
                    t0 = time.perf_counter()
                    backend = create_backend(
                        model_name,
                        device=args.device,
                        **options,
                    )
                    load_time = time.perf_counter() - t0

                load_info = {
                    "model": model_name,
                    "model_id": MODEL_SPECS[model_name]["model_id"],
                    "load_time_s": load_time,
                    "load_peak_vram_mb": monitor.peak_mb,
                    "load_incremental_vram_mb": monitor.incremental_peak_mb,
                    "cache_signature": signature,
                }

                print(f"loaded in {load_time:.2f}s", flush=True)

            except Exception as exc:
                write_json(
                    model_dir / "model_error.json",
                    {
                        "model": model_name,
                        "model_id": MODEL_SPECS[model_name]["model_id"],
                        "error": repr(exc),
                        "model_page": MODEL_SPECS[model_name]["page"],
                    },
                )

                print(f"MODEL LOAD FAILED: {exc}", flush=True)

                if args.strict:
                    raise

                continue

            if not args.skip_warmup and audio_items:
                try:
                    _, warm_path = audio_items[0]
                    audio, sr, _ = decode_audio_16k(warm_path)

                    with GpuMemoryMonitor(args.device) as monitor:
                        t0 = time.perf_counter()
                        backend.transcribe(audio, sr)
                        warmup_time = time.perf_counter() - t0

                    load_info["warmup_time_s"] = warmup_time
                    load_info["warmup_peak_vram_mb"] = monitor.peak_mb
                    print(f"warm-up {warmup_time:.2f}s", flush=True)

                except Exception as exc:
                    load_info["warmup_error"] = repr(exc)
                    print(f"warm-up failed (benchmark continues): {exc}", flush=True)

            write_json(load_info_path, load_info)

        elif load_info_path.exists():
            load_info = load_json(load_info_path)
            print("all requested transcripts already cached; model load skipped", flush=True)

        transcript_payloads: list[dict[str, Any]] = []

        for transcript_id, audio_path in audio_items:
            cache_path = transcript_cache_path(output_dir, model_name, transcript_id)

            if cache_path.exists() and not args.force:
                try:
                    cached = load_json(cache_path)
                    if cached.get("cache_signature") == signature:
                        validate_transcript_payload(cached)
                        transcript_payloads.append(cached)
                        print(f"  {transcript_id}: cached", flush=True)
                        continue
                except Exception:
                    pass

            if backend is None:
                raise RuntimeError("Backend unavailable after cache miss.")

            try:
                decode_t0 = time.perf_counter()
                audio, sr, duration_s = decode_audio_16k(audio_path)
                decode_s = time.perf_counter() - decode_t0

                with GpuMemoryMonitor(args.device) as monitor:
                    t0 = time.perf_counter()
                    result = backend.transcribe(audio, sr)
                    inference_s = time.perf_counter() - t0

                payload = {
                    "schema_version": SCHEMA_VERSION,
                    "cache_signature": signature,
                    "model": model_name,
                    "model_id": MODEL_SPECS[model_name]["model_id"],
                    "transcript_id": transcript_id,
                    "audio_file": str(audio_path),
                    "audio_duration_s": duration_s,
                    "timing": {
                        "decode_s": decode_s,
                        "inference_s": inference_s,
                        "decode_plus_inference_s": decode_s + inference_s,
                        "rtf": inference_s / duration_s if duration_s > 0 else None,
                    },
                    "memory": {
                        "baseline_process_vram_mb": monitor.baseline_mb,
                        "peak_process_vram_mb": monitor.peak_mb,
                        "incremental_peak_vram_mb": monitor.incremental_peak_mb,
                    },
                    "transcript": result.to_dict(),
                }

                validate_transcript_payload(payload)
                write_json(cache_path, payload)
                transcript_payloads.append(payload)

                print(
                    f"  {transcript_id}: {inference_s:.2f}s, "
                    f"RTF={payload['timing']['rtf']:.3f}, "
                    f"words={len(payload['transcript']['words'])}",
                    flush=True,
                )

            except Exception as exc:
                payload = {
                    "schema_version": SCHEMA_VERSION,
                    "cache_signature": signature,
                    "model": model_name,
                    "model_id": MODEL_SPECS[model_name]["model_id"],
                    "transcript_id": transcript_id,
                    "audio_file": str(audio_path),
                    "error": repr(exc),
                }

                write_json(cache_path, payload)
                transcript_payloads.append(payload)
                print(f"  {transcript_id}: FAILED: {exc}", flush=True)

                if args.strict:
                    raise

        by_id = {
            p["transcript_id"]: p
            for p in transcript_payloads
            if not p.get("error") and p.get("transcript")
        }

        details: list[dict[str, Any]] = []

        for row in positive_rows:
            transcript = by_id.get(row["transcript_id"])
            if transcript is None:
                continue

            details.append(
                evidence_detail(
                    model_name,
                    row,
                    transcript,
                    window_words=args.window_words,
                    window_stride=args.window_stride,
                    context_s=args.context_s,
                )
            )

        all_details.extend(details)

        summary = summarize_model(
            model_name,
            transcript_payloads,
            details,
            load_info,
        )
        all_summaries.append(summary)

        write_json(model_dir / "summary.json", summary)
        write_csv(model_dir / "evidence_details.csv", details)

        del backend

        try:
            import gc
            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    write_csv(output_dir / "model_summary.csv", all_summaries)
    write_json(
        output_dir / "model_summary.json",
        {"environment": env, "models": all_summaries},
    )
    write_csv(output_dir / "evidence_details_all_models.csv", all_details)
    write_report(output_dir / "report.md", all_summaries, env)

    print(f"\nWrote benchmark report to {output_dir / 'report.md'}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_SPECS),
        default=DEFAULT_MODELS,
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--questions-csv", default=None)
    parser.add_argument("--output-dir", default="benchmark_results/asr")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--medasr-chunk-length-s", type=float, default=20.0)
    parser.add_argument("--medasr-stride-length-s", type=float, default=2.0)
    parser.add_argument("--whisper-beam-size", type=int, default=5)
    parser.add_argument("--window-words", type=int, default=28)
    parser.add_argument("--window-stride", type=int, default=10)
    parser.add_argument("--context-s", type=float, default=2.0)

    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
