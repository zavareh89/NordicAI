from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any


def _ffmpeg_decode(input_arg: list[str]) -> tuple[Any, int, float]:
    import numpy as np

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required. Install with: sudo apt install -y ffmpeg")

    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        *input_arg,
        "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", "16000", "pipe:1",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg decode failed: {stderr or 'unknown error'}")

    audio = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    if audio.size == 0:
        raise RuntimeError("Decoded zero audio samples")
    sample_rate = 16000
    return audio, sample_rate, float(audio.size) / sample_rate


def decode_audio_bytes_16k(audio_bytes: bytes) -> tuple[Any, int, float]:
    import numpy as np

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required. Install with: sudo apt install -y ffmpeg")
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", "pipe:0",
        "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", "16000", "pipe:1",
    ]
    proc = subprocess.run(
        cmd,
        input=audio_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg decode failed: {stderr or 'unknown error'}")
    audio = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    if audio.size == 0:
        raise RuntimeError("Decoded zero audio samples")
    sample_rate = 16000
    return audio, sample_rate, float(audio.size) / sample_rate


def decode_audio_path_16k(path: str | Path) -> tuple[Any, int, float]:
    return _ffmpeg_decode(["-i", str(path)])
