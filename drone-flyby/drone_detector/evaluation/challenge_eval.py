from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run_official_local_evaluator(
    challenge_root: str | Path,
    *,
    url: str = "http://localhost:9053/predict",
    scene: str = "helsinki",
    realtime: bool = False,
    verbose: bool = False,
) -> int:
    """Thin subprocess wrapper; it does not reimplement or alter official scoring."""
    challenge_root = Path(challenge_root).resolve()
    evaluator = challenge_root / "local_evaluator.py"
    if not evaluator.is_file():
        raise FileNotFoundError(f"Official local_evaluator.py not found at {evaluator}")
    command = [sys.executable, str(evaluator), "--url", url, "--scene", scene]
    if realtime:
        command.append("--realtime")
    if verbose:
        command.append("--verbose")
    return subprocess.run(command, cwd=challenge_root, check=False).returncode
