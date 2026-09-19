from __future__ import annotations

import os
import random
from typing import Any

import numpy as np


def seed_everything(seed: int, deterministic: bool = False) -> dict[str, Any]:
    """Seed Python/NumPy/PyTorch without importing torch unless installed."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    state: dict[str, Any] = {
        "seed": seed,
        "python": True,
        "numpy": True,
        "torch": False,
        "deterministic": deterministic,
    }
    try:
        import torch
    except ImportError:
        return state

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    state["torch"] = True
    return state


def dataloader_worker_init_fn(worker_id: int) -> None:
    """Deterministic worker seed derived from PyTorch's worker seed."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for DataLoader worker seeding") from exc
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
