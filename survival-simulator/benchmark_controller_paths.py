from __future__ import annotations

import argparse
import random
import statistics
import time
from pathlib import Path

from src.utils.DTOs import StepResponse
from survival_policy import PolicyConfig, SurvivalController, load_default_config
from test_semantic_equivalence import _make_payload


def _stats(values_ms: list[float]) -> str:
    s = sorted(values_ms)
    def q(p: float) -> float:
        return s[min(len(s) - 1, int(p * (len(s) - 1)))]
    return (
        f"mean={statistics.fmean(s):.3f} ms "
        f"p50={q(.50):.3f} ms p95={q(.95):.3f} ms p99={q(.99):.3f} ms"
    )


def run(config: PolicyConfig, ticks: int, agents: int, seed: int) -> None:
    rng = random.Random(seed)
    payloads = [_make_payload(rng, i + 1, agents) for i in range(ticks)]

    # Warm-up independently because the controller is stateful.
    warm_ref = SurvivalController(config)
    warm_fast = SurvivalController(config)
    for p in payloads[:20]:
        warm_ref.decide_step(StepResponse.model_validate(p))
        warm_fast.decide_step(p)

    reference = SurvivalController(config)
    fast = SurvivalController(config)
    ref_ms: list[float] = []
    fast_ms: list[float] = []

    for p in payloads:
        t0 = time.perf_counter_ns()
        reference.decide_step(StepResponse.model_validate(p))
        ref_ms.append((time.perf_counter_ns() - t0) / 1e6)

        t0 = time.perf_counter_ns()
        fast.decide_step(p)
        fast_ms.append((time.perf_counter_ns() - t0) / 1e6)

    print("DTO/Pydantic path :", _stats(ref_ms))
    print("raw-dict path     :", _stats(fast_ms))
    print(f"mean speedup      : {statistics.fmean(ref_ms)/statistics.fmean(fast_ms):.2f}x")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--ticks", type=int, default=2000)
    ap.add_argument("--agents", type=int, default=20)
    ap.add_argument("--seed", type=int, default=123456)
    args = ap.parse_args()
    cfg = PolicyConfig.from_json(args.config) if args.config else load_default_config()
    run(cfg, args.ticks, args.agents, args.seed)


if __name__ == "__main__":
    main()
