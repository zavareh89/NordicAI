from __future__ import annotations

import argparse
import random
import statistics
import time

import httpx

from test_semantic_equivalence import _make_payload


def q(values: list[float], p: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(p * (len(s) - 1)))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:9052/predict")
    ap.add_argument("--requests", type=int, default=2000)
    ap.add_argument("--agents", type=int, default=20)
    ap.add_argument("--seed", type=int, default=123456)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    payloads = [_make_payload(rng, i + 1, args.agents) for i in range(args.requests)]
    times: list[float] = []
    with httpx.Client(timeout=10.0, trust_env=False) as client:
        for p in payloads[:20]:
            r = client.post(args.url, json=p)
            r.raise_for_status()
        for p in payloads:
            t0 = time.perf_counter_ns()
            r = client.post(args.url, json=p)
            dt = (time.perf_counter_ns() - t0) / 1e6
            r.raise_for_status()
            times.append(dt)

    print(
        f"n={len(times)} mean={statistics.fmean(times):.3f} ms "
        f"p50={q(times,.50):.3f} p95={q(times,.95):.3f} "
        f"p99={q(times,.99):.3f} max={max(times):.3f}"
    )


if __name__ == "__main__":
    main()
