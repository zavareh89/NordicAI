from __future__ import annotations

import argparse
import statistics
import time

from survival_policy.config import PolicyConfig
from survival_policy.controller import SurvivalController


def synthetic_step(n_agents: int, tick: int) -> dict:
    statuses = []
    for i in range(n_agents):
        phase = (i % 8) * 0.35
        observations = [
            {"type": "Fruit", "distance": 55.0 + (i % 5) * 12.0, "angle": -0.55 + 0.13 * (i % 7)},
            {"type": "Fruit", "distance": 100.0 + (i % 4) * 15.0, "angle": 0.75 - 0.09 * (i % 5)},
            {"type": "Predator", "distance": 130.0 + (i % 6) * 16.0, "angle": 0.45 + 0.05 * (i % 4), "rel_dir": -0.20 + phase},
            {"type": "Edge", "coords": ((75.0, -120.0), (75.0, 120.0))},
            {"type": "Edge", "coords": ((-90.0, -120.0), (-90.0, 120.0))},
        ]
        if n_agents > 1:
            observations.append({
                "type": "Agent", "id": (i + 1) % n_agents,
                "distance": 35.0 + (i % 4) * 8.0, "angle": -0.6 + 0.07 * i,
                "rel_dir": 0.2,
            })
        statuses.append({
            "agent_id": i,
            "observations": observations,
            "energy": 260.0 + (i % 6) * 20.0,
            "biome": "grassland",
            "age": 20.0 + i,
            "speed": 10.0,
            "sprint_speed": 18.0,
            "hearing_radius": 50.0,
            "vision_angle": 1.2,
            "vision_range": 220.0,
            "max_energy": 500.0,
        })
    return {"game_status": "ok", "score": 0.0, "sim_time": tick * 0.1, "n_agents": n_agents, "agent_status": statuses}


def timed(controller: SurvivalController, *, steps: list[dict], n_agents: int, repeats: int) -> list[float]:
    samples = []
    for _ in range(repeats):
        controller.reset()
        start = time.perf_counter()
        for step in steps:
            controller.decide_step(step)
        samples.append(time.perf_counter() - start)
    return samples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/C376_v3_hpo_base.json")
    ap.add_argument("--ticks", type=int, default=250)
    ap.add_argument("--agents", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    base = PolicyConfig.from_json(args.config)
    v2d = base.to_dict(); v2d["architecture_v3_enabled"] = False
    v2 = SurvivalController(PolicyConfig.from_dict(v2d))
    v3 = SurvivalController(base)

    # warm up interpreter/cache effects
    v2.decide_step(synthetic_step(args.agents, 0))
    v3.decide_step(synthetic_step(args.agents, 0))

    steps = [synthetic_step(args.agents, t) for t in range(args.ticks)]
    t2 = timed(v2, steps=steps, n_agents=args.agents, repeats=args.repeats)
    t3 = timed(v3, steps=steps, n_agents=args.agents, repeats=args.repeats)
    m2, m3 = statistics.median(t2), statistics.median(t3)
    overhead = (m3 / m2 - 1.0) * 100.0
    print(f"v2_median_s={m2:.6f}")
    print(f"v3_median_s={m3:.6f}")
    print(f"v2_us_per_agent={(m2 / (args.ticks * args.agents)) * 1e6:.2f}")
    print(f"v3_us_per_agent={(m3 / (args.ticks * args.agents)) * 1e6:.2f}")
    print(f"overhead_percent={overhead:.2f}")
    if overhead > 60.0:
        raise SystemExit("FAIL: v3 controller overhead exceeds 60% target")


if __name__ == "__main__":
    main()
