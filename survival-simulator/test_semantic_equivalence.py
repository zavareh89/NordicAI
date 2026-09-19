from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from src.utils.DTOs import StepResponse
from survival_policy import PolicyConfig, SurvivalController, load_default_config


def _make_payload(rng: random.Random, tick: int, n_agents: int) -> dict[str, Any]:
    statuses = []
    for aid in range(n_agents):
        observations: list[dict[str, Any]] = []
        for _ in range(rng.randint(0, 6)):
            observations.append(
                {
                    "type": "Fruit",
                    "distance": rng.uniform(5.0, 220.0),
                    "angle": rng.uniform(-1.5, 1.5),
                }
            )
        for _ in range(rng.randint(0, 3)):
            observations.append(
                {
                    "type": "Predator",
                    "distance": rng.uniform(10.0, 260.0),
                    "angle": rng.uniform(-1.5, 1.5),
                    "rel_dir": rng.uniform(-3.14, 3.14),
                }
            )
        for j in range(rng.randint(0, 4)):
            oid = (aid + j + 1) % max(1, n_agents)
            if oid == aid:
                continue
            observations.append(
                {
                    "type": "Agent",
                    "id": oid,
                    "distance": rng.uniform(5.0, 160.0),
                    "angle": rng.uniform(-1.5, 1.5),
                    "rel_dir": rng.uniform(-3.14, 3.14),
                }
            )
        if rng.random() < 0.7:
            x = rng.uniform(15.0, 120.0)
            observations.append(
                {
                    "type": "Edge",
                    "coords": [[x, -150.0], [x, 150.0]],
                }
            )
        statuses.append(
            {
                "agent_id": aid,
                "observations": observations,
                "energy": rng.uniform(80.0, 480.0),
                "biome": "grassland",
                "age": 5.0 + tick * 0.1,
                "speed": 10.0,
                "sprint_speed": 20.0,
                "hearing_radius": 50.0,
                "vision_angle": 1.0471975512,
                "vision_range": 200.0,
                "max_energy": 500.0,
            }
        )
    return {
        "game_status": "ok",
        "score": 0.0,
        "sim_time": tick * 0.1,
        "n_agents": n_agents,
        "agent_status": statuses,
    }


def _load_payloads(path: Path | None, ticks: int, agents: int, seed: int) -> list[dict[str, Any]]:
    if path is not None:
        payloads = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    payloads.append(json.loads(line))
        return payloads
    rng = random.Random(seed)
    return [_make_payload(rng, i + 1, agents) for i in range(ticks)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--payloads", type=Path, default=None, help="Optional NDJSON of real /predict payloads")
    ap.add_argument("--ticks", type=int, default=2000)
    ap.add_argument("--agents", type=int, default=20)
    ap.add_argument("--seed", type=int, default=123456)
    args = ap.parse_args()

    cfg = PolicyConfig.from_json(args.config) if args.config else load_default_config()
    payloads = _load_payloads(args.payloads, args.ticks, args.agents, args.seed)

    # Reference path: official Pydantic DTO -> existing controller.
    reference = SurvivalController(cfg)
    # Fast path: already-valid JSON dict -> the exact same controller code.
    fast = SurvivalController(cfg)

    for i, payload in enumerate(payloads):
        dto = StepResponse.model_validate(payload)
        expected = reference.decide_step(dto)
        actual = fast.decide_step(payload)
        if actual != expected:
            raise AssertionError(
                f"semantic mismatch at payload {i}\nexpected={expected!r}\nactual={actual!r}"
            )

    print(f"PASS: {len(payloads)} sequential payloads produced byte-for-byte identical action objects")


if __name__ == "__main__":
    main()
