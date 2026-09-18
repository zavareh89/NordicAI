from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class PopulationSample:
    sim_time: float
    population: int
    score: float


@dataclass
class RunMetrics:
    seed: int
    score: float = 0.0
    survival_time: float = 0.0
    reached_3000_seconds: bool = False
    final_population: int = 0
    maximum_population: int = 0
    predator_deaths: int = 0
    starvation_deaths: int = 0
    total_herbivore_deaths: int = 0
    reproduction_count: int = 0
    runtime_seconds: float = 0.0
    status: str = "unknown"
    error: str | None = None
    population_trajectory: List[PopulationSample] = field(default_factory=list)

    def to_summary_dict(self) -> Dict[str, Any]:
        """Return the scalar fields used by parallel batch aggregation."""
        payload = asdict(self)
        payload.pop("population_trajectory", None)
        return payload

    def save(self, output_dir: str | Path) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        payload.pop("population_trajectory", None)
        (out / "metrics.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        with (out / "population_trajectory.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["sim_time", "population", "score"])
            writer.writeheader()
            for sample in self.population_trajectory:
                writer.writerow(asdict(sample))
