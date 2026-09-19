from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from survival_policy import PolicyConfig


def robust_score(summary: dict[str, Any]) -> float:
    score = summary["aggregate"]["score"]
    return 0.8 * float(score["mean"]) + 0.2 * float(score["p10"])


def run_candidate(
    *,
    case_id: str,
    config_path: Path,
    results_root: Path,
    base_seed: int,
    n_seeds: int,
    n_cores: int,
    legacy_v1: bool,
) -> dict[str, Any]:
    if n_seeds != n_cores:
        raise ValueError(
            "The current evaluator maps one run to each worker. For holdout selection use "
            "--n-seeds 30 --n-cores 30 (or keep the two values equal)."
        )
    from evaluate import run_parallel

    cfg = PolicyConfig.from_json(config_path)
    if legacy_v1:
        values = cfg.to_dict()
        values["architecture_v2_enabled"] = False
        cfg = PolicyConfig.from_dict(values)
    out = results_root / case_id
    summary = run_parallel(
        base_seed=base_seed,
        config=cfg,
        output_dir=out,
        n_cores=n_cores,
    )
    metadata = {
        "case_id": case_id,
        "source_config": str(config_path),
        "base_seed": base_seed,
        "n_seeds": n_seeds,
        "n_cores": n_cores,
        "seeds": [base_seed + i for i in range(n_seeds)],
        "controller_config_hash": cfg.stable_hash,
        "legacy_v1": legacy_v1,
        "architecture_v2_enabled": cfg.architecture_v2_enabled,
        "robust_score": robust_score(summary) if summary.get("failed_runs", 0) == 0 else None,
    }
    (out / "holdout_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def compare_results(results_root: Path, *, expected_runs: int = 30) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    reference_seeds: list[int] | None = None
    for case_dir in sorted(p for p in results_root.iterdir() if p.is_dir()):
        summary_path = case_dir / "summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if int(summary.get("n_runs", 0)) != expected_runs or int(summary.get("failed_runs", 0)) != 0:
            continue
        seeds = [int(s) for s in summary.get("seeds", [])]
        if reference_seeds is None:
            reference_seeds = seeds
        elif seeds != reference_seeds:
            raise ValueError(
                f"{case_dir.name} was evaluated on different seeds. Holdout candidates must use identical seeds."
            )
        score = summary["aggregate"]["score"]
        survival = summary["aggregate"].get("survival_time", {})
        rows.append({
            "case_id": case_dir.name,
            "robust_score": robust_score(summary),
            "mean_score": float(score["mean"]),
            "median_score": float(score["median"]),
            "p10_score": float(score["p10"]),
            "min_score": float(score["min"]),
            "max_score": float(score["max"]),
            "mean_survival_time": float(survival.get("mean", 0.0)),
            "config_hash": summary.get("controller_config_hash"),
        })

    rows.sort(key=lambda r: (r["robust_score"], r["mean_score"], r["p10_score"]), reverse=True)
    fields = [
        "case_id", "robust_score", "mean_score", "median_score", "p10_score",
        "min_score", "max_score", "mean_survival_time", "config_hash",
    ]
    with (results_root / "holdout_leaderboard.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "selection_metric": "0.8 * mean_score + 0.2 * p10_score",
        "expected_runs_per_candidate": expected_runs,
        "common_seeds": reference_seeds or [],
        "ranking": rows,
        "best_case": rows[0]["case_id"] if rows else None,
    }
    (results_root / "holdout_comparison.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate shortlisted HPO configurations on a completely new common seed set."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run one candidate on the holdout seeds.")
    run.add_argument("--case-id", required=True)
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--results-root", type=Path, default=Path("results_holdout"))
    run.add_argument("--base-seed", type=int, default=50000)
    run.add_argument("--n-seeds", type=int, default=30)
    run.add_argument("--n-cores", type=int, default=30)
    run.add_argument(
        "--legacy-v1",
        action="store_true",
        help="Disable architecture-v2 additions so original HPO candidates are holdout-tested apples-to-apples.",
    )

    compare = sub.add_parser("compare", help="Compare all completed candidate folders.")
    compare.add_argument("--results-root", type=Path, default=Path("results_holdout"))
    compare.add_argument("--expected-runs", type=int, default=30)

    args = parser.parse_args()
    if args.command == "run":
        summary = run_candidate(
            case_id=args.case_id,
            config_path=args.config,
            results_root=args.results_root,
            base_seed=args.base_seed,
            n_seeds=args.n_seeds,
            n_cores=args.n_cores,
            legacy_v1=args.legacy_v1,
        )
        score = summary.get("aggregate", {}).get("score", {})
        print(
            f"{args.case_id}: mean={score.get('mean', float('nan')):.3f}, "
            f"p10={score.get('p10', float('nan')):.3f}, robust={robust_score(summary):.3f}"
        )
        if summary.get("failed_runs"):
            raise SystemExit(1)
    else:
        rows = compare_results(args.results_root, expected_runs=args.expected_runs)
        for i, row in enumerate(rows, 1):
            print(
                f"{i:>2}. {row['case_id']} robust={row['robust_score']:.3f} "
                f"mean={row['mean_score']:.3f} p10={row['p10_score']:.3f}"
            )


if __name__ == "__main__":
    main()
