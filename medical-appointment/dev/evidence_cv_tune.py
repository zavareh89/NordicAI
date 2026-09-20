#!/usr/bin/env python3
"""Cross-validated evidence-boundary/ranking tuning using cached proposals.

Run this only AFTER step 4 and after regenerating the development cache, so the
cache contains pause/clause/multi-scale proposals.  No model inference happens
inside this script.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dev.common import (
    load_config,
    read_json,
    select_cached_evidence,
    temporal_iou,
    write_csv,
    write_json,
)
from dev.threshold_logic import decide_cached_record

SCALAR_PARAMETERS = [
    "evidence_min_entailment",
    "evidence_entailment_weight",
    "evidence_fact_weight",
    "evidence_topic_weight",
    "evidence_compactness_weight",
    "evidence_medasr_tie_bonus",
]
KIND_GROUPS = [
    "pause",
    "clause",
    "retrieval",
    "anchor",
    "fixed_short",
    "fixed_long",
    "scan",
]


def _candidate_values(name: str, base: float) -> list[float]:
    fixed = {
        "evidence_min_entailment": [0.30, 0.38, 0.44, 0.48, 0.54, 0.60, 0.66],
        "evidence_entailment_weight": [0.35, 0.45, 0.55, 0.60, 0.68, 0.75],
        "evidence_fact_weight": [0.00, 0.08, 0.15, 0.20, 0.28, 0.36],
        "evidence_topic_weight": [0.00, 0.06, 0.12, 0.18, 0.25, 0.32],
        "evidence_compactness_weight": [-0.08, -0.04, 0.00, 0.03, 0.06, 0.10],
        "evidence_medasr_tie_bonus": [0.00, 0.01, 0.015, 0.025, 0.04, 0.06],
    }
    if name in fixed:
        return sorted(set(fixed[name] + [round(base, 4)]))
    if name.startswith("kind:"):
        return [-0.10, -0.06, -0.03, 0.0, 0.03, 0.06, 0.10, 0.14]
    return [base]


def _get_parameter(config: dict[str, Any], name: str) -> float:
    if name.startswith("kind:"):
        group = name.split(":", 1)[1]
        return float((config.get("evidence_kind_bias") or {}).get(group, 0.0))
    return float(config[name])


def _set_parameter(config: dict[str, Any], name: str, value: float) -> dict[str, Any]:
    out = dict(config)
    if name.startswith("kind:"):
        group = name.split(":", 1)[1]
        biases = dict(out.get("evidence_kind_bias") or {})
        biases[group] = value
        out["evidence_kind_bias"] = biases
    else:
        out[name] = value
    return out


def evidence_metrics(records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, float]:
    positive_count = 0
    yes_positive_count = 0
    tious_all: list[float] = []
    tious_yes: list[float] = []

    for record in records:
        if int(record["label"]) != 1:
            continue
        positive_count += 1
        decision = decide_cached_record(record, config)
        iou = 0.0
        if decision["answer"] and decision.get("event_index") is not None:
            yes_positive_count += 1
            event = record["events"][int(decision["event_index"])]
            evidence = select_cached_evidence(event, config)
            if evidence is not None:
                iou = temporal_iou(
                    record.get("gold_start"),
                    record.get("gold_end"),
                    evidence.get("start"),
                    evidence.get("end"),
                )
            tious_yes.append(iou)
        tious_all.append(iou)

    return {
        "positive_questions": float(positive_count),
        "answered_yes_positive": float(yes_positive_count),
        "mean_tiou_all_positives": (
            sum(tious_all) / len(tious_all) if tious_all else 0.0
        ),
        "mean_tiou_answered_yes": (
            sum(tious_yes) / len(tious_yes) if tious_yes else 0.0
        ),
    }


def coordinate_search(
    records: list[dict[str, Any]],
    base_config: dict[str, Any],
    parameters: list[str],
    passes: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = dict(base_config)
    history: list[dict[str, Any]] = []

    for pass_index in range(passes):
        changed = False
        for name in parameters:
            base_value = _get_parameter(config, name)
            best_value = base_value
            best_metrics = evidence_metrics(records, config)

            for value in _candidate_values(name, base_value):
                trial = _set_parameter(config, name, value)
                metrics = evidence_metrics(records, trial)
                history.append({
                    "pass": pass_index + 1,
                    "parameter": name,
                    "value": value,
                    "mean_tiou_all_positives": metrics["mean_tiou_all_positives"],
                    "mean_tiou_answered_yes": metrics["mean_tiou_answered_yes"],
                })
                key = (
                    metrics["mean_tiou_all_positives"],
                    metrics["mean_tiou_answered_yes"],
                )
                best_key = (
                    best_metrics["mean_tiou_all_positives"],
                    best_metrics["mean_tiou_answered_yes"],
                )
                if key > best_key:
                    best_value = value
                    best_metrics = metrics

            if best_value != base_value:
                config = _set_parameter(config, name, best_value)
                changed = True

        if not changed:
            break

    return config, history


def _folds(records: list[dict[str, Any]], k: int) -> list[set[str]]:
    conversations = sorted({str(record["transcript_id"]) for record in records})
    folds = [set() for _ in range(k)]
    for index, conversation in enumerate(conversations):
        folds[index % k].add(conversation)
    return folds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default="dev_cache/e1_dev_cache_step4.json.gz")
    parser.add_argument(
        "--config",
        default="config/e1_threshold_tuned.json",
        help="Use the threshold-tuned config if step 1 produced one; otherwise e1_default.json.",
    )
    parser.add_argument("--output-dir", default="dev_results/evidence_cv")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument(
        "--write-config",
        default="config/e1_evidence_tuned.json",
    )
    parser.add_argument(
        "--skip-kind-bias",
        action="store_true",
        help="Tune only scalar weights, not proposal-family biases.",
    )
    args = parser.parse_args()

    cache = read_json(args.cache)
    records = list(cache["records"])
    base = load_config(args.config)
    parameters = list(SCALAR_PARAMETERS)
    if not args.skip_kind_bias:
        parameters.extend(f"kind:{group}" for group in KIND_GROUPS)

    base_metrics = evidence_metrics(records, base)
    folds = _folds(records, max(2, args.folds))
    cv_rows: list[dict[str, Any]] = []
    history_all: list[dict[str, Any]] = []

    for fold_index, validation_ids in enumerate(folds, start=1):
        train = [r for r in records if str(r["transcript_id"]) not in validation_ids]
        validation = [r for r in records if str(r["transcript_id"]) in validation_ids]
        tuned, history = coordinate_search(train, base, parameters, args.passes)
        metrics = evidence_metrics(validation, tuned)
        cv_rows.append({
            "fold": fold_index,
            "validation_positive_questions": int(metrics["positive_questions"]),
            "validation_answered_yes_positive": int(metrics["answered_yes_positive"]),
            "mean_tiou_all_positives": metrics["mean_tiou_all_positives"],
            "mean_tiou_answered_yes": metrics["mean_tiou_answered_yes"],
            "evidence_parameters": json.dumps({
                name: _get_parameter(tuned, name)
                for name in parameters
            }, sort_keys=True),
        })
        for row in history:
            row = dict(row)
            row["fold"] = fold_index
            history_all.append(row)

    full_tuned, full_history = coordinate_search(records, base, parameters, args.passes)
    full_metrics = evidence_metrics(records, full_tuned)
    for row in full_history:
        row = dict(row)
        row["fold"] = "full"
        history_all.append(row)

    summary = {
        "base_metrics": base_metrics,
        "cv": {
            "folds": len(cv_rows),
            "mean_tiou_all_positives": (
                sum(row["mean_tiou_all_positives"] for row in cv_rows) / len(cv_rows)
            ),
            "mean_tiou_answered_yes": (
                sum(row["mean_tiou_answered_yes"] for row in cv_rows) / len(cv_rows)
            ),
        },
        "full_training_recommendation_metrics": full_metrics,
        "recommended_evidence_parameters": {
            name: _get_parameter(full_tuned, name)
            for name in parameters
        },
        "important": (
            "Use CV improvement to decide whether the boundary tuner generalizes. "
            "The full-training configuration is the deployment candidate."
        ),
    }

    output_dir = Path(args.output_dir)
    write_csv(output_dir / "cv_folds.csv", cv_rows)
    write_csv(output_dir / "search_history.csv", history_all)
    write_json(output_dir / "summary.json", summary)
    write_json(args.write_config, full_tuned)

    print(json.dumps(summary, indent=2))
    print(f"Wrote tuned config: {args.write_config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
