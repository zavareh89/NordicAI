#!/usr/bin/env python3
"""Break positive false negatives down by failure stage.

Consumes the model-free development cache created in step 1.  The categories
are diagnostic, not mutually causal truth, but they tell you where the next
engineering effort should go.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any

from dev.common import interval_overlaps, load_config, read_json, write_csv, write_json
from dev.threshold_logic import decide_cached_record, event_metrics_for_debug


def _event_overlaps_gold(event: dict[str, Any], record: dict[str, Any]) -> bool:
    gs, ge = record.get("gold_start"), record.get("gold_end")
    if gs is None or ge is None:
        return False
    return any(
        interval_overlaps(
            member.get("candidate", {}).get("start"),
            member.get("candidate", {}).get("end"),
            gs,
            ge,
        )
        for member in event.get("members", [])
    )


def _paired_disagreement(event: dict[str, Any], config: dict[str, Any]) -> bool:
    members = list(event.get("members") or [])
    if len(members) != 2:
        return False
    floor = float(config.get("event_member_yes_floor", 0.48))
    yes_values = [float(member.get("yes_score", 0.0)) for member in members]
    no_values = [float(member.get("no_score", 0.0)) for member in members]
    strict = [bool(member.get("strict_contradiction", False)) for member in members]
    return (
        (max(yes_values) >= floor and max(no_values) >= floor)
        or any(strict) and not all(strict)
    )


def classify_false_negative(
    record: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    events = list(record.get("events") or [])
    gold_events = [
        (index, event)
        for index, event in enumerate(events)
        if _event_overlaps_gold(event, record)
    ]

    decision = decide_cached_record(record, config)
    if decision["answer"]:
        raise ValueError("record is not a false negative under this config")

    if not gold_events:
        stage = "retrieval_miss"
        detail = "No retrieved candidate from either ASR overlaps the annotated evidence span."
        best_gold_yes = 0.0
        best_gold_entailment = 0.0
        best_gold_event = None
    else:
        gold_metrics = [
            (index, event, event_metrics_for_debug(event, config))
            for index, event in gold_events
        ]
        best_gold_event = max(
            gold_metrics,
            key=lambda item: float(item[2]["yes_score"]),
        )
        best_gold_yes = float(best_gold_event[2]["yes_score"])
        best_gold_entailment = max(
            float((member.get("nli") or {}).get("entailment", 0.0))
            for _, event, _ in gold_metrics
            for member in event.get("members", [])
        )

        strict_on_gold = any(
            member.get("strict_contradiction", False)
            for _, event, _ in gold_metrics
            for member in event.get("members", [])
        )
        disagreement = any(
            _paired_disagreement(event, config)
            for _, event, _ in gold_metrics
        )

        all_event_metrics = [
            (index, event_metrics_for_debug(event, config))
            for index, event in enumerate(events)
        ]
        top_any = max(
            all_event_metrics,
            key=lambda item: float(item[1]["yes_score"]),
        ) if all_event_metrics else None

        if strict_on_gold:
            stage = "fact_guard_conflict"
            detail = "At least one gold-overlapping candidate carries a deterministic strict fact mismatch."
        elif disagreement:
            stage = "asr_disagreement"
            detail = "The ASRs disagree inside a gold-overlapping temporal event."
        elif best_gold_entailment < float(config.get("event_member_yes_floor", 0.48)):
            stage = "nli_or_semantic_support_weak"
            detail = "Gold evidence was retrieved, but NLI entailment/support is weak."
        elif top_any is not None and top_any[0] not in {index for index, _ in gold_events}:
            stage = "wrong_event_ranking"
            detail = "A non-gold event has a higher YES score than the retrieved gold event."
        else:
            stage = "consensus_threshold_rejection"
            detail = "Gold evidence has usable support, but current consensus thresholds still return NO."

    return {
        "question_id": record["question_id"],
        "transcript_id": record["transcript_id"],
        "question": record["question"],
        "stage": stage,
        "detail": detail,
        "best_gold_yes_score": best_gold_yes,
        "best_gold_entailment": best_gold_entailment,
        "decision_reason": decision["reason"],
        "decision_confidence": decision["confidence"],
        "gold_start": record.get("gold_start"),
        "gold_end": record.get("gold_end"),
        "best_gold_event_index": (
            best_gold_event[0] if best_gold_event is not None else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default="dev_cache/e1_dev_cache.json.gz")
    parser.add_argument("--config", default="config/e1_default.json")
    parser.add_argument("--output-dir", default="dev_results/fn_breakdown")
    args = parser.parse_args()

    cache = read_json(args.cache)
    config = load_config(args.config)
    rows: list[dict[str, Any]] = []

    for record in cache["records"]:
        if int(record["label"]) != 1:
            continue
        decision = decide_cached_record(record, config)
        if decision["answer"]:
            continue
        rows.append(classify_false_negative(record, config))

    counts = collections.Counter(row["stage"] for row in rows)
    summary = {
        "false_negatives": len(rows),
        "counts": dict(counts),
        "fractions": {
            key: value / len(rows) if rows else 0.0
            for key, value in counts.items()
        },
    }

    output_dir = Path(args.output_dir)
    write_csv(output_dir / "false_negatives.csv", rows)
    write_json(output_dir / "summary.json", summary)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
