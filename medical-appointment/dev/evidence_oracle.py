#!/usr/bin/env python3
"""Measure how much tIoU is lost to evidence proposal generation vs ranking.

The development cache already contains every scored evidence proposal. This
script therefore runs without ASR/NLI inference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dev.common import (
    interval_overlaps,
    load_config,
    read_json,
    select_cached_evidence,
    temporal_iou,
    write_csv,
    write_json,
)
from dev.threshold_logic import decide_cached_record


def _proposal_iou(record: dict[str, Any], proposal: dict[str, Any], config: dict[str, Any]) -> float:
    before = float(config.get("evidence_padding_before_s", 0.03))
    after = float(config.get("evidence_padding_after_s", 0.06))
    return temporal_iou(
        record.get("gold_start"),
        record.get("gold_end"),
        max(0.0, float(proposal["raw_start"]) - before),
        float(proposal["raw_end"]) + after,
    )


def _event_overlaps_gold(event: dict[str, Any], record: dict[str, Any]) -> bool:
    return any(
        interval_overlaps(
            member.get("candidate", {}).get("start"),
            member.get("candidate", {}).get("end"),
            record.get("gold_start"),
            record.get("gold_end"),
        )
        for member in event.get("members", [])
    )


def analyze_record(record: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    decision = decide_cached_record(record, config)
    events = list(record.get("events") or [])

    selected_iou = 0.0
    selected_kind = None
    selected_event_oracle = 0.0
    selected_event_index = decision.get("event_index")
    if decision["answer"] and selected_event_index is not None:
        event = events[int(selected_event_index)]
        evidence = select_cached_evidence(event, config)
        if evidence:
            selected_kind = evidence.get("kind")
            selected_iou = temporal_iou(
                record.get("gold_start"),
                record.get("gold_end"),
                evidence.get("start"),
                evidence.get("end"),
            )
        selected_event_oracle = max(
            (_proposal_iou(record, p, config) for p in event.get("proposals", [])),
            default=0.0,
        )

    all_proposals = [
        (event_index, proposal)
        for event_index, event in enumerate(events)
        for proposal in event.get("proposals", [])
    ]
    best_all = max(
        all_proposals,
        key=lambda item: _proposal_iou(record, item[1], config),
        default=None,
    )
    oracle_all = (
        _proposal_iou(record, best_all[1], config)
        if best_all is not None else 0.0
    )

    gold_event_proposals = [
        (event_index, proposal)
        for event_index, event in enumerate(events)
        if _event_overlaps_gold(event, record)
        for proposal in event.get("proposals", [])
    ]
    best_gold_event = max(
        gold_event_proposals,
        key=lambda item: _proposal_iou(record, item[1], config),
        default=None,
    )
    oracle_gold_event = (
        _proposal_iou(record, best_gold_event[1], config)
        if best_gold_event is not None else 0.0
    )

    best_candidate_window_iou = max(
        (
            temporal_iou(
                record.get("gold_start"),
                record.get("gold_end"),
                member.get("candidate", {}).get("start"),
                member.get("candidate", {}).get("end"),
            )
            for event in events
            for member in event.get("members", [])
        ),
        default=0.0,
    )

    return {
        "question_id": record["question_id"],
        "transcript_id": record["transcript_id"],
        "question": record["question"],
        "answered_yes": bool(decision["answer"]),
        "decision_reason": decision["reason"],
        "selected_event_index": selected_event_index,
        "selected_kind": selected_kind,
        "selected_tiou": selected_iou,
        "selected_event_proposal_oracle_tiou": selected_event_oracle,
        "gold_event_proposal_oracle_tiou": oracle_gold_event,
        "all_proposal_oracle_tiou": oracle_all,
        "retrieval_window_oracle_tiou": best_candidate_window_iou,
        "best_oracle_kind": best_all[1].get("kind") if best_all else None,
        "best_oracle_source": best_all[1].get("source") if best_all else None,
        "best_oracle_event_index": best_all[0] if best_all else None,
        "ranking_gap": max(0.0, selected_event_oracle - selected_iou),
        "proposal_generation_gap_to_1": max(0.0, 1.0 - oracle_all),
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default="dev_cache/e1_dev_cache.json.gz")
    parser.add_argument("--config", default="config/e1_default.json")
    parser.add_argument("--output-dir", default="dev_results/evidence_oracle")
    args = parser.parse_args()

    cache = read_json(args.cache)
    config = load_config(args.config)
    rows = [
        analyze_record(record, config)
        for record in cache["records"]
        if int(record["label"]) == 1
    ]

    answered = [row for row in rows if row["answered_yes"]]
    oracle_values = [float(row["all_proposal_oracle_tiou"]) for row in rows]
    gold_event_oracle_values = [float(row["gold_event_proposal_oracle_tiou"]) for row in rows]
    selected_values = [float(row["selected_tiou"]) for row in rows]
    selected_answered = [float(row["selected_tiou"]) for row in answered]
    selected_event_oracles = [
        float(row["selected_event_proposal_oracle_tiou"])
        for row in answered
    ]

    summary = {
        "positive_questions": len(rows),
        "answered_yes": len(answered),
        "mean_selected_tiou_over_all_positives": _mean(selected_values),
        "mean_selected_tiou_when_answered_yes": _mean(selected_answered),
        "mean_selected_event_proposal_oracle_when_answered_yes": _mean(selected_event_oracles),
        "mean_gold_event_proposal_oracle": _mean(gold_event_oracle_values),
        "mean_all_proposal_oracle": _mean(oracle_values),
        "oracle_at_least_0_50": sum(v >= 0.50 for v in oracle_values) / len(oracle_values) if oracle_values else 0.0,
        "oracle_at_least_0_70": sum(v >= 0.70 for v in oracle_values) / len(oracle_values) if oracle_values else 0.0,
        "oracle_at_least_0_85": sum(v >= 0.85 for v in oracle_values) / len(oracle_values) if oracle_values else 0.0,
        "interpretation": {
            "large selected-event oracle minus selected": "ranking/scoring is the bottleneck",
            "large all-proposal oracle minus selected-event oracle": "event selection is the bottleneck",
            "low all-proposal oracle": "proposal generation/boundaries are the bottleneck",
        },
    }

    output_dir = Path(args.output_dir)
    write_csv(output_dir / "positive_questions.csv", rows)
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
