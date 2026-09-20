from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
from typing import Any, Iterable

ACCURACY_WEIGHT = 0.4
TIOU_WEIGHT = 0.6


def read_json(path: str | Path) -> Any:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def temporal_iou(
    gold_start: float | None,
    gold_end: float | None,
    pred_start: float | None,
    pred_end: float | None,
) -> float:
    if None in (gold_start, gold_end, pred_start, pred_end):
        return 0.0
    gs, ge, ps, pe = map(float, (gold_start, gold_end, pred_start, pred_end))
    if ge < gs or pe < ps:
        return 0.0
    intersection = max(0.0, min(ge, pe) - max(gs, ps))
    union = max(ge, pe) - min(gs, ps)
    return intersection / union if union > 0 else 0.0


def interval_overlaps(
    a_start: float | None,
    a_end: float | None,
    b_start: float | None,
    b_end: float | None,
) -> bool:
    if None in (a_start, a_end, b_start, b_end):
        return False
    return min(float(a_end), float(b_end)) > max(float(a_start), float(b_start))


def load_config(path: str | Path) -> dict[str, Any]:
    return dict(read_json(path))


def merge_config(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    merged.update(updates)
    return merged


def _cfg(config: dict[str, Any], key: str, default: Any) -> Any:
    return config.get(key, default)


def proposal_kind_group(kind: str) -> str:
    kind = (kind or "compact").lower()
    if kind.startswith("pause_"):
        return "pause"
    if kind.startswith("clause"):
        return "clause"
    if kind.startswith("retrieval"):
        return "retrieval"
    if kind.startswith("anchor"):
        return "anchor"
    if kind.startswith("fixed_"):
        try:
            width = int(kind.split("_", 1)[1])
        except Exception:
            width = 0
        return "fixed_long" if width >= 16 else "fixed_short"
    if kind.startswith("scan_"):
        return "scan"
    return kind


def cached_proposal_score(
    proposal: dict[str, Any],
    config: dict[str, Any],
) -> float:
    nli = proposal.get("nli", {})
    entailment = float(nli.get("entailment", 0.0))
    fact_fraction = float(proposal.get("exact_fact_fraction", 0.0))
    topic_coverage = float(proposal.get("topic_coverage", 0.0))
    word_count = int(proposal.get("word_count", 1))

    min_words = int(_cfg(config, "evidence_min_words", 3))
    max_words = int(_cfg(config, "evidence_max_words", 14))
    compactness = 1.0 - (
        (word_count - min_words) / max(1, max_words - min_words)
    )
    compactness = max(0.0, min(1.0, compactness))

    score = (
        float(_cfg(config, "evidence_entailment_weight", 0.60)) * entailment
        + float(_cfg(config, "evidence_fact_weight", 0.20)) * fact_fraction
        + float(_cfg(config, "evidence_topic_weight", 0.12)) * topic_coverage
        + float(_cfg(config, "evidence_compactness_weight", 0.08)) * compactness
    )

    if proposal.get("source") == "medasr":
        score += float(_cfg(config, "evidence_medasr_tie_bonus", 0.015))

    kind_biases = _cfg(config, "evidence_kind_bias", {}) or {}
    score += float(kind_biases.get(proposal_kind_group(str(proposal.get("kind", "compact"))), 0.0))

    if proposal.get("has_strict_contradiction"):
        score -= 2.0

    return score


def select_cached_evidence(
    event: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any] | None:
    proposals = list(event.get("proposals") or [])
    if not proposals:
        fallback = event.get("fallback_evidence")
        return dict(fallback) if fallback else None

    min_entailment = float(_cfg(config, "evidence_min_entailment", 0.48))
    scored: list[tuple[float, float, int, dict[str, Any]]] = []
    for proposal in proposals:
        entail = float((proposal.get("nli") or {}).get("entailment", 0.0))
        word_count = int(proposal.get("word_count", 1))
        scored.append((
            cached_proposal_score(proposal, config),
            entail,
            -word_count,
            proposal,
        ))

    eligible = [
        item for item in scored
        if item[1] >= min_entailment
        and not item[3].get("has_strict_contradiction", False)
    ]
    pool = eligible if eligible else scored
    if not pool:
        return None

    _, _, _, best = max(pool, key=lambda item: (item[0], item[1], item[2]))
    before = float(_cfg(config, "evidence_padding_before_s", 0.03))
    after = float(_cfg(config, "evidence_padding_after_s", 0.06))
    return {
        "source": best.get("source"),
        "start": max(0.0, float(best["raw_start"]) - before),
        "end": float(best["raw_end"]) + after,
        "start_word": best.get("start_word"),
        "end_word": best.get("end_word"),
        "kind": best.get("kind", "compact"),
        "score": cached_proposal_score(best, config),
    }


def score_records(
    records: Iterable[dict[str, Any]],
    decisions: Iterable[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    records = list(records)
    decisions = list(decisions)
    if len(records) != len(decisions):
        raise ValueError("records and decisions must have the same length")

    correct = 0
    positive_tious: list[float] = []
    by_type: dict[str, list[int]] = {}
    answered_yes_positive_tious: list[float] = []
    fp = fn = tp = tn = 0

    for record, decision in zip(records, decisions):
        label = int(record["label"])
        predicted = int(bool(decision["answer"]))
        correct += int(label == predicted)

        bucket = by_type.setdefault(str(record["question_type"]), [0, 0])
        bucket[0] += int(label == predicted)
        bucket[1] += 1

        if label == 1 and predicted == 1:
            tp += 1
        elif label == 1:
            fn += 1
        elif predicted == 1:
            fp += 1
        else:
            tn += 1

        if label == 1:
            iou = 0.0
            event_index = decision.get("event_index")
            if predicted == 1 and event_index is not None:
                events = record.get("events") or []
                if 0 <= int(event_index) < len(events):
                    evidence = select_cached_evidence(events[int(event_index)], config)
                    if evidence is not None:
                        iou = temporal_iou(
                            record.get("gold_start"),
                            record.get("gold_end"),
                            evidence.get("start"),
                            evidence.get("end"),
                        )
            positive_tious.append(iou)
            if predicted == 1:
                answered_yes_positive_tious.append(iou)

    total = len(records)
    accuracy = correct / total if total else 0.0
    mean_tiou = (
        sum(positive_tious) / len(positive_tious)
        if positive_tious else 0.0
    )
    final_score = ACCURACY_WEIGHT * accuracy + TIOU_WEIGHT * mean_tiou

    return {
        "questions": total,
        "correct": correct,
        "accuracy": accuracy,
        "mean_tiou": mean_tiou,
        "score": final_score,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "yes_precision": tp / (tp + fp) if tp + fp else 0.0,
        "yes_recall": tp / (tp + fn) if tp + fn else 0.0,
        "tiou_when_answered_yes": (
            sum(answered_yes_positive_tious) / len(answered_yes_positive_tious)
            if answered_yes_positive_tious else 0.0
        ),
        "by_type": {
            key: {
                "correct": value[0],
                "total": value[1],
                "accuracy": value[0] / value[1] if value[1] else 0.0,
            }
            for key, value in by_type.items()
        },
    }
