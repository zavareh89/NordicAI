from __future__ import annotations

from typing import Any


def _cfg(config: dict[str, Any], key: str, default: Any) -> Any:
    return config.get(key, default)


def _event_metrics(
    event: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    members = list(event.get("members") or [])
    if not members:
        raise ValueError("event contains no members")

    yes_values = [float(member.get("yes_score", 0.0)) for member in members]
    no_values = [float(member.get("no_score", 0.0)) for member in members]
    exact_flags = [bool(member.get("exact_match", False)) for member in members]
    strict_flags = [bool(member.get("strict_contradiction", False)) for member in members]

    best_index = max(range(len(members)), key=lambda i: yes_values[i])
    best = members[best_index]

    dual_yes = (
        len(members) == 2
        and all(
            value >= float(_cfg(config, "event_member_yes_floor", 0.48))
            for value in yes_values
        )
        and not any(strict_flags)
    )

    if dual_yes:
        yes_score = (
            0.5 * (yes_values[0] + yes_values[1])
            + float(_cfg(config, "dual_agreement_bonus", 0.05))
        )
    elif len(members) == 2:
        high = max(yes_values)
        other = min(yes_values)
        neutral_other = any(
            float((member.get("nli") or {}).get("neutral", 0.0))
            >= max(
                float((member.get("nli") or {}).get("entailment", 0.0)),
                float((member.get("nli") or {}).get("contradiction", 0.0)),
            )
            for member in members
        )
        yes_score = high + (
            float(_cfg(config, "paired_neutral_bonus", 0.02))
            if neutral_other
            else 0.20 * other
        )
    else:
        yes_score = yes_values[0]

    if len(no_values) == 2:
        no_score = 0.65 * max(no_values) + 0.35 * min(no_values)
    else:
        no_score = no_values[0]

    relevance = max(float(member.get("relevance", 0.0)) for member in members)

    return {
        "members": members,
        "best_member": best,
        "yes_score": min(1.0, yes_score),
        "no_score": min(1.0, no_score),
        "dual_yes": dual_yes,
        "any_exact_match": any(exact_flags),
        "any_strict_contradiction": any(strict_flags),
        "all_strict_contradiction": all(strict_flags),
        "relevance": relevance,
    }


def _relevant(metrics: dict[str, Any], config: dict[str, Any]) -> bool:
    return (
        float(metrics["relevance"])
        >= float(_cfg(config, "minimum_relevance_for_yes", 0.10))
        or bool(metrics["any_exact_match"])
    )


def _choose_source(metrics: dict[str, Any], config: dict[str, Any]) -> str:
    members = metrics["members"]
    if len(members) == 1:
        return str(members[0]["source"])

    med = next((m for m in members if m.get("source") == "medasr"), None)
    par = next((m for m in members if m.get("source") == "parakeet_v3"), None)
    if med is None:
        return str(par["source"])
    if par is None:
        return str(med["source"])

    tolerance = float(_cfg(config, "medasr_evidence_preference_tolerance", 0.05))
    if float(med.get("yes_score", 0.0)) >= float(par.get("yes_score", 0.0)) - tolerance:
        return "medasr"
    return "parakeet_v3"


def decide_cached_record(
    record: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    events = list(record.get("events") or [])
    if not events:
        return {
            "answer": False,
            "confidence": 0.0,
            "event_index": None,
            "source": None,
            "reason": "no_asr_candidates",
        }

    scored = [
        (index, event, _event_metrics(event, config))
        for index, event in enumerate(events)
    ]
    scored.sort(
        key=lambda item: (float(item[2]["yes_score"]), float(item[2]["relevance"])),
        reverse=True,
    )

    # 1. Both ASRs support the same spoken event.
    for index, _, metrics in scored:
        if (
            metrics["dual_yes"]
            and float(metrics["yes_score"])
            >= float(_cfg(config, "dual_event_yes_threshold", 0.56))
            and _relevant(metrics, config)
        ):
            return {
                "answer": True,
                "confidence": float(metrics["yes_score"]),
                "event_index": index,
                "source": _choose_source(metrics, config),
                "reason": "paired_dual_yes",
            }

    # 2. Exact fact support. A strict contradiction from the other ASR raises
    # the threshold rather than globally vetoing this event.
    for index, _, metrics in scored:
        exact_members = [
            member for member in metrics["members"]
            if member.get("exact_match")
            and not member.get("strict_contradiction")
        ]
        if not exact_members:
            continue
        best_exact = max(exact_members, key=lambda member: float(member.get("yes_score", 0.0)))
        threshold = (
            float(_cfg(config, "asr_fact_disagreement_yes_threshold", 0.64))
            if metrics["any_strict_contradiction"]
            else float(_cfg(config, "exact_fact_yes_threshold", 0.58))
        )
        if (
            float(best_exact.get("yes_score", 0.0)) >= threshold
            and _relevant(metrics, config)
        ):
            return {
                "answer": True,
                "confidence": float(best_exact.get("yes_score", 0.0)),
                "event_index": index,
                "source": str(best_exact.get("source")),
                "reason": (
                    "paired_asr_fact_disagreement_yes"
                    if metrics["any_strict_contradiction"]
                    else "exact_fact_yes"
                ),
            }

    # 3. Strong one-source support with no exact contradiction on that source.
    for index, _, metrics in scored:
        best = metrics["best_member"]
        if best.get("strict_contradiction") or not _relevant(metrics, config):
            continue
        if (
            float(best.get("yes_score", 0.0))
            >= float(_cfg(config, "single_source_yes_threshold", 0.62))
            and float(metrics["no_score"])
            <= float(best.get("yes_score", 0.0))
            + float(_cfg(config, "event_yes_no_margin", 0.03))
        ):
            return {
                "answer": True,
                "confidence": float(best.get("yes_score", 0.0)),
                "event_index": index,
                "source": str(best.get("source")),
                "reason": "strong_event_yes",
            }

    # 4. Generic disagreement recovery.
    for index, _, metrics in scored:
        best = metrics["best_member"]
        if (
            float(best.get("yes_score", 0.0))
            >= float(_cfg(config, "generic_disagreement_yes_threshold", 0.68))
            and not best.get("strict_contradiction")
            and _relevant(metrics, config)
        ):
            return {
                "answer": True,
                "confidence": float(best.get("yes_score", 0.0)),
                "event_index": index,
                "source": str(best.get("source")),
                "reason": "strong_generic_disagreement_yes",
            }

    # 5. Final high-entailment recovery.
    for index, _, metrics in scored:
        best = metrics["best_member"]
        if (
            float(best.get("yes_score", 0.0))
            >= float(_cfg(config, "strong_entailment_threshold", 0.66))
            and not best.get("strict_contradiction")
            and _relevant(metrics, config)
        ):
            return {
                "answer": True,
                "confidence": float(best.get("yes_score", 0.0)),
                "event_index": index,
                "source": str(best.get("source")),
                "reason": "strong_entailment_recovery",
            }

    strict = [item for item in scored if item[2]["all_strict_contradiction"]]
    if strict:
        index, _, metrics = max(strict, key=lambda item: float(item[2]["no_score"]))
        return {
            "answer": False,
            "confidence": float(metrics["no_score"]),
            "event_index": index,
            "source": None,
            "reason": "strict_fact_contradiction",
        }

    index, _, metrics = max(scored, key=lambda item: float(item[2]["no_score"]))
    return {
        "answer": False,
        "confidence": max(0.5, float(metrics["no_score"])),
        "event_index": index,
        "source": None,
        "reason": "no_supporting_event",
    }


def event_metrics_for_debug(
    event: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Public diagnostic wrapper used by false-negative analysis."""
    return _event_metrics(event, config)
