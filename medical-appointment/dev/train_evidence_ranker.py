#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from dev.ranker_training import fit_ridge, fold_for, predict
from e1.evidence_ranker import FEATURE_NAMES, feature_dict_from_values, vector_from_dict
from e1.boundary_calibration import BoundaryCalibrator
from e1.schemas import EvidenceSpan


def read_json(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def temporal_iou(gs, ge, ps, pe):
    if None in (gs, ge, ps, pe):
        return 0.0
    gs, ge, ps, pe = map(float, (gs, ge, ps, pe))
    inter = max(0.0, min(ge, pe) - max(gs, ps))
    union = max(ge, pe) - min(gs, ps)
    return inter / union if union > 0 else 0.0


def member_for_source(event, source):
    for member in event.get("members") or []:
        if (member.get("candidate") or {}).get("source") == source:
            return member
        if member.get("source") == source:
            return member
    # Current dev-cache format stores source at member top-level.
    for member in event.get("members") or []:
        if member.get("source") == source:
            return member
    return None


def proposal_features(proposal, event):
    source = proposal["source"]
    member = member_for_source(event, source) or {}
    candidate = member.get("candidate") or {}
    nli = proposal.get("nli") or {}
    return feature_dict_from_values(
        nli_entailment=float(nli.get("entailment", 0.0)),
        nli_neutral=float(nli.get("neutral", 0.0)),
        nli_contradiction=float(nli.get("contradiction", 0.0)),
        topic_coverage=float(proposal.get("topic_coverage", 0.0)),
        exact_fact_fraction=float(proposal.get("exact_fact_fraction", 0.0)),
        heuristic_score=float(proposal.get("heuristic_score", 0.0)),
        word_count=int(proposal.get("word_count", 1)),
        duration_s=max(0.0, float(proposal["raw_end"]) - float(proposal["raw_start"])),
        parent_yes_score=float(member.get("yes_score", 0.0)),
        parent_no_score=float(member.get("no_score", 0.0)),
        parent_retrieval_score=float(candidate.get("retrieval_score", 0.0)),
        parent_relevance=float(member.get("relevance", 0.0)),
        parent_topic_overlap=float(candidate.get("topic_overlap", 0.0)),
        parent_fuzzy_overlap=float(candidate.get("fuzzy_overlap", 0.0)),
        parent_fact_type_overlap=float(candidate.get("fact_type_overlap", 0.0)),
        source=source,
        kind=str(proposal.get("kind", "")),
        pause_threshold_s=proposal.get("pause_threshold_s"),
    )


def calibrated_interval(proposal, calibrator):
    span = EvidenceSpan(
        source=proposal["source"],
        start_word=int(proposal["start_word"]),
        end_word=int(proposal["end_word"]),
        start=float(proposal["raw_start"]),
        end=float(proposal["raw_end"]),
    )
    out = calibrator.apply(span)
    return out.start, out.end


def build_questions(records, calibrator):
    questions = []
    for record in records:
        decision = record.get("baseline_decision") or {}
        if int(record.get("label", 0)) != 1 or not bool(decision.get("answer")):
            continue
        event_index = decision.get("event_index")
        events = record.get("events") or []
        if event_index is None or not (0 <= int(event_index) < len(events)):
            continue
        event = events[int(event_index)]
        rows = []
        for proposal in event.get("proposals") or []:
            if proposal.get("has_strict_contradiction"):
                continue
            x = vector_from_dict(proposal_features(proposal, event))
            ps, pe = calibrated_interval(proposal, calibrator)
            y = temporal_iou(
                record.get("gold_start"), record.get("gold_end"), ps, pe
            )
            rows.append((x, float(y), proposal))
        if rows:
            questions.append(
                {
                    "transcript_id": record["transcript_id"],
                    "question_id": record["question_id"],
                    "rows": rows,
                }
            )
    return questions


def flatten(questions):
    X = np.stack([row[0] for q in questions for row in q["rows"]])
    y = np.asarray([row[1] for q in questions for row in q["rows"]], dtype=np.float64)
    return X, y


def evaluate(questions, model):
    means, scales, weights, intercept = model
    scores = []
    for q in questions:
        X = np.stack([row[0] for row in q["rows"]])
        pred = predict(X, means, scales, weights, intercept)
        best_index = int(np.argmax(pred))
        scores.append(q["rows"][best_index][1])
    return sum(scores) / len(scores) if scores else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="dev_cache/e1_dev_cache.json.gz")
    parser.add_argument(
        "--calibration", default="config/e2_1_boundary_calibration.json"
    )
    parser.add_argument("--output", default="config/e2_1_evidence_ranker.json")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--alphas", default="0.01,0.1,1,10,100")
    args = parser.parse_args()

    payload = read_json(args.cache)
    calibrator = BoundaryCalibrator.from_json(args.calibration)
    questions = build_questions(payload["records"], calibrator)
    if not questions:
        raise RuntimeError(
            "No positive selected-event proposals. Regenerate dev cache with your final E1 config."
        )

    alphas = [float(x) for x in args.alphas.split(",")]
    alpha_scores = {}

    for alpha in alphas:
        fold_scores = []
        for fold in range(args.folds):
            train = [
                q for q in questions
                if fold_for(q["transcript_id"], args.folds) != fold
            ]
            test = [
                q for q in questions
                if fold_for(q["transcript_id"], args.folds) == fold
            ]
            if not train or not test:
                continue
            X, y = flatten(train)
            model = fit_ridge(X, y, alpha)
            fold_scores.append(evaluate(test, model))
        alpha_scores[alpha] = sum(fold_scores) / len(fold_scores) if fold_scores else 0.0

    best_alpha = max(alpha_scores, key=alpha_scores.get)
    X, y = flatten(questions)
    means, scales, weights, intercept = fit_ridge(X, y, best_alpha)
    fit_score = evaluate(questions, (means, scales, weights, intercept))
    oracle = sum(max(row[1] for row in q["rows"]) for q in questions) / len(questions)

    output = {
        "schema_version": 1,
        "feature_names": FEATURE_NAMES,
        "means": means.tolist(),
        "scales": scales.tolist(),
        "weights": weights.tolist(),
        "intercept": float(intercept),
        "metadata": {
            "alpha": float(best_alpha),
            "cv_mean_tiou": float(alpha_scores[best_alpha]),
            "alpha_scores": {str(k): float(v) for k, v in alpha_scores.items()},
            "fit_mean_tiou": float(fit_score),
            "oracle_mean_tiou": float(oracle),
            "questions": len(questions),
            "proposals": int(len(y)),
            "calibration": args.calibration,
        },
    }
    Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output["metadata"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
