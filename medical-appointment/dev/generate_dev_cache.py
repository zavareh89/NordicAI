#!/usr/bin/env python3
"""Generate one reusable development cache for E1 diagnostics/tuning.

This is intentionally an OFFLINE tool.  It runs the existing ASR + initial NLI
once, then also scores evidence proposals for every temporal event so later
threshold/evidence searches do not rerun MedASR, Parakeet or DeBERTa.

Typical use:

    python dev/generate_dev_cache.py \
      --config config/e1_default.json \
      --output dev_cache/e1_dev_cache.json.gz

The cache contains training labels/evidence, so never ship it as part of the
competition submission.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from e1.audio import decode_audio_path_16k
from e1.consensus import decide
from e1.evidence import build_evidence_proposals, fallback_evidence
from e1.pipeline import DualASRE1Pipeline
from e1.schemas import EvidenceEvent, EvidenceProposal, NLIResult
from utils import AUDIO_DIRECTORY, group_questions_by_conversation

from dev.common import write_json

CACHE_SCHEMA_VERSION = 2


def _member_dict(member: Any) -> dict[str, Any]:
    c = member.candidate
    return {
        "source": c.source,
        "yes_score": float(member.yes_score),
        "no_score": float(member.no_score),
        "exact_match": bool(member.facts.has_exact_match),
        "strict_contradiction": bool(member.facts.has_strict_contradiction),
        "matched_facts": list(member.facts.matched),
        "mismatched_facts": list(member.facts.mismatched),
        "nli": {
            "entailment": float(member.nli.entailment),
            "neutral": float(member.nli.neutral),
            "contradiction": float(member.nli.contradiction),
        },
        "relevance": float(c.relevance),
        "candidate": {
            "left": int(c.left),
            "right": int(c.right),
            "start": float(c.start),
            "end": float(c.end),
            "text": c.text,
            "rank": int(c.rank),
            "retrieval_score": float(c.retrieval_score),
            "raw_score": float(c.raw_score),
            "topic_overlap": float(c.topic_overlap),
            "fuzzy_overlap": float(c.fuzzy_overlap),
            "fact_type_overlap": float(c.fact_type_overlap),
        },
    }


def _pseudo_evidence_nli(proposal: EvidenceProposal) -> NLIResult:
    contradiction = 0.95 if proposal.has_strict_contradiction else 0.05
    entailment = max(
        0.05,
        min(
            0.95,
            0.35
            + 0.35 * float(proposal.topic_coverage)
            + 0.25 * float(proposal.exact_fact_fraction),
        ),
    )
    neutral = max(0.0, 1.0 - entailment - contradiction)
    return NLIResult(entailment, neutral, contradiction)


def _proposal_dict(
    proposal: EvidenceProposal,
    nli: NLIResult,
    transcript: Any,
) -> dict[str, Any]:
    start_word = int(proposal.start_word)
    end_word = int(proposal.end_word)
    return {
        "source": proposal.source,
        "kind": str(getattr(proposal, "kind", "compact")),
        "start_word": start_word,
        "end_word": end_word,
        "word_count": end_word - start_word + 1,
        "raw_start": float(transcript.words[start_word].start),
        "raw_end": float(transcript.words[end_word].end),
        "text": proposal.text,
        "topic_coverage": float(proposal.topic_coverage),
        "exact_fact_fraction": float(proposal.exact_fact_fraction),
        "has_strict_contradiction": bool(proposal.has_strict_contradiction),
        "heuristic_score": float(proposal.heuristic_score),
        "pause_threshold_s": getattr(proposal, "pause_threshold_s", None),
        "nli": {
            "entailment": float(nli.entailment),
            "neutral": float(nli.neutral),
            "contradiction": float(nli.contradiction),
        },
    }


def _fallback_dict(event: EvidenceEvent, transcripts: dict[str, Any], pipeline: Any) -> dict[str, Any] | None:
    members = list(event.members)
    if not members:
        return None
    best = max(members, key=lambda item: float(item.yes_score))
    transcript = transcripts.get(best.candidate.source)
    if transcript is None:
        return None
    span = fallback_evidence(best, transcript, pipeline.config)
    return {
        "source": span.source,
        "start": float(span.start),
        "end": float(span.end),
        "start_word": int(span.start_word),
        "end_word": int(span.end_word),
        "kind": "fallback",
    }


def _event_index(events: list[EvidenceEvent], chosen: EvidenceEvent | None) -> int | None:
    if chosen is None:
        return None
    for index, event in enumerate(events):
        if event is chosen:
            return index
    return None


def _conversation_records(
    pipeline: DualASRE1Pipeline,
    audio_filename: str,
    rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    audio_path = AUDIO_DIRECTORY / audio_filename
    audio, sample_rate, duration = decode_audio_path_16k(audio_path)
    questions = [row["question"] for row in rows]

    transcripts = pipeline._transcribe(audio, sample_rate)
    (
        _question_candidates,
        assessments_by_question,
        events_by_question,
    ) = pipeline._initial_assessments(questions, transcripts)

    decisions = [
        decide(question, assessments, events, pipeline.config)
        for question, assessments, events in zip(
            questions,
            assessments_by_question,
            events_by_question,
        )
    ]

    # Build evidence proposals for every event, not just the current YES event.
    # This makes later threshold sweeps counterfactual: a question that is NO
    # today can become YES without rerunning any model.
    flat_proposals: list[EvidenceProposal] = []
    flat_questions: list[str] = []
    flat_event_keys: list[tuple[int, int]] = []

    event_proposals: dict[tuple[int, int], list[EvidenceProposal]] = {}
    for q_index, (question, events) in enumerate(zip(questions, events_by_question)):
        for e_index, event in enumerate(events):
            proposals: list[EvidenceProposal] = []
            seen: set[tuple[str, int, int, str]] = set()
            for member in event.members:
                transcript = transcripts.get(member.candidate.source)
                if transcript is None:
                    continue
                for proposal in build_evidence_proposals(
                    question,
                    member,
                    transcript,
                    pipeline.config,
                ):
                    key = (
                        proposal.source,
                        int(proposal.start_word),
                        int(proposal.end_word),
                        str(getattr(proposal, "kind", "compact")),
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    proposals.append(proposal)
                    flat_proposals.append(proposal)
                    flat_questions.append(question)
                    flat_event_keys.append((q_index, e_index))
            event_proposals[(q_index, e_index)] = proposals

    if flat_proposals:
        try:
            flat_nli = pipeline.nli.predict(
                [proposal.text for proposal in flat_proposals],
                flat_questions,
            )
            if len(flat_nli) != len(flat_proposals):
                raise RuntimeError("unexpected evidence NLI result count")
        except Exception:
            flat_nli = [_pseudo_evidence_nli(proposal) for proposal in flat_proposals]
    else:
        flat_nli = []

    nli_by_event: dict[tuple[int, int], list[NLIResult]] = {}
    for key, nli in zip(flat_event_keys, flat_nli):
        nli_by_event.setdefault(key, []).append(nli)

    output: list[dict[str, Any]] = []
    for q_index, (row, question, assessments, events, decision) in enumerate(zip(
        rows,
        questions,
        assessments_by_question,
        events_by_question,
        decisions,
    )):
        serialized_events: list[dict[str, Any]] = []
        for e_index, event in enumerate(events):
            proposals = event_proposals.get((q_index, e_index), [])
            nlies = nli_by_event.get((q_index, e_index), [])
            serialized_proposals: list[dict[str, Any]] = []
            for proposal, nli in zip(proposals, nlies):
                transcript = transcripts[proposal.source]
                serialized_proposals.append(_proposal_dict(proposal, nli, transcript))

            serialized_events.append({
                "event_index": e_index,
                "temporal_iou": float(event.temporal_iou),
                "center_distance_s": (
                    None
                    if event.center_distance_s == float("inf")
                    else float(event.center_distance_s)
                ),
                "members": [_member_dict(member) for member in event.members],
                "proposals": serialized_proposals,
                "fallback_evidence": _fallback_dict(event, transcripts, pipeline),
            })

        output.append({
            "question_id": row["question_id"],
            "transcript_id": row["transcript_id"],
            "audio_filename": audio_filename,
            "audio_duration_s": float(duration),
            "question": question,
            "answer": row["answer"],
            "label": int(row["label"]),
            "question_type": row["question_type"],
            "gold_start": float(row["evidence_start"]) if row.get("evidence_start") else None,
            "gold_end": float(row["evidence_end"]) if row.get("evidence_end") else None,
            "baseline_decision": {
                "answer": bool(decision.answer),
                "confidence": float(decision.confidence),
                "reason": decision.reason,
                "event_index": _event_index(events, decision.event),
                "source": (
                    decision.candidate.candidate.source
                    if decision.candidate is not None
                    else None
                ),
            },
            "events": serialized_events,
        })

    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/e1_default.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="dev_cache/e1_dev_cache.json.gz")
    parser.add_argument("--limit-conversations", type=int, default=None)
    args = parser.parse_args()

    pipeline = DualASRE1Pipeline.from_config_file(args.config, device=args.device)
    conversations = list(group_questions_by_conversation())
    if args.limit_conversations:
        conversations = conversations[: args.limit_conversations]

    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, (audio_filename, rows) in enumerate(conversations, start=1):
        t0 = time.perf_counter()
        new_records = _conversation_records(pipeline, audio_filename, rows)
        records.extend(new_records)
        print(
            f"[{index}/{len(conversations)}] {audio_filename}: "
            f"{len(new_records)} questions in {time.perf_counter() - t0:.1f}s",
            flush=True,
        )

    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "source_config": pipeline.config.to_dict(),
        "records": records,
        "generation_seconds": time.perf_counter() - started,
    }
    write_json(args.output, payload)
    print(f"Wrote {len(records)} records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
