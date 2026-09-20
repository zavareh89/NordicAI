from __future__ import annotations

from dataclasses import dataclass

from .config import E1Config
from .facts import compare_facts, extract_facts
from .schemas import (
    CandidateAssessment,
    CandidateWindow,
    EvidenceEvent,
    NLIResult,
    QuestionDecision,
)


@dataclass(frozen=True)
class EventScore:
    event: EvidenceEvent
    yes_score: float
    no_score: float
    best_yes: CandidateAssessment
    dual_yes: bool
    any_exact_match: bool
    any_strict_contradiction: bool
    all_strict_contradiction: bool
    relevance: float


def assess_candidate(
    question: str,
    candidate: CandidateWindow,
    nli: NLIResult,
    config: E1Config,
) -> CandidateAssessment:
    q_facts = extract_facts(question)
    comparison = compare_facts(q_facts, candidate.facts)

    entail = nli.entailment
    contradiction = nli.contradiction

    # Exact-value contradictions remain the strongest deterministic rule.
    if comparison.has_strict_contradiction:
        contradiction = max(
            contradiction,
            config.guard_contradiction_probability,
        )
        entail = min(
            entail,
            1.0 - config.guard_contradiction_probability,
        )
    elif comparison.has_exact_match:
        entail = min(
            1.0,
            entail + config.fact_match_entailment_bonus,
        )

    # E1 v1 used 0.65 + 0.35*retrieval, which could suppress a valid NLI hit
    # simply because several overlapping retrieval windows had similar scores.
    quality = (
        config.assessment_retrieval_floor
        + (1.0 - config.assessment_retrieval_floor)
        * candidate.retrieval_score
    )

    yes_score = entail * quality
    no_score = contradiction * quality

    if comparison.has_strict_contradiction:
        no_score = max(
            no_score,
            config.guard_contradiction_probability,
        )

    return CandidateAssessment(
        candidate=candidate,
        nli=NLIResult(entail, nli.neutral, contradiction),
        facts=comparison,
        yes_score=min(1.0, yes_score),
        no_score=min(1.0, no_score),
    )


def _event_score(
    event: EvidenceEvent,
    config: E1Config,
) -> EventScore:
    members = list(event.members)
    if not members:
        raise ValueError("EvidenceEvent has no members")

    best_yes = max(members, key=lambda item: item.yes_score)
    yes_values = [item.yes_score for item in members]
    no_values = [item.no_score for item in members]

    strict = [
        item.facts.has_strict_contradiction
        for item in members
    ]
    exact = [
        item.facts.has_exact_match
        for item in members
    ]

    dual_yes = (
        len(members) == 2
        and all(
            score >= config.event_member_yes_floor
            for score in yes_values
        )
        and not any(strict)
    )

    if dual_yes:
        yes_score = (
            0.5 * (yes_values[0] + yes_values[1])
            + config.dual_agreement_bonus
        )
    elif len(members) == 2:
        # One source may be strongly supportive while the other is merely
        # neutral. That should increase confidence slightly, not veto the YES.
        high = max(yes_values)
        other = min(yes_values)
        neutral_other = any(
            item.nli.neutral >= max(
                item.nli.entailment,
                item.nli.contradiction,
            )
            for item in members
        )
        yes_score = high + (
            config.paired_neutral_bonus
            if neutral_other
            else 0.20 * other
        )
    else:
        yes_score = yes_values[0]

    # Generic NLI contradictions do not automatically veto another passage.
    # Deterministic fact contradictions are handled explicitly in decide().
    if len(no_values) == 2:
        no_score = 0.65 * max(no_values) + 0.35 * min(no_values)
    else:
        no_score = no_values[0]

    relevance = max(item.candidate.relevance for item in members)

    return EventScore(
        event=event,
        yes_score=min(1.0, yes_score),
        no_score=min(1.0, no_score),
        best_yes=best_yes,
        dual_yes=dual_yes,
        any_exact_match=any(exact),
        any_strict_contradiction=any(strict),
        all_strict_contradiction=all(strict),
        relevance=relevance,
    )


def _relevant_for_yes(score: EventScore, config: E1Config) -> bool:
    return (
        score.relevance >= config.minimum_relevance_for_yes
        or score.any_exact_match
    )


def _choose_event_candidate(
    event: EvidenceEvent,
    config: E1Config,
) -> CandidateAssessment:
    members = list(event.members)
    if len(members) == 1:
        return members[0]

    med = event.medasr
    par = event.parakeet
    assert med is not None and par is not None

    # Prefer MedASR only when support is genuinely close; otherwise preserve
    # the stronger lexical/NLI source. Evidence refinement later evaluates both.
    if (
        med.yes_score
        >= par.yes_score - config.medasr_evidence_preference_tolerance
    ):
        return med
    return par


def decide(
    question: str,
    assessments: list[CandidateAssessment],
    events: list[EvidenceEvent],
    config: E1Config,
) -> QuestionDecision:
    """Recall-oriented event-level E1 v2 consensus.

    The crucial difference from E1 v1 is that a contradiction in one temporal
    event cannot veto an entailment in a different event. Exact numeric/value
    contradictions remain strict inside the event that contains them.
    """

    if not assessments or not events:
        return QuestionDecision(
            False,
            0.0,
            None,
            "no_asr_candidates",
        )

    scored = sorted(
        (_event_score(event, config) for event in events),
        key=lambda item: (item.yes_score, item.relevance),
        reverse=True,
    )

    # 1) Best case: both ASRs support the same spoken event.
    for item in scored:
        if (
            item.dual_yes
            and item.yes_score >= config.dual_event_yes_threshold
            and _relevant_for_yes(item, config)
        ):
            chosen = _choose_event_candidate(item.event, config)
            return QuestionDecision(
                True,
                item.yes_score,
                chosen,
                "paired_dual_yes",
                event=item.event,
            )

    # 2) Exact fact support is especially valuable on this challenge. If one
    # ASR contains the exact queried value and supports the statement, retain it.
    # A conflicting value from the paired ASR is treated as an ASR disagreement,
    # not as an automatic global NO.
    for item in scored:
        exact_members = [
            member
            for member in item.event.members
            if member.facts.has_exact_match
            and not member.facts.has_strict_contradiction
        ]
        if not exact_members:
            continue

        best_exact = max(exact_members, key=lambda member: member.yes_score)

        threshold = (
            config.asr_fact_disagreement_yes_threshold
            if item.any_strict_contradiction
            else config.exact_fact_yes_threshold
        )

        if (
            best_exact.yes_score >= threshold
            and _relevant_for_yes(item, config)
        ):
            return QuestionDecision(
                True,
                best_exact.yes_score,
                best_exact,
                (
                    "paired_asr_fact_disagreement_yes"
                    if item.any_strict_contradiction
                    else "exact_fact_yes"
                ),
                event=item.event,
            )

    # 3) One source strongly supports an event and the paired source is absent
    # or neutral. This is where E1 v1 lost many positives.
    for item in scored:
        best = item.best_yes
        if best.facts.has_strict_contradiction:
            continue
        if not _relevant_for_yes(item, config):
            continue

        if (
            best.yes_score >= config.single_source_yes_threshold
            and item.no_score
            <= best.yes_score + config.event_yes_no_margin
        ):
            return QuestionDecision(
                True,
                best.yes_score,
                best,
                "strong_event_yes",
                event=item.event,
            )

    # 4) Generic ASR/NLI disagreement without an exact-value contradiction is
    # recoverable when one side is very strong. A generic contradiction from a
    # second ASR is deliberately weaker than a deterministic fact mismatch.
    for item in scored:
        best = item.best_yes
        if (
            best.yes_score >= config.generic_disagreement_yes_threshold
            and not best.facts.has_strict_contradiction
            and _relevant_for_yes(item, config)
        ):
            return QuestionDecision(
                True,
                best.yes_score,
                best,
                "strong_generic_disagreement_yes",
                event=item.event,
            )

    # 5) Final high-entailment recovery. This is still relevance-gated, so an
    # off-topic passage cannot become YES only because it is the best of bad
    # retrieval candidates.
    for item in scored:
        best = item.best_yes
        if (
            best.yes_score >= config.strong_entailment_threshold
            and not best.facts.has_strict_contradiction
            and _relevant_for_yes(item, config)
        ):
            return QuestionDecision(
                True,
                best.yes_score,
                best,
                "strong_entailment_recovery",
                event=item.event,
            )

    # No YES survived. Now exact-value contradictions are allowed to dominate
    # the NO decision. This ordering is deliberate: an unrelated contradictory
    # event no longer vetoes a different valid supporting event.
    strict_no = [
        item
        for item in scored
        if item.all_strict_contradiction
    ]
    if strict_no:
        strongest = max(strict_no, key=lambda item: item.no_score)
        return QuestionDecision(
            False,
            strongest.no_score,
            None,
            "strict_fact_contradiction",
            event=strongest.event,
        )

    strongest_no = max(
        scored,
        key=lambda item: item.no_score,
    )
    return QuestionDecision(
        False,
        max(0.5, strongest_no.no_score),
        None,
        "no_supporting_event",
        event=strongest_no.event,
    )
