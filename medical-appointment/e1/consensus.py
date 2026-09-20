from __future__ import annotations

from .config import E1Config
from .facts import compare_facts, extract_facts
from .pairing import candidates_align
from .schemas import (
    CandidateAssessment,
    CandidateWindow,
    NLIResult,
    QuestionDecision,
    SourceVote,
)


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

    if comparison.has_strict_contradiction:
        contradiction = max(contradiction, config.guard_contradiction_probability)
        entail = min(entail, 1.0 - config.guard_contradiction_probability)
    elif comparison.has_exact_match:
        entail = min(1.0, entail + config.fact_match_entailment_bonus)

    # Retrieval acts as evidence quality rather than a hard gate. A top passage
    # gets full weight, while weaker candidates can still win through strong NLI.
    quality = 0.65 + 0.35 * candidate.retrieval_score
    yes_score = entail * quality
    no_score = contradiction * quality

    if comparison.has_strict_contradiction:
        no_score = max(no_score, config.guard_contradiction_probability)

    return CandidateAssessment(
        candidate=candidate,
        nli=NLIResult(entail, nli.neutral, contradiction),
        facts=comparison,
        yes_score=min(1.0, yes_score),
        no_score=min(1.0, no_score),
    )


def source_vote(
    source: str,
    assessments: list[CandidateAssessment],
    config: E1Config,
) -> SourceVote | None:
    source_items = [a for a in assessments if a.candidate.source == source]
    if not source_items:
        return None

    yes_candidate = max(source_items, key=lambda a: a.yes_score)
    no_candidate = max(source_items, key=lambda a: a.no_score)
    yes_score = yes_candidate.yes_score
    no_score = no_candidate.no_score

    if (
        yes_score >= config.source_yes_threshold
        and yes_score >= no_score + config.source_vote_margin
    ):
        label = "yes"
    elif (
        no_score >= config.source_no_threshold
        and no_score >= yes_score + config.source_vote_margin
    ):
        label = "no"
    else:
        label = "neutral"

    return SourceVote(
        source=source,  # type: ignore[arg-type]
        label=label,
        yes_score=yes_score,
        no_score=no_score,
        yes_candidate=yes_candidate,
        no_candidate=no_candidate,
    )


def _choose_yes_evidence(
    med: SourceVote | None,
    par: SourceVote | None,
    config: E1Config,
) -> CandidateAssessment | None:
    med_c = med.yes_candidate if med else None
    par_c = par.yes_candidate if par else None

    if med_c is None:
        return par_c
    if par_c is None:
        return med_c

    # MedASR had the slightly better timestamp ceiling in the measured bake-off.
    # Prefer it when its support score is close to Parakeet; otherwise preserve
    # the stronger lexical/NLI evidence.
    if med_c.yes_score >= par_c.yes_score - config.medasr_evidence_preference_tolerance:
        return med_c
    return par_c


def decide(
    question: str,
    assessments: list[CandidateAssessment],
    config: E1Config,
) -> QuestionDecision:
    med = source_vote("medasr", assessments, config)
    par = source_vote("parakeet_v3", assessments, config)

    # Catastrophic degradation to one ASR is supported deliberately.
    if med is None and par is None:
        return QuestionDecision(False, 0.0, None, "no_asr_candidates")

    if med is None or par is None:
        vote = med or par
        assert vote is not None
        if vote.label == "yes" and vote.yes_score >= config.single_source_yes_threshold:
            return QuestionDecision(True, vote.yes_score, vote.yes_candidate, "single_source_yes", med, par)
        return QuestionDecision(False, max(vote.no_score, 1.0 - vote.yes_score), None, "single_source_no_or_uncertain", med, par)

    # Easy consensus.
    if med.label == "yes" and par.label == "yes":
        aligned = candidates_align(
            med.yes_candidate.candidate if med.yes_candidate else None,
            par.yes_candidate.candidate if par.yes_candidate else None,
            config,
        )
        if aligned or min(med.yes_score, par.yes_score) >= config.strong_entailment_threshold:
            chosen = _choose_yes_evidence(med, par, config)
            confidence = 0.5 * (med.yes_score + par.yes_score)
            return QuestionDecision(True, confidence, chosen, "dual_yes_consensus", med, par)

    if med.label == "no" and par.label == "no":
        return QuestionDecision(False, 0.5 * (med.no_score + par.no_score), None, "dual_no_consensus", med, par)

    # One side says yes, the other is neutral. Allow a high-confidence YES to
    # preserve recall, provided the other source does not contain a strong
    # deterministic contradiction.
    yes_vote = med if med.label == "yes" else par if par.label == "yes" else None
    other_vote = par if yes_vote is med else med if yes_vote is par else None

    if yes_vote is not None and other_vote is not None and other_vote.label == "neutral":
        other_guard_no = (
            other_vote.no_candidate is not None
            and other_vote.no_candidate.facts.has_strict_contradiction
            and other_vote.no_score >= config.source_no_threshold
        )
        if yes_vote.yes_score >= config.single_source_yes_threshold and not other_guard_no:
            return QuestionDecision(True, yes_vote.yes_score, yes_vote.yes_candidate, "yes_plus_neutral", med, par)

    # Hardest E1 case: ASRs disagree. If the YES side has an exact question-fact
    # match and the other side's NO is caused by a conflicting ASR value, retain
    # the strongly supported YES. This exploits disagreement rather than merging
    # transcripts. Otherwise remain conservative and answer NO.
    if {med.label, par.label} == {"yes", "no"}:
        yes_vote = med if med.label == "yes" else par
        no_vote = par if med.label == "yes" else med
        yes_candidate = yes_vote.yes_candidate
        no_candidate = no_vote.no_candidate

        if (
            yes_candidate is not None
            and yes_candidate.facts.has_exact_match
            and yes_vote.yes_score >= config.yes_vs_no_exact_fact_threshold
            and no_candidate is not None
            and no_candidate.facts.has_strict_contradiction
        ):
            return QuestionDecision(True, yes_vote.yes_score, yes_candidate, "asr_fact_disagreement_yes", med, par)

        return QuestionDecision(False, max(no_vote.no_score, 1.0 - yes_vote.yes_score), None, "asr_disagreement_no", med, par)

    # A strong YES may not have crossed the vote margin because another passage
    # from the same source looked contradictory. Require both strong entailment
    # and exact fact support before recovering it here.
    all_yes = [a for a in assessments if a.yes_score >= config.strong_entailment_threshold]
    exact_yes = [a for a in all_yes if a.facts.has_exact_match]
    if exact_yes:
        chosen = max(exact_yes, key=lambda a: a.yes_score)
        return QuestionDecision(True, chosen.yes_score, chosen, "strong_exact_fact_yes", med, par)

    return QuestionDecision(False, max(med.no_score, par.no_score, 0.5), None, "uncertain_defaults_no", med, par)
