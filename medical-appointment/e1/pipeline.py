from __future__ import annotations

import logging
from typing import Any

from asr_backends import (
    MedASRBackend,
    ParakeetV3Backend,
    TranscriptResult,
)

from .config import E1Config
from .consensus import assess_candidate, decide
from .evidence import (
    build_evidence_proposals,
    fallback_evidence,
    select_refined_evidence,
)
from .facts import compare_facts, extract_facts
from .nli import NLIBackend
from .pairing import pair_assessments
from .retrieval import retrieve
from .schemas import (
    CandidateAssessment,
    CandidateWindow,
    E1Prediction,
    EvidenceEvent,
    EvidenceProposal,
    NLIResult,
    NormalizedWord,
    QuestionDecision,
    QuestionTrace,
    TranscriptView,
)
from .textnorm import normalize_text, topic_tokens

logger = logging.getLogger(__name__)


class DualASRE1Pipeline:
    """Dual-ASR E1 v2.

    MedASR and Parakeet remain independent. Retrieval candidates are paired
    into temporal spoken events, and consensus is performed per event rather
    than by a global vote over each complete ASR transcript.
    """

    def __init__(
        self,
        config: E1Config | None = None,
        device: str = "cuda",
        medasr: Any | None = None,
        parakeet: Any | None = None,
        nli: Any | None = None,
    ) -> None:
        self.config = config or E1Config()
        self.device = device
        self.medasr = (
            medasr
            if medasr is not None
            else MedASRBackend(device=device)
        )
        self.parakeet = (
            parakeet
            if parakeet is not None
            else ParakeetV3Backend(device=device)
        )
        self.nli = (
            nli
            if nli is not None
            else NLIBackend(self.config, device=device)
        )

    @classmethod
    def from_config_file(
        cls,
        path: str | None,
        device: str = "cuda",
    ) -> "DualASRE1Pipeline":
        return cls(
            E1Config.from_json(path),
            device=device,
        )

    def _view(
        self,
        source: str,
        result: TranscriptResult,
    ) -> TranscriptView:
        words = [
            NormalizedWord(
                index=i,
                text=w.text,
                normalized=normalize_text(w.text),
                start=float(w.start),
                end=float(w.end),
                confidence=w.confidence,
            )
            for i, w in enumerate(result.words)
        ]

        return TranscriptView(
            source=source,
            text=result.text,
            words=words,
        )  # type: ignore[arg-type]

    def _transcribe(
        self,
        audio: Any,
        sample_rate: int,
    ) -> dict[str, TranscriptView]:
        transcripts: dict[str, TranscriptView] = {}

        try:
            med = self.medasr.transcribe(
                audio,
                sample_rate,
            )
            transcripts["medasr"] = self._view(
                "medasr",
                med,
            )
        except Exception:
            logger.exception(
                "MedASR failed; continuing with Parakeet"
            )

        try:
            par = self.parakeet.transcribe(
                audio,
                sample_rate,
            )
            transcripts["parakeet_v3"] = self._view(
                "parakeet_v3",
                par,
            )
        except Exception:
            logger.exception(
                "Parakeet failed; continuing with MedASR"
            )

        if not transcripts:
            raise RuntimeError("Both ASR backends failed")

        return transcripts

    def _fallback_nli(
        self,
        question: str,
        candidate: CandidateWindow,
    ) -> NLIResult:
        """Deterministic degradation if the NLI model fails."""

        comparison = compare_facts(
            extract_facts(question),
            candidate.facts,
        )

        if comparison.has_strict_contradiction:
            return NLIResult(0.03, 0.07, 0.90)

        topic = set(topic_tokens(question))
        doc = set(topic_tokens(candidate.text))
        overlap = len(topic & doc) / max(1, len(topic))

        if comparison.has_exact_match and overlap >= 0.20:
            return NLIResult(0.70, 0.25, 0.05)
        if overlap >= 0.40:
            return NLIResult(0.56, 0.39, 0.05)

        return NLIResult(0.12, 0.76, 0.12)

    def _initial_assessments(
        self,
        questions: list[str],
        transcripts: dict[str, TranscriptView],
    ) -> tuple[
        list[dict[str, list[CandidateWindow]]],
        list[list[CandidateAssessment]],
        list[list[EvidenceEvent]],
    ]:
        question_candidates: list[
            dict[str, list[CandidateWindow]]
        ] = []
        flat_candidates: list[CandidateWindow] = []
        flat_questions: list[str] = []
        question_ranges: list[tuple[int, int]] = []

        for question in questions:
            start = len(flat_candidates)
            by_source: dict[str, list[CandidateWindow]] = {}

            for source, transcript in transcripts.items():
                candidates = retrieve(
                    question,
                    transcript,
                    self.config,
                )
                by_source[source] = candidates
                flat_candidates.extend(candidates)
                flat_questions.extend(
                    [question] * len(candidates)
                )

            question_candidates.append(by_source)
            question_ranges.append(
                (start, len(flat_candidates))
            )

        try:
            nli_results = self.nli.predict(
                [candidate.text for candidate in flat_candidates],
                flat_questions,
            )
            if len(nli_results) != len(flat_candidates):
                raise RuntimeError(
                    "NLI returned an unexpected number of results"
                )
        except Exception:
            logger.exception(
                "Initial NLI batch failed; using deterministic fallback"
            )
            nli_results = [
                self._fallback_nli(question, candidate)
                for question, candidate in zip(
                    flat_questions,
                    flat_candidates,
                )
            ]

        assessments_by_question: list[
            list[CandidateAssessment]
        ] = []
        events_by_question: list[list[EvidenceEvent]] = []

        for question, by_source, (start, end) in zip(
            questions,
            question_candidates,
            question_ranges,
        ):
            candidates = flat_candidates[start:end]
            nlies = nli_results[start:end]

            assessments = [
                assess_candidate(
                    question,
                    candidate,
                    nli,
                    self.config,
                )
                for candidate, nli in zip(
                    candidates,
                    nlies,
                )
            ]
            assessments_by_question.append(assessments)

            med = [
                item
                for item in assessments
                if item.candidate.source == "medasr"
            ]
            par = [
                item
                for item in assessments
                if item.candidate.source == "parakeet_v3"
            ]

            events_by_question.append(
                pair_assessments(
                    med,
                    par,
                    self.config,
                )
            )

        return (
            question_candidates,
            assessments_by_question,
            events_by_question,
        )

    def _build_evidence_work(
        self,
        questions: list[str],
        decisions: list[QuestionDecision],
        transcripts: dict[str, TranscriptView],
    ) -> tuple[
        list[EvidenceProposal],
        list[str],
        list[tuple[int, int]],
    ]:
        flat: list[EvidenceProposal] = []
        flat_questions: list[str] = []
        ranges: list[tuple[int, int]] = []

        for question, decision in zip(
            questions,
            decisions,
        ):
            start = len(flat)

            if decision.answer and decision.candidate is not None:
                # Refine evidence from both ASRs when they represent the same
                # chosen spoken event. This lets Parakeet win on lexical support
                # while MedASR still wins close timestamp ties.
                members: list[CandidateAssessment]
                if decision.event is not None:
                    members = list(decision.event.members)
                else:
                    members = [decision.candidate]

                seen: set[tuple[str, int, int]] = set()

                for assessment in members:
                    source = assessment.candidate.source
                    transcript = transcripts.get(source)
                    if transcript is None:
                        continue

                    for proposal in build_evidence_proposals(
                        question,
                        assessment,
                        transcript,
                        self.config,
                    ):
                        key = (
                            proposal.source,
                            proposal.start_word,
                            proposal.end_word,
                        )
                        if key in seen:
                            continue
                        seen.add(key)
                        flat.append(proposal)
                        flat_questions.append(question)

            ranges.append((start, len(flat)))

        return flat, flat_questions, ranges

    def _refine_evidence_batch(
        self,
        questions: list[str],
        decisions: list[QuestionDecision],
        transcripts: dict[str, TranscriptView],
    ) -> list[Any | None]:
        proposals, proposal_questions, ranges = (
            self._build_evidence_work(
                questions,
                decisions,
                transcripts,
            )
        )

        if not proposals:
            return [None] * len(questions)

        try:
            proposal_nli = self.nli.predict(
                [proposal.text for proposal in proposals],
                proposal_questions,
            )
            if len(proposal_nli) != len(proposals):
                raise RuntimeError(
                    "Evidence NLI returned an unexpected number of results"
                )
        except Exception:
            logger.exception(
                "Evidence refinement NLI failed; using heuristic pseudo-NLI"
            )
            proposal_nli = [
                NLIResult(
                    entailment=max(
                        0.05,
                        min(
                            0.95,
                            0.35
                            + 0.35 * proposal.topic_coverage
                            + 0.25 * proposal.exact_fact_fraction
                        ),
                    ),
                    neutral=0.55,
                    contradiction=(
                        0.95
                        if proposal.has_strict_contradiction
                        else 0.05
                    ),
                )
                for proposal in proposals
            ]

        evidence_by_question: list[Any | None] = []

        for decision, (start, end) in zip(
            decisions,
            ranges,
        ):
            if not decision.answer or decision.candidate is None:
                evidence_by_question.append(None)
                continue

            local_proposals = proposals[start:end]
            local_nli = proposal_nli[start:end]

            if local_proposals:
                try:
                    evidence_by_question.append(
                        select_refined_evidence(
                            local_proposals,
                            local_nli,
                            transcripts,
                            self.config,
                        )
                    )
                    continue
                except Exception:
                    logger.exception(
                        "Compact evidence selection failed; using fallback"
                    )

            source = decision.candidate.candidate.source
            evidence_by_question.append(
                fallback_evidence(
                    decision.candidate,
                    transcripts[source],
                    self.config,
                )
            )

        return evidence_by_question

    def predict_audio(
        self,
        audio: Any,
        questions: list[str],
        sample_rate: int = 16000,
        return_traces: bool = False,
    ) -> (
        list[E1Prediction]
        | tuple[list[E1Prediction], list[QuestionTrace]]
    ):
        transcripts = self._transcribe(
            audio,
            sample_rate,
        )

        (
            question_candidates,
            assessments_by_question,
            events_by_question,
        ) = self._initial_assessments(
            questions,
            transcripts,
        )

        decisions = [
            decide(
                question,
                assessments,
                events,
                self.config,
            )
            for question, assessments, events in zip(
                questions,
                assessments_by_question,
                events_by_question,
            )
        ]

        evidence_by_question = self._refine_evidence_batch(
            questions,
            decisions,
            transcripts,
        )

        predictions: list[E1Prediction] = []
        traces: list[QuestionTrace] = []

        for (
            question,
            by_source,
            assessments,
            events,
            decision,
            evidence,
        ) in zip(
            questions,
            question_candidates,
            assessments_by_question,
            events_by_question,
            decisions,
            evidence_by_question,
        ):
            if decision.answer and evidence is not None:
                prediction = E1Prediction(
                    answer=True,
                    evidence_start=evidence.start,
                    evidence_end=evidence.end,
                    confidence=decision.confidence,
                    reason=decision.reason,
                    source=evidence.source,
                )
            elif decision.answer and decision.candidate is not None:
                # This should be rare; preserve a valid YES even if refinement
                # unexpectedly produced no proposal.
                source = decision.candidate.candidate.source
                fallback = fallback_evidence(
                    decision.candidate,
                    transcripts[source],
                    self.config,
                )
                evidence = fallback
                prediction = E1Prediction(
                    answer=True,
                    evidence_start=fallback.start,
                    evidence_end=fallback.end,
                    confidence=decision.confidence,
                    reason=decision.reason,
                    source=fallback.source,
                )
            else:
                prediction = E1Prediction(
                    answer=False,
                    evidence_start=None,
                    evidence_end=None,
                    confidence=decision.confidence,
                    reason=decision.reason,
                    source=None,
                )

            predictions.append(prediction)

            if return_traces:
                traces.append(
                    QuestionTrace(
                        question=question,
                        candidates=by_source,
                        assessments=assessments,
                        events=events,
                        decision=decision,
                        evidence=evidence,
                    )
                )

        return (
            (predictions, traces)
            if return_traces
            else predictions
        )
