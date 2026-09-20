from __future__ import annotations

import logging
from typing import Any

from asr_backends import MedASRBackend, ParakeetV3Backend, TranscriptResult

from .config import E1Config
from .consensus import assess_candidate, decide
from .evidence import select_evidence
from .facts import compare_facts, extract_facts
from .nli import NLIBackend
from .pairing import pair_candidates
from .retrieval import retrieve
from .schemas import (
    CandidateAssessment,
    CandidateWindow,
    E1Prediction,
    NLIResult,
    NormalizedWord,
    QuestionTrace,
    TranscriptView,
)
from .textnorm import normalize_text, topic_tokens

logger = logging.getLogger(__name__)


class DualASRE1Pipeline:
    """Dual-ASR E1: MedASR + Parakeet + retrieval + NLI + fact guard.

    Models are injected for tests, but default construction loads all three
    production models once and keeps them resident on the GPU.
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
        self.medasr = medasr if medasr is not None else MedASRBackend(device=device)
        self.parakeet = parakeet if parakeet is not None else ParakeetV3Backend(device=device)
        self.nli = nli if nli is not None else NLIBackend(self.config, device=device)

    @classmethod
    def from_config_file(cls, path: str | None, device: str = "cuda") -> "DualASRE1Pipeline":
        return cls(E1Config.from_json(path), device=device)

    def _view(self, source: str, result: TranscriptResult) -> TranscriptView:
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
        return TranscriptView(source=source, text=result.text, words=words)  # type: ignore[arg-type]

    def _transcribe(self, audio: Any, sample_rate: int) -> dict[str, TranscriptView]:
        transcripts: dict[str, TranscriptView] = {}

        try:
            med = self.medasr.transcribe(audio, sample_rate)
            transcripts["medasr"] = self._view("medasr", med)
        except Exception:
            logger.exception("MedASR failed; continuing with Parakeet")

        try:
            par = self.parakeet.transcribe(audio, sample_rate)
            transcripts["parakeet_v3"] = self._view("parakeet_v3", par)
        except Exception:
            logger.exception("Parakeet failed; continuing with MedASR")

        if not transcripts:
            raise RuntimeError("Both ASR backends failed")
        return transcripts

    def _fallback_nli(self, question: str, candidate: CandidateWindow) -> NLIResult:
        """Deterministic degradation if the NLI model fails at runtime."""
        comparison = compare_facts(extract_facts(question), candidate.facts)
        if comparison.has_strict_contradiction:
            return NLIResult(0.03, 0.07, 0.90)
        topic = set(topic_tokens(question))
        doc = set(topic_tokens(candidate.text))
        overlap = len(topic & doc) / max(1, len(topic))
        if comparison.has_exact_match and overlap >= 0.25:
            return NLIResult(0.72, 0.23, 0.05)
        if overlap >= 0.45:
            return NLIResult(0.56, 0.39, 0.05)
        return NLIResult(0.12, 0.76, 0.12)

    def predict_audio(
        self,
        audio: Any,
        questions: list[str],
        sample_rate: int = 16000,
        return_traces: bool = False,
    ) -> list[E1Prediction] | tuple[list[E1Prediction], list[QuestionTrace]]:
        transcripts = self._transcribe(audio, sample_rate)

        question_candidates: list[dict[str, list[CandidateWindow]]] = []
        all_candidates: list[CandidateWindow] = []
        all_questions_for_nli: list[str] = []

        for question in questions:
            by_source: dict[str, list[CandidateWindow]] = {}
            for source, transcript in transcripts.items():
                candidates = retrieve(question, transcript, self.config)
                by_source[source] = candidates
                all_candidates.extend(candidates)
                all_questions_for_nli.extend([question] * len(candidates))

            # Pairing is computed now because it is an explicit diagnostic and
            # consensus later uses the same temporal criteria.
            pair_candidates(
                by_source.get("medasr", []),
                by_source.get("parakeet_v3", []),
                self.config,
            )
            question_candidates.append(by_source)

        try:
            nli_results = self.nli.predict(
                [c.text for c in all_candidates],
                all_questions_for_nli,
            )
            if len(nli_results) != len(all_candidates):
                raise RuntimeError("NLI returned an unexpected number of results")
        except Exception:
            logger.exception("NLI batch failed; using deterministic fallback")
            nli_results = [
                self._fallback_nli(q, c)
                for q, c in zip(all_questions_for_nli, all_candidates)
            ]

        cursor = 0
        predictions: list[E1Prediction] = []
        traces: list[QuestionTrace] = []

        for question, by_source in zip(questions, question_candidates):
            count = sum(len(v) for v in by_source.values())
            candidate_slice = all_candidates[cursor : cursor + count]
            nli_slice = nli_results[cursor : cursor + count]
            cursor += count

            assessments: list[CandidateAssessment] = [
                assess_candidate(question, candidate, nli, self.config)
                for candidate, nli in zip(candidate_slice, nli_slice)
            ]
            decision = decide(question, assessments, self.config)

            evidence = None
            if decision.answer and decision.candidate is not None:
                source = decision.candidate.candidate.source
                transcript = transcripts[source]
                evidence = select_evidence(
                    question,
                    decision.candidate,
                    transcript,
                    self.config,
                )
                prediction = E1Prediction(
                    answer=True,
                    evidence_start=evidence.start,
                    evidence_end=evidence.end,
                    confidence=decision.confidence,
                    reason=decision.reason,
                    source=evidence.source,
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
                        decision=decision,
                        evidence=evidence,
                    )
                )

        return (predictions, traces) if return_traces else predictions
