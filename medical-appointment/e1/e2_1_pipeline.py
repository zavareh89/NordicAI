from __future__ import annotations

import logging
from typing import Any

from .config import E1Config
from .e2_1_config import E2_1Config
from .evidence import (
    fallback_evidence,
    select_refined_evidence,
)
from .llm_evidence import (
    LLMEvidenceReranker,
    RerankTask,
    build_rerank_candidates,
    proposal_to_evidence_span,
)
from .pipeline import DualASRE1Pipeline
from .schemas import (
    EvidenceProposal,
    EvidenceSpan,
    NLIResult,
    QuestionDecision,
    TranscriptView,
)

logger = logging.getLogger(__name__)


def _temporal_iou(
    a_start: float,
    a_end: float,
    b_start: float,
    b_end: float,
) -> float:
    intersection = max(
        0.0,
        min(a_end, b_end) - max(a_start, b_start),
    )
    union = max(a_end, b_end) - min(a_start, b_start)
    return intersection / union if union > 0.0 else 0.0


def _span_center(span: EvidenceSpan) -> float:
    return 0.5 * (float(span.start) + float(span.end))


def _apply_iou_safety_guard(
    baseline_span: EvidenceSpan,
    llm_span: EvidenceSpan,
) -> EvidenceSpan:
    """Cheap evidence safety guard.

    - Zero overlap with the original E1 span -> keep E1.
    - Center moves >3s from E1 -> keep E1.
    - Tiny overlap (<0.15) on the same ASR -> use the union.
    - Otherwise keep the LLM span.
    - Add ±0.35s padding to very short (<3s) final spans.
    """

    overlap = _temporal_iou(
        llm_span.start,
        llm_span.end,
        baseline_span.start,
        baseline_span.end,
    )

    center_gap = abs(
        _span_center(llm_span)
        - _span_center(baseline_span)
    )

    if overlap <= 0.0 or center_gap > 3.0:
        chosen = baseline_span

    elif (
        overlap < 0.15
        and llm_span.source == baseline_span.source
    ):
        chosen = EvidenceSpan(
            source=baseline_span.source,
            start_word=min(
                baseline_span.start_word,
                llm_span.start_word,
            ),
            end_word=max(
                baseline_span.end_word,
                llm_span.end_word,
            ),
            start=min(
                baseline_span.start,
                llm_span.start,
            ),
            end=max(
                baseline_span.end,
                llm_span.end,
            ),
        )

    else:
        chosen = llm_span

    duration = float(chosen.end) - float(chosen.start)

    if duration < 3.0:
        chosen = EvidenceSpan(
            source=chosen.source,
            start_word=chosen.start_word,
            end_word=chosen.end_word,
            start=max(
                0.0,
                float(chosen.start) - 0.35,
            ),
            end=float(chosen.end) + 0.35,
        )

    return chosen


class DualASRE2_1Pipeline(DualASRE1Pipeline):
    """E2.1 = frozen E1 classification + selective local-LLM evidence reranking.

    Critical invariant:
      E2.1 NEVER changes a YES/NO answer produced by E1.

    It only reranks evidence proposals for questions that E1 already classified
    as YES. Any LLM/model/parser failure falls back to the original E1 evidence.
    """

    def __init__(
        self,
        config: E1Config | None = None,
        e2_config: E2_1Config | None = None,
        device: str = "cuda",
        medasr: Any | None = None,
        parakeet: Any | None = None,
        nli: Any | None = None,
        llm_reranker: Any | None = None,
    ) -> None:
        super().__init__(
            config=config,
            device=device,
            medasr=medasr,
            parakeet=parakeet,
            nli=nli,
        )

        self.e2_config = (
            e2_config
            if e2_config is not None
            else E2_1Config()
        )

        self.llm_reranker = (
            llm_reranker
            if llm_reranker is not None
            else LLMEvidenceReranker(
                self.e2_config,
                device=device,
            )
        )

    @classmethod
    def from_config_files(
        cls,
        e1_path: str | None,
        e2_path: str | None,
        device: str = "cuda",
    ) -> "DualASRE2_1Pipeline":
        return cls(
            config=E1Config.from_json(e1_path),
            e2_config=E2_1Config.from_json(e2_path),
            device=device,
        )

    def _proposal_nli(
        self,
        proposals: list[EvidenceProposal],
        proposal_questions: list[str],
    ) -> list[NLIResult]:
        try:
            results = self.nli.predict(
                [proposal.text for proposal in proposals],
                proposal_questions,
            )

            if len(results) != len(proposals):
                raise RuntimeError(
                    "Evidence NLI returned an unexpected number of results"
                )

            return results

        except Exception:
            logger.exception(
                "E2.1 evidence NLI failed; using the same E1 heuristic fallback"
            )

            return [
                NLIResult(
                    entailment=max(
                        0.05,
                        min(
                            0.95,
                            0.35
                            + 0.35 * proposal.topic_coverage
                            + 0.25 * proposal.exact_fact_fraction,
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

    def _baseline_e1_evidence(
        self,
        decision: QuestionDecision,
        local_proposals: list[EvidenceProposal],
        local_nli: list[NLIResult],
        transcripts: dict[str, TranscriptView],
    ) -> EvidenceSpan | None:
        if not decision.answer or decision.candidate is None:
            return None

        if local_proposals:
            try:
                return select_refined_evidence(
                    local_proposals,
                    local_nli,
                    transcripts,
                    self.config,
                )
            except Exception:
                logger.exception(
                    "E2.1 could not compute baseline E1 evidence; "
                    "using deterministic E1 fallback"
                )

        source = decision.candidate.candidate.source

        return fallback_evidence(
            decision.candidate,
            transcripts[source],
            self.config,
        )

    def _refine_evidence_batch(
        self,
        questions: list[str],
        decisions: list[QuestionDecision],
        transcripts: dict[str, TranscriptView],
    ) -> list[EvidenceSpan | None]:
        (
            proposals,
            proposal_questions,
            ranges,
        ) = self._build_evidence_work(
            questions,
            decisions,
            transcripts,
        )

        if not proposals:
            return [
                None
                if not decision.answer
                else fallback_evidence(
                    decision.candidate,
                    transcripts[
                        decision.candidate.candidate.source
                    ],
                    self.config,
                )
                if decision.candidate is not None
                else None
                for decision in decisions
            ]

        proposal_nli = self._proposal_nli(
            proposals,
            proposal_questions,
        )

        baseline_evidence: list[EvidenceSpan | None] = []

        rerank_tasks: list[RerankTask] = []
        task_to_question: dict[int, int] = {}
        task_candidates: dict[int, tuple[Any, ...]] = {}

        next_task_id = 0

        for question_index, (
            question,
            decision,
            (start, end),
        ) in enumerate(
            zip(
                questions,
                decisions,
                ranges,
            )
        ):
            local_proposals = proposals[start:end]
            local_nli = proposal_nli[start:end]

            e1_span = self._baseline_e1_evidence(
                decision,
                local_proposals,
                local_nli,
                transcripts,
            )
            baseline_evidence.append(e1_span)

            if (
                not decision.answer
                or decision.candidate is None
                or not local_proposals
            ):
                continue

            candidates = build_rerank_candidates(
                task_id=next_task_id,
                proposals=local_proposals,
                nli_results=local_nli,
                baseline_span=e1_span,
                e1_config=self.config,
                e2_config=self.e2_config,
            )

            if len(candidates) < 2:
                continue

            rerank_tasks.append(
                RerankTask(
                    task_id=next_task_id,
                    question=question,
                    candidates=candidates,
                )
            )
            task_to_question[next_task_id] = question_index
            task_candidates[next_task_id] = candidates
            next_task_id += 1

        if not rerank_tasks:
            return baseline_evidence

        try:
            selections = self.llm_reranker.rank(
                rerank_tasks
            )
        except Exception:
            logger.exception(
                "E2.1 LLM reranker failed; preserving all E1 evidence spans"
            )
            return baseline_evidence

        final_evidence = list(baseline_evidence)

        for task in rerank_tasks:
            selected_id = selections.get(task.task_id)

            if selected_id is None:
                continue

            selected = next(
                (
                    candidate
                    for candidate in task_candidates[task.task_id]
                    if candidate.candidate_id == selected_id
                ),
                None,
            )

            if selected is None:
                continue

            if (
                self.e2_config.require_no_strict_fact_contradiction
                and selected.proposal.has_strict_contradiction
            ):
                continue

            question_index = task_to_question[task.task_id]
            baseline_span = baseline_evidence[question_index]

            try:
                llm_span = proposal_to_evidence_span(
                    selected.proposal,
                    transcripts,
                    self.e2_config,
                )

                if baseline_span is not None:
                    llm_span = _apply_iou_safety_guard(
                        baseline_span,
                        llm_span,
                    )

                final_evidence[question_index] = llm_span

            except Exception:
                logger.exception(
                    "E2.1 selected span could not be materialized; "
                    "preserving E1 evidence"
                )

        return final_evidence
