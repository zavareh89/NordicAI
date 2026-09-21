from __future__ import annotations

import logging
from typing import Any

from .config import E1Config
from .e2_1b_config import E2_1BConfig
from .evidence import fallback_evidence, select_refined_evidence
from .llm_word_evidence import (
    LLMWordEvidenceSelector,
    WordSelectionTask,
    build_word_selection_task,
    validate_word_selection,
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


class DualASRE2_1BPipeline(DualASRE1Pipeline):
    """E2.1-B: frozen E1 classification + direct LLM word-index evidence.

    This is an alternative to E2.1 proposal-ID selection. It still cannot alter
    classification. The LLM chooses only a source and contiguous word indices
    inside the E1-selected temporal event.
    """

    def __init__(
        self,
        config: E1Config | None = None,
        e2_config: E2_1BConfig | None = None,
        device: str = "cuda",
        medasr: Any | None = None,
        parakeet: Any | None = None,
        nli: Any | None = None,
        word_selector: Any | None = None,
    ) -> None:
        super().__init__(
            config=config,
            device=device,
            medasr=medasr,
            parakeet=parakeet,
            nli=nli,
        )
        self.e2_config = e2_config or E2_1BConfig()
        self.word_selector = word_selector or LLMWordEvidenceSelector(
            self.e2_config,
            device=device,
        )

    @classmethod
    def from_config_files(
        cls,
        e1_path: str | None,
        e2_path: str | None,
        device: str = "cuda",
    ) -> "DualASRE2_1BPipeline":
        return cls(
            config=E1Config.from_json(e1_path),
            e2_config=E2_1BConfig.from_json(e2_path),
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
                raise RuntimeError("unexpected evidence NLI result count")
            return results
        except Exception:
            logger.exception("E2.1-B evidence NLI failed; using heuristic fallback")
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
                    contradiction=0.95 if proposal.has_strict_contradiction else 0.05,
                )
                for proposal in proposals
            ]

    def _baseline_e1_evidence(
        self,
        decision: QuestionDecision,
        proposals: list[EvidenceProposal],
        nlies: list[NLIResult],
        transcripts: dict[str, TranscriptView],
    ) -> EvidenceSpan | None:
        if not decision.answer or decision.candidate is None:
            return None
        if proposals:
            try:
                return select_refined_evidence(
                    proposals,
                    nlies,
                    transcripts,
                    self.config,
                )
            except Exception:
                logger.exception("E2.1-B baseline evidence selection failed")
        source = decision.candidate.candidate.source
        return fallback_evidence(decision.candidate, transcripts[source], self.config)

    def _postprocess_selected_span(
        self,
        question: str,
        decision: QuestionDecision,
        transcripts: dict[str, TranscriptView],
        proposals: list[EvidenceProposal],
        nlies: list[NLIResult],
        baseline: EvidenceSpan | None,
        selected: EvidenceSpan | None,
    ) -> EvidenceSpan | None:
        return selected or baseline

    def _refine_evidence_batch(
        self,
        questions: list[str],
        decisions: list[QuestionDecision],
        transcripts: dict[str, TranscriptView],
    ) -> list[EvidenceSpan | None]:
        proposals, proposal_questions, ranges = self._build_evidence_work(
            questions,
            decisions,
            transcripts,
        )
        proposal_nli = self._proposal_nli(proposals, proposal_questions) if proposals else []

        baselines: list[EvidenceSpan | None] = []
        task_rows: list[tuple[int, WordSelectionTask, list[EvidenceProposal], list[NLIResult]]] = []
        next_task_id = 0

        for question_index, (question, decision, (start, end)) in enumerate(
            zip(questions, decisions, ranges)
        ):
            local_proposals = proposals[start:end]
            local_nli = proposal_nli[start:end]
            baseline = self._baseline_e1_evidence(
                decision,
                local_proposals,
                local_nli,
                transcripts,
            )
            baselines.append(baseline)

            if not decision.answer or decision.candidate is None:
                continue

            task = build_word_selection_task(
                task_id=next_task_id,
                question=question,
                decision=decision,
                transcripts=transcripts,
                proposals=local_proposals,
                nlies=local_nli,
                baseline=baseline,
                e1_config=self.config,
                config=self.e2_config,
            )
            if task is None:
                continue

            task_rows.append((question_index, task, local_proposals, local_nli))
            next_task_id += 1

        if not task_rows:
            return baselines

        tasks = [row[1] for row in task_rows]
        try:
            selections = self.word_selector.select(tasks, transcripts)
        except Exception:
            logger.exception("E2.1-B word selector failed; preserving E1 evidence")
            return baselines

        final = list(baselines)

        for question_index, task, local_proposals, local_nli in task_rows:
            selection = selections.get(task.task_id)
            selected_span = None
            if selection is not None:
                try:
                    selected_span = validate_word_selection(
                        task,
                        selection,
                        transcripts,
                        self.e2_config,
                    )
                except Exception:
                    logger.exception("Invalid E2.1-B word selection")
                    selected_span = None

            final[question_index] = self._postprocess_selected_span(
                question=questions[question_index],
                decision=decisions[question_index],
                transcripts=transcripts,
                proposals=local_proposals,
                nlies=local_nli,
                baseline=baselines[question_index],
                selected=selected_span,
            )

        return final
