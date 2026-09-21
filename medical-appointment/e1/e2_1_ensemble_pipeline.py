from __future__ import annotations

import logging
from typing import Any

from .boundary_calibration import BoundaryCalibrator
from .config import E1Config
from .e2_1b_config import E2_1BConfig
from .e2_1b_pipeline import DualASRE2_1BPipeline
from .e2_1_ensemble_config import E2_1EnsembleConfig
from .evidence_ranker import EvidenceLinearRanker
from .schemas import EvidenceProposal, EvidenceSpan, NLIResult, QuestionDecision, TranscriptView

logger = logging.getLogger(__name__)


def temporal_iou(a: EvidenceSpan, b: EvidenceSpan) -> float:
    inter = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    union = max(a.end, b.end) - min(a.start, b.start)
    return inter / union if union > 0 else 0.0


def should_use_llm(
    llm_span: EvidenceSpan,
    llm_nli: NLIResult,
    ranker_span: EvidenceSpan,
    ranker_nli: NLIResult,
    config: E2_1EnsembleConfig,
) -> bool:
    if llm_nli.entailment < config.minimum_llm_entailment:
        return False
    if temporal_iou(llm_span, ranker_span) >= config.agreement_tiou_threshold:
        return True
    return (
        llm_nli.entailment
        >= ranker_nli.entailment + config.llm_entailment_advantage
    )


class DualASRE2_1EnsemblePipeline(DualASRE2_1BPipeline):
    """Direct word-index LLM + learned proposal ranker ensemble.

    The ranker supplies a conservative structural proposal. The direct LLM span
    is used when it agrees temporally with that proposal or has materially
    stronger semantic entailment. The final timestamp span is calibrated.
    Classification remains frozen to E1.
    """

    def __init__(
        self,
        config: E1Config | None = None,
        e2_config: E2_1BConfig | None = None,
        ensemble_config: E2_1EnsembleConfig | None = None,
        device: str = "cuda",
        medasr: Any | None = None,
        parakeet: Any | None = None,
        nli: Any | None = None,
        word_selector: Any | None = None,
        ranker: EvidenceLinearRanker | None = None,
    ) -> None:
        super().__init__(
            config=config,
            e2_config=e2_config,
            device=device,
            medasr=medasr,
            parakeet=parakeet,
            nli=nli,
            word_selector=word_selector,
        )
        self.ensemble_config = ensemble_config or E2_1EnsembleConfig()
        self.ranker = ranker or EvidenceLinearRanker.from_json(
            self.ensemble_config.ranker_model_path
        )
        self.calibrator = BoundaryCalibrator.from_json(
            self.ensemble_config.calibration_path
        )

    @classmethod
    def from_config_files(
        cls,
        e1_path: str | None,
        word_path: str | None,
        ensemble_path: str | None,
        device: str = "cuda",
    ) -> "DualASRE2_1EnsemblePipeline":
        return cls(
            config=E1Config.from_json(e1_path),
            e2_config=E2_1BConfig.from_json(word_path),
            ensemble_config=E2_1EnsembleConfig.from_json(ensemble_path),
            device=device,
        )

    def _raw_span(self, span: EvidenceSpan, transcripts: dict[str, TranscriptView]):
        transcript = transcripts[span.source]
        return EvidenceSpan(
            span.source,
            span.start_word,
            span.end_word,
            transcript.words[span.start_word].start,
            transcript.words[span.end_word].end,
        )

    def _proposal_raw_span(self, proposal, transcripts):
        transcript = transcripts[proposal.source]
        return EvidenceSpan(
            proposal.source,
            proposal.start_word,
            proposal.end_word,
            transcript.words[proposal.start_word].start,
            transcript.words[proposal.end_word].end,
        )

    def _nli_for_span(self, question, span, transcripts):
        transcript = transcripts[span.source]
        text = " ".join(
            w.text for w in transcript.words[span.start_word : span.end_word + 1]
        )
        try:
            result = self.nli.predict([text], [question])
            return result[0] if result else NLIResult(0.0, 1.0, 0.0)
        except Exception:
            logger.exception("Ensemble direct-span NLI failed")
            return NLIResult(0.0, 1.0, 0.0)

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
        ranker_choice = self.ranker.select_best(
            proposals,
            nlies,
            transcripts,
            min_entailment=self.ensemble_config.ranker_min_nli_entailment,
            require_no_strict_contradiction=(
                self.ensemble_config.require_no_strict_fact_contradiction
            ),
        )

        ranker_span = None
        ranker_nli = None
        if ranker_choice is not None:
            proposal, ranker_nli, _ = ranker_choice
            ranker_span = self._proposal_raw_span(proposal, transcripts)

        llm_span = self._raw_span(selected, transcripts) if selected is not None else None

        if llm_span is None and ranker_span is None:
            return baseline
        if llm_span is None:
            return self.calibrator.apply(ranker_span)
        if ranker_span is None:
            return self.calibrator.apply(llm_span)

        llm_nli = self._nli_for_span(question, llm_span, transcripts)
        if should_use_llm(
            llm_span,
            llm_nli,
            ranker_span,
            ranker_nli,
            self.ensemble_config,
        ):
            return self.calibrator.apply(llm_span)
        return self.calibrator.apply(ranker_span)
