from __future__ import annotations

import logging
from typing import Any

from .boundary_calibration import BoundaryCalibrator
from .config import E1Config
from .e2_1_ranker_config import E2_1RankerConfig
from .evidence import fallback_evidence, select_refined_evidence
from .evidence_ranker import EvidenceLinearRanker
from .pipeline import DualASRE1Pipeline
from .schemas import EvidenceProposal, EvidenceSpan, NLIResult, QuestionDecision, TranscriptView

logger = logging.getLogger(__name__)


class DualASRE2_1RankerPipeline(DualASRE1Pipeline):
    """Frozen E1 classification + learned proposal ranking."""

    def __init__(
        self,
        config: E1Config | None = None,
        ranker_config: E2_1RankerConfig | None = None,
        device: str = "cuda",
        medasr: Any | None = None,
        parakeet: Any | None = None,
        nli: Any | None = None,
        ranker: EvidenceLinearRanker | None = None,
    ) -> None:
        super().__init__(config, device, medasr, parakeet, nli)
        self.ranker_config = ranker_config or E2_1RankerConfig()
        self.ranker = ranker or EvidenceLinearRanker.from_json(
            self.ranker_config.model_path
        )
        self.calibrator = BoundaryCalibrator.from_json(
            self.ranker_config.calibration_path
        )

    @classmethod
    def from_config_files(
        cls,
        e1_path: str | None,
        ranker_path: str | None,
        device: str = "cuda",
    ) -> "DualASRE2_1RankerPipeline":
        return cls(
            config=E1Config.from_json(e1_path),
            ranker_config=E2_1RankerConfig.from_json(ranker_path),
            device=device,
        )

    def _proposal_nli(self, proposals, questions):
        try:
            return self.nli.predict([p.text for p in proposals], questions)
        except Exception:
            logger.exception("Ranker evidence NLI failed")
            return [
                NLIResult(
                    max(0.05, min(0.95, 0.35 + 0.35*p.topic_coverage + 0.25*p.exact_fact_fraction)),
                    0.55,
                    0.95 if p.has_strict_contradiction else 0.05,
                )
                for p in proposals
            ]

    def _baseline(self, decision, proposals, nlies, transcripts):
        if not decision.answer or decision.candidate is None:
            return None
        if proposals:
            try:
                return select_refined_evidence(proposals, nlies, transcripts, self.config)
            except Exception:
                pass
        source = decision.candidate.candidate.source
        return fallback_evidence(decision.candidate, transcripts[source], self.config)

    def _raw_proposal_span(self, proposal, transcripts):
        transcript = transcripts[proposal.source]
        return EvidenceSpan(
            proposal.source,
            proposal.start_word,
            proposal.end_word,
            transcript.words[proposal.start_word].start,
            transcript.words[proposal.end_word].end,
        )

    def _refine_evidence_batch(self, questions, decisions, transcripts):
        proposals, proposal_questions, ranges = self._build_evidence_work(
            questions, decisions, transcripts
        )
        nlies = self._proposal_nli(proposals, proposal_questions) if proposals else []
        result = []

        for decision, (start, end) in zip(decisions, ranges):
            local_p = proposals[start:end]
            local_n = nlies[start:end]
            baseline = self._baseline(decision, local_p, local_n, transcripts)
            if not decision.answer or decision.candidate is None:
                result.append(None)
                continue

            selected = self.ranker.select_best(
                local_p,
                local_n,
                transcripts,
                min_entailment=self.ranker_config.min_nli_entailment,
                require_no_strict_contradiction=(
                    self.ranker_config.require_no_strict_fact_contradiction
                ),
            )
            if selected is None:
                result.append(baseline)
                continue

            proposal, _, _ = selected
            span = self._raw_proposal_span(proposal, transcripts)
            result.append(self.calibrator.apply(span))

        return result
