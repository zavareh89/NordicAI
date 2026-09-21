from __future__ import annotations

from dataclasses import replace
from typing import Any

from .boundary_calibration import BoundaryCalibrator
from .config import E1Config
from .e2_1b_config import E2_1BConfig
from .e2_1b_pipeline import DualASRE2_1BPipeline
from .schemas import EvidenceProposal, EvidenceSpan, NLIResult, QuestionDecision, TranscriptView


class DualASRE2_1CPipeline(DualASRE2_1BPipeline):
    """E2.1-C: direct word-index selection plus CV-tuned boundary shifts."""

    def __init__(
        self,
        config: E1Config | None = None,
        e2_config: E2_1BConfig | None = None,
        calibrator: BoundaryCalibrator | None = None,
        device: str = "cuda",
        medasr: Any | None = None,
        parakeet: Any | None = None,
        nli: Any | None = None,
        word_selector: Any | None = None,
    ) -> None:
        base_e2_config = e2_config or E2_1BConfig()
        # Calibration is fitted on raw ASR word boundaries, so do not apply
        # E2.1-B's fixed padding before the learned shifts.
        raw_e2_config = replace(
            base_e2_config,
            padding_before_s=0.0,
            padding_after_s=0.0,
        )
        super().__init__(
            config=config,
            e2_config=raw_e2_config,
            device=device,
            medasr=medasr,
            parakeet=parakeet,
            nli=nli,
            word_selector=word_selector,
        )
        self.calibrator = calibrator or BoundaryCalibrator()

    @classmethod
    def from_config_files(
        cls,
        e1_path: str | None,
        e2_path: str | None,
        calibration_path: str | None,
        device: str = "cuda",
    ) -> "DualASRE2_1CPipeline":
        return cls(
            config=E1Config.from_json(e1_path),
            e2_config=E2_1BConfig.from_json(e2_path),
            calibrator=BoundaryCalibrator.from_json(calibration_path),
            device=device,
        )

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
        if selected is None:
            return baseline
        return self.calibrator.apply(selected)
