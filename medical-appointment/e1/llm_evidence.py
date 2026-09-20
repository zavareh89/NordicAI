from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import re
from typing import Any, Iterable, Sequence

from .e2_1_config import E2_1Config
from .evidence import proposal_final_score
from .schemas import (
    EvidenceProposal,
    EvidenceSpan,
    NLIResult,
    TranscriptView,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RerankCandidate:
    """A proposal exposed to the LLM.

    `candidate_id` is the only value the model is allowed to return.
    """

    candidate_id: str
    proposal: EvidenceProposal
    nli: NLIResult
    baseline_score: float


@dataclass(frozen=True)
class RerankTask:
    task_id: int
    question: str
    candidates: tuple[RerankCandidate, ...]


def _kind_group(kind: str) -> str:
    if kind.startswith("pause_"):
        return "pause"
    if kind.startswith("clause"):
        return "clause"
    if kind.startswith("retrieval"):
        return "retrieval"
    if kind.startswith("anchor"):
        return "anchor"
    if kind.startswith("fixed_"):
        try:
            width = int(kind.split("_", 1)[1])
        except Exception:
            width = 0
        return "fixed_long" if width >= 16 else "fixed_short"
    if kind.startswith("scan_"):
        return "scan"
    return kind


def _proposal_key(
    proposal: EvidenceProposal,
) -> tuple[str, int, int]:
    return (
        proposal.source,
        proposal.start_word,
        proposal.end_word,
    )


def _find_e1_selected_proposal(
    proposals: Sequence[EvidenceProposal],
    baseline_span: EvidenceSpan | None,
) -> EvidenceProposal | None:
    if baseline_span is None:
        return None

    for proposal in proposals:
        if (
            proposal.source == baseline_span.source
            and proposal.start_word == baseline_span.start_word
            and proposal.end_word == baseline_span.end_word
        ):
            return proposal

    return None


def build_rerank_candidates(
    task_id: int,
    proposals: Sequence[EvidenceProposal],
    nli_results: Sequence[NLIResult],
    baseline_span: EvidenceSpan | None,
    e1_config: Any,
    e2_config: E2_1Config,
) -> tuple[RerankCandidate, ...]:
    """Build a small, diverse, safe proposal set for one YES question.

    The LLM never receives a proposal with a deterministic fact contradiction.
    It also never receives arbitrary spans outside the E1-selected event.
    """

    if len(proposals) != len(nli_results):
        raise ValueError(
            "proposals and nli_results must have the same length"
        )

    scored: list[
        tuple[float, EvidenceProposal, NLIResult]
    ] = []

    for proposal, nli in zip(
        proposals,
        nli_results,
    ):
        if (
            e2_config.require_no_strict_fact_contradiction
            and proposal.has_strict_contradiction
        ):
            continue

        if nli.entailment < e2_config.min_nli_entailment:
            continue

        if proposal.topic_coverage < e2_config.min_topic_coverage:
            continue

        score = proposal_final_score(
            proposal,
            nli,
            e1_config,
        )
        scored.append(
            (score, proposal, nli)
        )

    # If the normal safety thresholds filtered everything, retain the strongest
    # non-contradictory proposal so E2.1 can still fall back cleanly.
    if not scored:
        for proposal, nli in zip(
            proposals,
            nli_results,
        ):
            if (
                e2_config.require_no_strict_fact_contradiction
                and proposal.has_strict_contradiction
            ):
                continue
            scored.append(
                (
                    proposal_final_score(
                        proposal,
                        nli,
                        e1_config,
                    ),
                    proposal,
                    nli,
                )
            )

    if not scored:
        return ()

    scored.sort(
        key=lambda item: (
            -item[0],
            -item[2].entailment,
            item[1].end_word - item[1].start_word,
        )
    )

    best_score = scored[0][0]

    # Prevent the LLM from picking a proposal that E1 considered wildly worse.
    scored = [
        item
        for item in scored
        if (
            best_score - item[0]
            <= e2_config.max_baseline_score_drop
        )
    ] or [scored[0]]

    e1_selected = _find_e1_selected_proposal(
        proposals,
        baseline_span,
    )

    selected: list[
        tuple[float, EvidenceProposal, NLIResult]
    ] = []
    used_spans: set[tuple[str, int, int]] = set()
    used_groups: set[str] = set()

    def add_item(
        item: tuple[float, EvidenceProposal, NLIResult],
    ) -> None:
        key = _proposal_key(item[1])
        if key in used_spans:
            return
        selected.append(item)
        used_spans.add(key)
        used_groups.add(
            _kind_group(item[1].kind)
        )

    if (
        e2_config.always_include_e1_choice
        and e1_selected is not None
    ):
        for item in scored:
            if _proposal_key(item[1]) == _proposal_key(e1_selected):
                add_item(item)
                break
        else:
            # E1's selected proposal can have fallen below the optional E2
            # filters. It is still safe to include unless it has a strict
            # deterministic contradiction.
            if not e1_selected.has_strict_contradiction:
                for proposal, nli in zip(
                    proposals,
                    nli_results,
                ):
                    if proposal is e1_selected:
                        add_item(
                            (
                                proposal_final_score(
                                    proposal,
                                    nli,
                                    e1_config,
                                ),
                                proposal,
                                nli,
                            )
                        )
                        break

    if e2_config.prefer_candidate_family_diversity:
        # First pass: best proposal from each structural family.
        for item in scored:
            group = _kind_group(
                item[1].kind
            )
            if group in used_groups:
                continue
            add_item(item)
            if (
                len(selected)
                >= e2_config.max_candidates_per_question
            ):
                break

    # Second pass: fill remaining slots by E1 evidence score.
    for item in scored:
        if (
            len(selected)
            >= e2_config.max_candidates_per_question
        ):
            break
        add_item(item)

    # Stable candidate IDs are task-local and intentionally contain no timing.
    result: list[RerankCandidate] = []

    for local_index, (
        score,
        proposal,
        nli,
    ) in enumerate(selected):
        result.append(
            RerankCandidate(
                candidate_id=(
                    f"T{task_id}_C{local_index}"
                ),
                proposal=proposal,
                nli=nli,
                baseline_score=float(score),
            )
        )

    return tuple(result)


def _task_prompt(
    task: RerankTask,
) -> str:
    lines = [
        f"TASK {task.task_id}",
        f"QUESTION: {task.question}",
        (
            "The YES/NO answer is already fixed to YES. "
            "Choose the evidence span only."
        ),
        "CANDIDATES:",
    ]

    for candidate in task.candidates:
        proposal = candidate.proposal
        lines.append(
            json.dumps(
                {
                    "id": candidate.candidate_id,
                    "text": proposal.text,
                    "family": _kind_group(
                        proposal.kind
                    ),
                    "words": (
                        proposal.end_word
                        - proposal.start_word
                        + 1
                    ),
                    "nli_entailment": round(
                        candidate.nli.entailment,
                        4,
                    ),
                    "exact_fact_fraction": round(
                        proposal.exact_fact_fraction,
                        4,
                    ),
                },
                ensure_ascii=False,
            )
        )

    return "\n".join(lines)


def build_batch_prompt(
    tasks: Sequence[RerankTask],
) -> str:
    """Build a compact deterministic reranking prompt.

    No timestamps are included. The model can select only a candidate ID.
    """

    task_ids = [
        str(task.task_id)
        for task in tasks
    ]

    return "\n\n".join(
        [
            (
                "You are an evidence-boundary selector for a medical "
                "conversation benchmark."
            ),
            (
                "For every task, the answer is already known to be YES. "
                "DO NOT reconsider or change the YES/NO answer."
            ),
            (
                "Choose the single candidate that is the best complete spoken "
                "evidence for the question. Prefer a complete clause or "
                "utterance that directly states the clinical fact/action. "
                "Include essential qualifiers such as medication, dose, "
                "frequency, duration, side, measurement, or result when they "
                "matter. Avoid a bare value fragment if it omits the clinical "
                "action, and avoid unrelated neighboring speech."
            ),
            (
                "You may choose ONLY one of the candidate IDs shown for each "
                "task. Never invent text, word indices, or timestamps."
            ),
            (
                "Return exactly one JSON object and no explanation. "
                "Keys are task numbers and values are candidate IDs. "
                f"Required keys: {json.dumps(task_ids)}"
            ),
            *[
                _task_prompt(task)
                for task in tasks
            ],
        ]
    )


_JSON_OBJECT_RE = re.compile(
    r"\{.*?\}",
    re.DOTALL,
)


def parse_selection_output(
    text: str,
    tasks: Sequence[RerankTask],
) -> dict[int, str]:
    """Parse model output defensively.

    Invalid/missing task selections are simply omitted; the caller then keeps
    the E1 evidence for those questions.
    """

    allowed = {
        task.task_id: {
            candidate.candidate_id
            for candidate in task.candidates
        }
        for task in tasks
    }

    parsed: dict[int, str] = {}

    # Preferred path: strict JSON.
    candidates = [
        text.strip()
    ]
    match = _JSON_OBJECT_RE.search(text)
    if match:
        candidates.append(
            match.group(0)
        )

    for raw in candidates:
        try:
            data = json.loads(raw)
        except Exception:
            continue

        if not isinstance(data, dict):
            continue

        for key, value in data.items():
            try:
                task_id = int(key)
            except Exception:
                continue

            if (
                task_id in allowed
                and isinstance(value, str)
                and value in allowed[task_id]
            ):
                parsed[task_id] = value

        if parsed:
            return parsed

    # Fallback: recover explicit candidate IDs if the model wrapped the JSON
    # in prose or produced slightly malformed formatting.
    for task in tasks:
        for candidate in task.candidates:
            if candidate.candidate_id in text:
                # Accept only when exactly one allowed candidate for this task
                # appears in the output.
                hits = [
                    item.candidate_id
                    for item in task.candidates
                    if item.candidate_id in text
                ]
                if len(hits) == 1:
                    parsed[
                        task.task_id
                    ] = hits[0]
                break

    return parsed


def proposal_to_evidence_span(
    proposal: EvidenceProposal,
    transcripts: dict[str, TranscriptView],
    config: E2_1Config,
) -> EvidenceSpan:
    transcript = transcripts[
        proposal.source
    ]

    start = max(
        0.0,
        transcript.words[
            proposal.start_word
        ].start
        - config.padding_before_s,
    )

    end = (
        transcript.words[
            proposal.end_word
        ].end
        + config.padding_after_s
    )

    return EvidenceSpan(
        source=proposal.source,
        start_word=proposal.start_word,
        end_word=proposal.end_word,
        start=start,
        end=end,
    )


class LLMEvidenceReranker:
    """Local Qwen evidence reranker.

    This component cannot alter the E1 answer. It only returns candidate IDs
    supplied by the caller.
    """

    def __init__(
        self,
        config: E2_1Config,
        device: str = "cuda",
    ) -> None:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
        )

        self.config = config
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_id
        )

        dtype = (
            torch.float16
            if device.startswith("cuda")
            else torch.float32
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            dtype=dtype,
            low_cpu_mem_usage=True,
        )
        self.model.to(device)
        self.model.eval()

    def _generate(
        self,
        tasks: Sequence[RerankTask],
    ) -> dict[int, str]:
        import torch

        if not tasks:
            return {}

        prompt = build_batch_prompt(
            tasks
        )

        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        input_length = int(
            inputs["input_ids"].shape[-1]
        )

        if (
            input_length
            > self.config.max_input_tokens
        ):
            # Split batches rather than truncating evidence text.
            if len(tasks) == 1:
                raise RuntimeError(
                    "E2.1 prompt exceeds max_input_tokens "
                    f"({input_length} > {self.config.max_input_tokens})"
                )

            midpoint = len(tasks) // 2
            result = {}
            result.update(
                self._generate(
                    tasks[:midpoint]
                )
            )
            result.update(
                self._generate(
                    tasks[midpoint:]
                )
            )
            return result

        inputs = {
            key: value.to(
                self.model.device
            )
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=(
                    self.config.max_new_tokens
                ),
                do_sample=False,
                use_cache=True,
                pad_token_id=(
                    self.tokenizer.eos_token_id
                ),
            )

        generated = outputs[
            0,
            input_length:,
        ]

        text = self.tokenizer.decode(
            generated,
            skip_special_tokens=True,
        )

        return parse_selection_output(
            text,
            tasks,
        )

    def rank(
        self,
        tasks: Sequence[RerankTask],
    ) -> dict[int, str]:
        """Rank all YES questions in as few generation calls as possible."""

        if not tasks:
            return {}

        result: dict[int, str] = {}
        step = max(
            1,
            self.config.max_tasks_per_generation,
        )

        for start in range(
            0,
            len(tasks),
            step,
        ):
            batch = tasks[
                start : start + step
            ]
            result.update(
                self._generate(batch)
            )

        return result
