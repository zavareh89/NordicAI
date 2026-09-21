from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import re
from typing import Any, Sequence

from .e2_1b_config import E2_1BConfig
from .evidence import proposal_final_score
from .facts import compare_facts, extract_facts
from .schemas import (
    EvidenceProposal,
    EvidenceSpan,
    NLIResult,
    QuestionDecision,
    TranscriptView,
)
from .textnorm import topic_tokens

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AllowedWordRange:
    source: str
    left: int
    right: int  # exclusive


@dataclass(frozen=True)
class WordSelectionTask:
    task_id: int
    question: str
    ranges: tuple[AllowedWordRange, ...]
    baseline: EvidenceSpan | None
    proposal_hints: tuple[EvidenceProposal, ...]


@dataclass(frozen=True)
class WordSelection:
    source: str
    start_word: int
    end_word: int  # inclusive


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _event_ranges(
    decision: QuestionDecision,
    transcripts: dict[str, TranscriptView],
    config: E2_1BConfig,
) -> tuple[AllowedWordRange, ...]:
    members = []
    if decision.event is not None:
        members = list(decision.event.members)
    elif decision.candidate is not None:
        members = [decision.candidate]

    out: list[AllowedWordRange] = []
    seen: set[str] = set()

    for member in members:
        candidate = member.candidate
        source = candidate.source
        if source in seen or source not in transcripts:
            continue
        seen.add(source)

        transcript = transcripts[source]
        left = max(0, candidate.left - config.event_context_words)
        right = min(
            len(transcript.words),
            candidate.right + config.event_context_words,
        )
        if right > left:
            out.append(AllowedWordRange(source, left, right))

    return tuple(out)


def _proposal_hints(
    proposals: Sequence[EvidenceProposal],
    nlies: Sequence[NLIResult],
    e1_config: Any,
    limit: int,
) -> tuple[EvidenceProposal, ...]:
    scored: list[tuple[float, EvidenceProposal]] = []
    for proposal, nli in zip(proposals, nlies):
        if proposal.has_strict_contradiction:
            continue
        scored.append((proposal_final_score(proposal, nli, e1_config), proposal))
    scored.sort(key=lambda item: -item[0])
    return tuple(item[1] for item in scored[: max(0, limit)])


def build_word_selection_task(
    task_id: int,
    question: str,
    decision: QuestionDecision,
    transcripts: dict[str, TranscriptView],
    proposals: Sequence[EvidenceProposal],
    nlies: Sequence[NLIResult],
    baseline: EvidenceSpan | None,
    e1_config: Any,
    config: E2_1BConfig,
) -> WordSelectionTask | None:
    ranges = _event_ranges(decision, transcripts, config)
    if not ranges:
        return None

    return WordSelectionTask(
        task_id=task_id,
        question=question,
        ranges=ranges,
        baseline=baseline,
        proposal_hints=_proposal_hints(
            proposals,
            nlies,
            e1_config,
            config.max_proposal_hints,
        ),
    )


def _render_words(
    transcript: TranscriptView,
    allowed: AllowedWordRange,
) -> str:
    return " ".join(
        f"[{word.index}] {word.text}"
        for word in transcript.words[allowed.left : allowed.right]
    )


def build_batch_prompt(
    tasks: Sequence[WordSelectionTask],
    transcripts: dict[str, TranscriptView],
) -> str:
    parts = [
        "You select evidence boundaries in timestamped medical transcripts.",
        (
            "The YES/NO decision is already fixed to YES. Do not reconsider, "
            "change, or discuss the answer. Your only job is to choose the "
            "best contiguous supporting word range."
        ),
        (
            "Choose the shortest complete spoken clause/utterance that directly "
            "establishes the answer, while keeping essential clinical qualifiers "
            "such as medication, dose, frequency, duration, side, measurement, "
            "result, or negation. Avoid bare values without their clinical action "
            "and avoid unrelated neighboring speech."
        ),
        (
            "You may select only indices shown in the allowed ranges. Never invent "
            "words or timestamps. Return exactly one JSON object and no prose."
        ),
        (
            'Format: {"0":{"source":"medasr","start_word":12,'
            '"end_word":19},"1":{...}}'
        ),
    ]

    for task in tasks:
        parts.append(f"\nTASK {task.task_id}")
        parts.append(f"QUESTION: {task.question}")

        if task.baseline is not None:
            parts.append(
                "CURRENT BASELINE: "
                + json.dumps(
                    {
                        "source": task.baseline.source,
                        "start_word": task.baseline.start_word,
                        "end_word": task.baseline.end_word,
                    }
                )
            )

        if task.proposal_hints:
            parts.append("STRUCTURAL HINTS (not mandatory choices):")
            for proposal in task.proposal_hints:
                parts.append(
                    json.dumps(
                        {
                            "source": proposal.source,
                            "start_word": proposal.start_word,
                            "end_word": proposal.end_word,
                            "kind": proposal.kind,
                            "text": proposal.text,
                        },
                        ensure_ascii=False,
                    )
                )

        for allowed in task.ranges:
            parts.append(
                f"SOURCE {allowed.source} ALLOWED "
                f"[{allowed.left}, {allowed.right - 1}]"
            )
            parts.append(_render_words(transcripts[allowed.source], allowed))

    return "\n".join(parts)


def parse_word_selections(
    text: str,
    tasks: Sequence[WordSelectionTask],
) -> dict[int, WordSelection]:
    allowed_ids = {task.task_id for task in tasks}
    candidates = [text.strip()]
    match = _JSON_OBJECT_RE.search(text)
    if match:
        candidates.append(match.group(0))

    for raw in candidates:
        try:
            payload = json.loads(raw)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue

        result: dict[int, WordSelection] = {}
        for key, value in payload.items():
            try:
                task_id = int(key)
            except Exception:
                continue
            if task_id not in allowed_ids or not isinstance(value, dict):
                continue
            try:
                source = str(value["source"])
                start_word = int(value["start_word"])
                end_word = int(value["end_word"])
            except Exception:
                continue
            result[task_id] = WordSelection(source, start_word, end_word)
        if result:
            return result

    return {}


def validate_word_selection(
    task: WordSelectionTask,
    selection: WordSelection,
    transcripts: dict[str, TranscriptView],
    config: E2_1BConfig,
) -> EvidenceSpan | None:
    allowed = next(
        (item for item in task.ranges if item.source == selection.source),
        None,
    )
    if allowed is None or selection.source not in transcripts:
        return None

    if selection.start_word < allowed.left or selection.end_word >= allowed.right:
        return None
    if selection.end_word < selection.start_word:
        return None

    length = selection.end_word - selection.start_word + 1
    if length < config.min_span_words or length > config.max_span_words:
        return None

    transcript = transcripts[selection.source]
    text = " ".join(
        word.text
        for word in transcript.words[
            selection.start_word : selection.end_word + 1
        ]
    )

    comparison = compare_facts(extract_facts(task.question), extract_facts(text))
    if config.require_no_strict_fact_contradiction and comparison.has_strict_contradiction:
        return None

    q_topic = set(topic_tokens(task.question))
    s_topic = set(topic_tokens(text))
    topic_coverage = len(q_topic & s_topic) / max(1, len(q_topic))

    if config.require_topic_or_exact_fact:
        if not comparison.has_exact_match and topic_coverage < config.min_topic_coverage:
            return None

    start = max(
        0.0,
        transcript.words[selection.start_word].start - config.padding_before_s,
    )
    end = transcript.words[selection.end_word].end + config.padding_after_s

    return EvidenceSpan(
        source=selection.source,
        start_word=selection.start_word,
        end_word=selection.end_word,
        start=start,
        end=end,
    )


class LLMWordEvidenceSelector:
    def __init__(self, config: E2_1BConfig, device: str = "cuda") -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.config = config
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_id)
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            dtype=dtype,
            low_cpu_mem_usage=True,
        )
        self.model.to(device)
        self.model.eval()

    def _generate(
        self,
        tasks: Sequence[WordSelectionTask],
        transcripts: dict[str, TranscriptView],
    ) -> dict[int, WordSelection]:
        import torch

        if not tasks:
            return {}

        prompt = build_batch_prompt(tasks, transcripts)
        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        input_length = int(inputs["input_ids"].shape[-1])

        if input_length > self.config.max_input_tokens:
            if len(tasks) == 1:
                raise RuntimeError(
                    "E2.1-B prompt exceeds max_input_tokens: "
                    f"{input_length} > {self.config.max_input_tokens}"
                )
            middle = len(tasks) // 2
            result: dict[int, WordSelection] = {}
            result.update(self._generate(tasks[:middle], transcripts))
            result.update(self._generate(tasks[middle:], transcripts))
            return result

        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = output[0, input_length:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return parse_word_selections(text, tasks)

    def select(
        self,
        tasks: Sequence[WordSelectionTask],
        transcripts: dict[str, TranscriptView],
    ) -> dict[int, WordSelection]:
        result: dict[int, WordSelection] = {}
        step = max(1, self.config.max_tasks_per_generation)
        for start in range(0, len(tasks), step):
            batch = tasks[start : start + step]
            result.update(self._generate(batch, transcripts))
        return result
