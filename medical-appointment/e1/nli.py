from __future__ import annotations

import re
from typing import Sequence

from .config import E1Config
from .schemas import NLIResult


_SPACE_RE = re.compile(r"\s+")


def question_to_hypothesis(question: str) -> str:
    """Convert common yes/no questions into declarative NLI hypotheses.

    MNLI-style models are trained on declarative premise/hypothesis pairs. E1
    v1 wrapped the raw question in "The correct answer is yes...", which is a
    distribution mismatch and made the model unnecessarily conservative.

    The rules below intentionally keep auxiliary verbs when that avoids risky
    grammatical transformations ("Did the patient report..." -> "The patient
    did report..."). The result is still a declarative factual claim.
    """

    q = _SPACE_RE.sub(" ", question.strip()).rstrip(" ?")
    if not q:
        return q

    rules = [
        (r"^Was the patient (.+)$", r"The patient was \1."),
        (r"^Were the patient(?:'s)? (.+)$", r"The patient's \1 were present."),
        (r"^Is the patient (.+)$", r"The patient is \1."),
        (r"^Are the patient(?:'s)? (.+)$", r"The patient's \1 are present."),
        (r"^Has the patient (.+)$", r"The patient has \1."),
        (r"^Had the patient (.+)$", r"The patient had \1."),
        (r"^Does the patient (.+)$", r"The patient does \1."),
        (r"^Did the patient (.+)$", r"The patient did \1."),
        (r"^Do the patient(?:s)? (.+)$", r"The patients do \1."),
        (r"^Was there (.+)$", r"There was \1."),
        (r"^Were there (.+)$", r"There were \1."),
        (r"^Is there (.+)$", r"There is \1."),
        (r"^Are there (.+)$", r"There are \1."),
        (r"^Has there been (.+)$", r"There has been \1."),
        (r"^Have there been (.+)$", r"There have been \1."),
        (r"^Was (.+?) (detected|found|observed|reported|documented|noted)$",
         r"\1 was \2."),
        (r"^Were (.+?) (detected|found|observed|reported|documented|noted)$",
         r"\1 were \2."),
        (r"^Is (.+)$", r"\1 is true."),
        (r"^Are (.+)$", r"\1 are true."),
        (r"^Did (.+)$", r"It is true that \1."),
        (r"^Does (.+)$", r"It is true that \1."),
        (r"^Has (.+)$", r"It is true that \1."),
        (r"^Have (.+)$", r"It is true that \1."),
    ]

    for pattern, replacement in rules:
        converted = re.sub(pattern, replacement, q, count=1, flags=re.IGNORECASE)
        if converted != q:
            # Normalize leading capitalization without modifying medical names.
            return converted[0].upper() + converted[1:]

    # Safe fallback: convert the raw yes/no question into an explicit claim
    # without asking the NLI model to reason about "the answer being yes".
    return f"It is true that {q}."


class NLIBackend:
    """Batched 3-way NLI using DeBERTa-v3 base."""

    def __init__(self, config: E1Config, device: str = "cuda") -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.config = config
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(config.nli_model_id)
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.nli_model_id,
            dtype=dtype,
        )
        self.model.to(device)
        self.model.eval()

        label2id = {
            str(k).lower(): int(v)
            for k, v in self.model.config.label2id.items()
        }
        self.entailment_id = label2id.get("entailment")
        self.neutral_id = label2id.get("neutral")
        self.contradiction_id = label2id.get("contradiction")
        if None in (
            self.entailment_id,
            self.neutral_id,
            self.contradiction_id,
        ):
            raise RuntimeError(
                f"Unexpected NLI labels: {self.model.config.label2id}"
            )

    def hypothesis(self, question: str) -> str:
        return question_to_hypothesis(question)

    def predict(
        self,
        premises: Sequence[str],
        questions: Sequence[str],
    ) -> list[NLIResult]:
        import torch

        if len(premises) != len(questions):
            raise ValueError("premises and questions must have the same length")
        if not premises:
            return []

        hypotheses = [self.hypothesis(q) for q in questions]
        out: list[NLIResult] = []
        bs = self.config.nli_batch_size

        for start in range(0, len(premises), bs):
            p = list(premises[start : start + bs])
            h = hypotheses[start : start + bs]
            tokens = self.tokenizer(
                p,
                h,
                padding=True,
                truncation=True,
                max_length=self.config.nli_max_length,
                return_tensors="pt",
            )
            tokens = {
                k: v.to(self.model.device)
                for k, v in tokens.items()
            }

            with torch.inference_mode():
                logits = self.model(**tokens).logits.float()
                probs = torch.softmax(logits, dim=-1).cpu()

            for row in probs:
                out.append(
                    NLIResult(
                        entailment=float(row[self.entailment_id]),
                        neutral=float(row[self.neutral_id]),
                        contradiction=float(row[self.contradiction_id]),
                    )
                )

        return out
