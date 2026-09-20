from __future__ import annotations

from typing import Sequence

from .config import E1Config
from .schemas import NLIResult


class NLIBackend:
    """Batched 3-way NLI using DeBERTa-v3 base.

    Model config labels are read dynamically rather than assumed, although the
    selected model publishes entailment=0, neutral=1, contradiction=2.
    """

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

        label2id = {str(k).lower(): int(v) for k, v in self.model.config.label2id.items()}
        self.entailment_id = label2id.get("entailment")
        self.neutral_id = label2id.get("neutral")
        self.contradiction_id = label2id.get("contradiction")
        if None in (self.entailment_id, self.neutral_id, self.contradiction_id):
            raise RuntimeError(f"Unexpected NLI labels: {self.model.config.label2id}")

    def hypothesis(self, question: str) -> str:
        return self.config.nli_hypothesis_template.format(question=question.strip())

    def predict(self, premises: Sequence[str], questions: Sequence[str]) -> list[NLIResult]:
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
            tokens = {k: v.to(self.model.device) for k, v in tokens.items()}
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
