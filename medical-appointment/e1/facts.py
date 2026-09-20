from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Iterable

from .schemas import Fact, FactComparison
from .textnorm import normalize_text

STRICT_FACT_KINDS = {
    "dose_mg",
    "dose_unit",
    "volume_ml",
    "duration_days",
    "frequency_per_day",
    "blood_pressure",
    "temperature_c",
    "lab_mmol",
    "lab_mg_dl",
    "percentage",
    "side",
}


def _f(value: str) -> float:
    return float(value)


def _round(value: float) -> float:
    return round(float(value), 6)


def extract_facts(text: str) -> list[Fact]:
    n = normalize_text(text)
    facts: list[Fact] = []

    # Blood pressure: 135/88 or 135 over 88.
    for m in re.finditer(r"\b(\d{2,3})\s*(?:/|over)\s*(\d{2,3})\b", n):
        facts.append(Fact("blood_pressure", (int(m.group(1)), int(m.group(2))), "mmhg", m.group(0)))

    # Mass doses canonicalized to mg.
    for m in re.finditer(r"\b(\d+(?:\.\d+)?)\s*(mg|mcg|g)\b", n):
        value = _f(m.group(1))
        unit = m.group(2)
        mg = value if unit == "mg" else value / 1000.0 if unit == "mcg" else value * 1000.0
        facts.append(Fact("dose_mg", _round(mg), "mg", m.group(0)))

    for m in re.finditer(r"\b(\d+(?:\.\d+)?)\s*unit\b", n):
        facts.append(Fact("dose_unit", _round(_f(m.group(1))), "unit", m.group(0)))

    for m in re.finditer(r"\b(\d+(?:\.\d+)?)\s*ml\b", n):
        facts.append(Fact("volume_ml", _round(_f(m.group(1))), "ml", m.group(0)))

    # Lab-like values where the unit carries semantics.
    for m in re.finditer(r"\b(\d+(?:\.\d+)?)\s*mmol(?:\s*(?:/\s*)?l)?\b", n):
        facts.append(Fact("lab_mmol", _round(_f(m.group(1))), "mmol/l", m.group(0)))
    for m in re.finditer(r"\b(\d+(?:\.\d+)?)\s*mg\s*(?:/\s*)?dl\b", n):
        facts.append(Fact("lab_mg_dl", _round(_f(m.group(1))), "mg/dl", m.group(0)))

    # Duration canonicalized to days. Avoid age phrases such as "30 years old".
    duration_scale = {"day": 1.0, "week": 7.0, "month": 30.0, "year": 365.0, "hour": 1.0 / 24.0, "minute": 1.0 / 1440.0}
    for m in re.finditer(r"\b(\d+(?:\.\d+)?)\s*(day|week|month|year|hour|minute)\b", n):
        tail = n[m.end() : m.end() + 8]
        if m.group(2) == "year" and re.match(r"\s*old\b", tail):
            continue
        facts.append(Fact("duration_days", _round(_f(m.group(1)) * duration_scale[m.group(2)]), "day", m.group(0)))

    # Frequencies.
    lexical_freq = {
        "once daily": 1.0,
        "twice daily": 2.0,
        "thrice daily": 3.0,
        "once a day": 1.0,
        "twice a day": 2.0,
        "once weekly": 1.0 / 7.0,
        "twice weekly": 2.0 / 7.0,
    }
    for phrase, value in lexical_freq.items():
        if phrase in n:
            facts.append(Fact("frequency_per_day", _round(value), "1/day", phrase))

    for m in re.finditer(r"\b(\d+(?:\.\d+)?)\s*times?\s*(?:a|per)?\s*day\b", n):
        facts.append(Fact("frequency_per_day", _round(_f(m.group(1))), "1/day", m.group(0)))
    for m in re.finditer(r"\b(\d+(?:\.\d+)?)\s*times?\s*daily\b", n):
        facts.append(Fact("frequency_per_day", _round(_f(m.group(1))), "1/day", m.group(0)))
    for m in re.finditer(r"\bevery\s+(\d+(?:\.\d+)?)\s*hour\b", n):
        hours = _f(m.group(1))
        if hours > 0:
            facts.append(Fact("frequency_per_day", _round(24.0 / hours), "1/day", m.group(0)))

    # Temperatures.
    for m in re.finditer(r"\b(\d{2,3}(?:\.\d+)?)\s*(?:degrees?\s*)?(c|f)\b", n):
        value = _f(m.group(1))
        if m.group(2) == "f":
            value = (value - 32.0) * 5.0 / 9.0
        facts.append(Fact("temperature_c", _round(value), "c", m.group(0)))

    for m in re.finditer(r"\b(\d+(?:\.\d+)?)\s*%", n):
        facts.append(Fact("percentage", _round(_f(m.group(1))), "%", m.group(0)))

    sides = {token for token in ("left", "right", "bilateral") if re.search(rf"\b{token}\b", n)}
    for side in sorted(sides):
        facts.append(Fact("side", side, None, side))

    return _dedupe(facts)


def _dedupe(facts: Iterable[Fact]) -> list[Fact]:
    seen: set[tuple[str, str, str | None]] = set()
    out: list[Fact] = []
    for fact in facts:
        key = (fact.kind, repr(fact.value), fact.unit)
        if key not in seen:
            seen.add(key)
            out.append(fact)
    return out


def _equal(a: Fact, b: Fact) -> bool:
    if a.kind != b.kind:
        return False
    if isinstance(a.value, float) or isinstance(b.value, float):
        try:
            return math.isclose(float(a.value), float(b.value), rel_tol=1e-4, abs_tol=1e-4)
        except (TypeError, ValueError):
            return a.value == b.value
    return a.value == b.value


def compare_facts(question_facts: list[Fact], candidate_facts: list[Fact]) -> FactComparison:
    matched: list[str] = []
    mismatched: list[str] = []

    for kind in sorted(STRICT_FACT_KINDS):
        q = [f for f in question_facts if f.kind == kind]
        c = [f for f in candidate_facts if f.kind == kind]
        if not q or not c:
            continue

        # A kind is considered matched when every question-side value has at
        # least one corresponding candidate value. This avoids falsely marking
        # a window contradictory just because it contains an additional dose.
        all_q_match = all(any(_equal(qf, cf) for cf in c) for qf in q)
        if all_q_match:
            matched.append(kind)
        else:
            mismatched.append(kind)

    return FactComparison(tuple(matched), tuple(mismatched))


def fact_type_overlap(question_facts: list[Fact], candidate_facts: list[Fact]) -> float:
    q = {f.kind for f in question_facts if f.kind in STRICT_FACT_KINDS}
    if not q:
        return 0.0
    c = {f.kind for f in candidate_facts if f.kind in STRICT_FACT_KINDS}
    return len(q & c) / len(q)
