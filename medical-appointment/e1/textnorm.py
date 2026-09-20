from __future__ import annotations

import re
import unicodedata

TOKEN_RE = re.compile(r"\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?|[a-z]+(?:'[a-z]+)?|%")

SMALL = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19,
}
TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
SCALES = {"hundred": 100, "thousand": 1000, "million": 1_000_000}
NUMBER_WORDS = set(SMALL) | set(TENS) | set(SCALES) | {"and", "point"}
DIGIT_WORDS = {k: v for k, v in SMALL.items() if 0 <= v <= 9}

UNIT_MAP = {
    "milligram": "mg", "milligrams": "mg", "mg": "mg",
    "microgram": "mcg", "micrograms": "mcg", "mcg": "mcg", "ug": "mcg",
    "gram": "g", "grams": "g", "g": "g",
    "milliliter": "ml", "milliliters": "ml", "millilitre": "ml",
    "millilitres": "ml", "ml": "ml",
    "unit": "unit", "units": "unit", "iu": "unit",
    "day": "day", "days": "day", "daily": "daily",
    "week": "week", "weeks": "week", "weekly": "weekly",
    "month": "month", "months": "month", "monthly": "monthly",
    "year": "year", "years": "year", "yearly": "yearly",
    "hour": "hour", "hours": "hour", "hourly": "hourly",
    "minute": "minute", "minutes": "minute",
    "mmhg": "mmhg", "mmol": "mmol", "percent": "%", "percentage": "%",
}

QUESTION_STOPWORDS = {
    "a", "an", "the", "patient", "doctor", "was", "were", "is", "are",
    "did", "does", "do", "has", "have", "had", "should", "would", "could",
    "can", "will", "about", "at", "by", "for", "from", "in", "into", "of",
    "on", "to", "with", "and", "or", "that", "this", "there", "it", "they",
    "he", "she", "their", "his", "her", "be", "been", "being", "yes", "no",
}


def basic_tokens(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).lower().replace("’", "'")
    text = text.replace("µg", " mcg ").replace("μg", " mcg ")
    return TOKEN_RE.findall(text)


def _parse_integer_words(tokens: list[str]) -> int | None:
    if not tokens:
        return None
    total = 0
    current = 0
    seen = False
    for token in tokens:
        if token == "and":
            continue
        if token in SMALL:
            current += SMALL[token]
            seen = True
        elif token in TENS:
            current += TENS[token]
            seen = True
        elif token == "hundred":
            current = max(1, current) * 100
            seen = True
        elif token in {"thousand", "million"}:
            scale = SCALES[token]
            total += max(1, current) * scale
            current = 0
            seen = True
        else:
            return None
    return total + current if seen else None


def _parse_number_words(tokens: list[str]) -> float | int | None:
    if "point" not in tokens:
        return _parse_integer_words(tokens)
    point = tokens.index("point")
    left = _parse_integer_words(tokens[:point])
    if left is None:
        left = 0
    right_tokens = tokens[point + 1 :]
    if not right_tokens or any(t not in DIGIT_WORDS for t in right_tokens):
        return None
    digits = "".join(str(DIGIT_WORDS[t]) for t in right_tokens)
    return float(f"{left}.{digits}")


def collapse_number_words(tokens: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(tokens):
        if tokens[i] not in NUMBER_WORDS or tokens[i] == "and":
            out.append(tokens[i])
            i += 1
            continue
        j = i
        seq: list[str] = []
        while j < len(tokens) and tokens[j] in NUMBER_WORDS:
            seq.append(tokens[j])
            j += 1
        value = _parse_number_words(seq)
        if value is None:
            out.append(tokens[i])
            i += 1
            continue
        if isinstance(value, float) and value.is_integer():
            out.append(str(int(value)))
        else:
            out.append(str(value))
        i = j
    return out


def normalize_tokens(text: str) -> list[str]:
    tokens = collapse_number_words(basic_tokens(text))
    return [UNIT_MAP.get(token, token) for token in tokens]


def normalize_text(text: str) -> str:
    return " ".join(normalize_tokens(text))


def topic_tokens(text: str) -> list[str]:
    tokens = normalize_tokens(text)
    result: list[str] = []
    for token in tokens:
        if token in QUESTION_STOPWORDS:
            continue
        if token in set(UNIT_MAP.values()):
            continue
        if token == "%" or any(ch.isdigit() for ch in token):
            continue
        if len(token) <= 1:
            continue
        result.append(token)
    return result


def is_number_word(token: str) -> bool:
    return token.lower() in NUMBER_WORDS or token.lower() in {"once", "twice", "thrice"}
