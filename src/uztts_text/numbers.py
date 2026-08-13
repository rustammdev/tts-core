from __future__ import annotations

import re

ONES = [
    "nol",
    "bir",
    "ikki",
    "uch",
    "toʻrt",
    "besh",
    "olti",
    "yetti",
    "sakkiz",
    "toʻqqiz",
]
TENS = [
    "",
    "oʻn",
    "yigirma",
    "oʻttiz",
    "qirq",
    "ellik",
    "oltmish",
    "yetmish",
    "sakson",
    "toʻqson",
]
SCALES = ["", "ming", "million", "milliard", "trillion", "kvadrillion"]

_NUMBER_RE = re.compile(r"(\d{1,3}(?:[ ,]\d{3})+(?!\d)|\d+)(\s*%)?")


def _triple_to_words(value: int) -> list[str]:
    words: list[str] = []
    hundreds, remainder = divmod(value, 100)
    if hundreds:
        if hundreds > 1:
            words.append(ONES[hundreds])
        words.append("yuz")
    tens, ones = divmod(remainder, 10)
    if tens:
        words.append(TENS[tens])
    if ones:
        words.append(ONES[ones])
    return words


def int_to_words(value: int) -> str:
    if value < 0:
        return "minus " + int_to_words(-value)
    if value == 0:
        return ONES[0]
    groups: list[int] = []
    while value:
        value, group = divmod(value, 1000)
        groups.append(group)
    if len(groups) > len(SCALES):
        raise ValueError("number too large")
    words: list[str] = []
    for index in range(len(groups) - 1, -1, -1):
        group = groups[index]
        if not group:
            continue
        if group == 1 and index == 1:
            words.append(SCALES[1])
            continue
        words.extend(_triple_to_words(group))
        if index:
            words.append(SCALES[index])
    return " ".join(words)


def expand_numbers(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        digits = match.group(1).replace(" ", "").replace(",", "")
        words = int_to_words(int(digits))
        return f"{words} foiz" if match.group(2) else words

    return _NUMBER_RE.sub(replace, text)
