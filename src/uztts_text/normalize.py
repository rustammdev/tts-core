from __future__ import annotations

import re

from uztts_text.apostrophes import OKINA, TUTUQ, normalize_apostrophes
from uztts_text.numbers import expand_numbers
from uztts_text.translit import cyrillic_to_latin

_PUNCT_RE = re.compile(rf"[^a-z{OKINA}{TUTUQ} ]")
_SPACE_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    converted = expand_numbers(cyrillic_to_latin(normalize_apostrophes(text)))
    stripped = _PUNCT_RE.sub(" ", converted.lower())
    return _SPACE_RE.sub(" ", stripped).strip()
