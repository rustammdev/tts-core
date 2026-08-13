from __future__ import annotations

import re

OKINA = "ʻ"
TUTUQ = "ʼ"

_APOSTROPHES = "'`´’ʼ‘ʽ‛′"
_OKINA_RE = re.compile(f"([ogOG])[{_APOSTROPHES}{OKINA}]")
_TUTUQ_RE = re.compile(f"[{_APOSTROPHES}]")


def normalize_apostrophes(text: str) -> str:
    marked = _OKINA_RE.sub("\\1\x00", text)
    return _TUTUQ_RE.sub(TUTUQ, marked).replace("\x00", OKINA)
