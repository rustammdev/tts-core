from __future__ import annotations

import re

_SENTENCE_START_RE = re.compile(r"(^\s*|[.?!]\s+)([a-z])")


def capitalize_sentences(text: str) -> str:
    return _SENTENCE_START_RE.sub(
        lambda match: match.group(1) + match.group(2).upper(), text
    )
