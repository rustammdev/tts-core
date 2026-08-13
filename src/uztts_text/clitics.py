from __future__ import annotations

JOINED_CLITICS = frozenset({"da", "ku", "chi", "mi", "yu", "ya", "ta"})


def join_clitics(normalized_text: str) -> str:
    joined: list[str] = []
    for token in normalized_text.split():
        if token in JOINED_CLITICS and joined:
            joined[-1] += token
        else:
            joined.append(token)
    return " ".join(joined)
