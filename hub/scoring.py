from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import jiwer

OKINA = "ʻ"
TUTUQ = "ʼ"
_APOSTROPHES = "'`´’ʼʻ‘"
_OKINA_RE = re.compile(f"([ogOG])[{_APOSTROPHES}{OKINA}]")
_TUTUQ_RE = re.compile(f"[{_APOSTROPHES}]")
_PUNCT_RE = re.compile(rf"[^a-z{OKINA}{TUTUQ} ]")
_SPACE_RE = re.compile(r"\s+")

JOINED_CLITICS = frozenset({"da", "ku", "chi", "mi", "yu", "ya", "ta"})


def to_okina(text: str) -> str:
    marked = _OKINA_RE.sub("\\1\x00", text)
    return _TUTUQ_RE.sub(TUTUQ, marked).replace("\x00", OKINA)


def normalize_text(text: str) -> str:
    lowered = to_okina(text).lower()
    stripped = _PUNCT_RE.sub(" ", lowered)
    return _SPACE_RE.sub(" ", stripped).strip()


def join_clitics(normalized_text: str) -> str:
    joined: list[str] = []
    for token in normalized_text.split():
        if token in JOINED_CLITICS and joined:
            joined[-1] += token
        else:
            joined.append(token)
    return " ".join(joined)


def score(references: list[str], hypotheses: list[str]) -> dict[str, float]:
    refs = [normalize_text(text) for text in references]
    hyps = [normalize_text(text) for text in hypotheses]
    return {
        "wer": jiwer.wer(refs, hyps),
        "cer": jiwer.cer(refs, hyps),
        "wer_canonical": jiwer.wer(
            [join_clitics(text) for text in refs], [join_clitics(text) for text in hyps]
        ),
        "cer_canonical": jiwer.cer(
            [join_clitics(text) for text in refs], [join_clitics(text) for text in hyps]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Score a predictions file against the benchmark. "
            "Expects JSONL rows with 'ref' and 'hyp' fields."
        )
    )
    parser.add_argument("predictions", type=Path, help="JSONL with ref/hyp per line")
    args = parser.parse_args()

    rows = [
        json.loads(line) for line in args.predictions.read_text().splitlines() if line
    ]
    result = score([row["ref"] for row in rows], [row["hyp"] for row in rows])
    print(f"clips: {len(rows)}")
    for name, value in result.items():
        print(f"{name}: {value * 100:.2f}%")


if __name__ == "__main__":
    main()
