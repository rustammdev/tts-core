from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from uztts_events.schema import AudioEvent, EventLabel, Word

TAGS: dict[EventLabel, str] = {
    EventLabel.LAUGHTER: "[kulgu]",
    EventLabel.MUSIC: "[musiqa]",
    EventLabel.APPLAUSE: "[qarsak]",
    EventLabel.COUGH: "[yoʻtal]",
}

_TAG_RE = re.compile("|".join(re.escape(tag) for tag in TAGS.values()))


def consolidate_events(
    events: Iterable[AudioEvent],
    *,
    merge_gap: float = 0.5,
    min_duration: float = 0.2,
) -> list[AudioEvent]:
    by_label: dict[EventLabel, list[AudioEvent]] = {}
    for event in sorted(events, key=lambda item: (item.start, item.end)):
        bucket = by_label.setdefault(event.label, [])
        if bucket and event.start - bucket[-1].end <= merge_gap:
            last = bucket[-1]
            bucket[-1] = AudioEvent(
                label=last.label,
                start=last.start,
                end=max(last.end, event.end),
                score=max(last.score, event.score),
            )
        else:
            bucket.append(event)
    kept = (
        event
        for bucket in by_label.values()
        for event in bucket
        if event.end - event.start >= min_duration
    )
    return sorted(kept, key=lambda item: (item.start, item.end, item.label))


def merge_transcript(
    words: Sequence[Word],
    events: Iterable[AudioEvent],
    *,
    merge_gap: float = 0.5,
    min_duration: float = 0.2,
) -> str:
    ordered = consolidate_events(events, merge_gap=merge_gap, min_duration=min_duration)
    insertions: list[list[str]] = [[] for _ in range(len(words) + 1)]
    for event in ordered:
        index = sum(1 for word in words if word.end <= event.start)
        tag = TAGS[event.label]
        if tag not in insertions[index]:
            insertions[index].append(tag)
    pieces: list[str] = []
    for index, word in enumerate(words):
        pieces.extend(insertions[index])
        pieces.append(word.text)
    pieces.extend(insertions[len(words)])
    return " ".join(pieces)


def strip_event_tags(text: str) -> str:
    return re.sub(r"\s+", " ", _TAG_RE.sub(" ", text)).strip()
