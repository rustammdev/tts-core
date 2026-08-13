from uztts_events import (
    AudioEvent,
    EventLabel,
    Word,
    consolidate_events,
    merge_transcript,
    strip_event_tags,
)


def word(text: str, start: float, end: float) -> Word:
    return Word(text=text, start=start, end=end)


def event(
    label: EventLabel, start: float, end: float, score: float = 0.9
) -> AudioEvent:
    return AudioEvent(label=label, start=start, end=end, score=score)


WORDS = [
    word("Salom", 0.0, 0.4),
    word("do'stlar,", 0.5, 1.0),
    word("bugun", 2.4, 2.8),
    word("boshlaymiz.", 2.9, 3.5),
]


def test_tag_inserted_between_words() -> None:
    merged = merge_transcript(WORDS, [event(EventLabel.LAUGHTER, 1.2, 2.1)])
    assert merged == "Salom do'stlar, [kulgu] bugun boshlaymiz."


def test_event_before_first_word_leads() -> None:
    merged = merge_transcript(WORDS, [event(EventLabel.MUSIC, 0.0, 0.3)])
    assert merged.startswith("[musiqa] Salom")


def test_event_after_last_word_trails() -> None:
    merged = merge_transcript(WORDS, [event(EventLabel.APPLAUSE, 3.6, 4.5)])
    assert merged.endswith("boshlaymiz. [qarsak]")


def test_no_words_yields_tags_only() -> None:
    events = [
        event(EventLabel.MUSIC, 0.0, 2.0),
        event(EventLabel.LAUGHTER, 2.5, 3.0),
    ]
    assert merge_transcript([], events) == "[musiqa] [kulgu]"


def test_no_events_keeps_transcript() -> None:
    assert merge_transcript(WORDS, []) == "Salom do'stlar, bugun boshlaymiz."


def test_same_tag_not_duplicated_at_one_position() -> None:
    events = [
        event(EventLabel.LAUGHTER, 1.1, 1.5),
        event(EventLabel.LAUGHTER, 2.2, 2.35),
    ]
    merged = merge_transcript(WORDS, events)
    assert merged.count("[kulgu]") == 1


def test_consolidate_merges_nearby_same_label() -> None:
    events = [
        event(EventLabel.LAUGHTER, 1.0, 1.4),
        event(EventLabel.LAUGHTER, 1.6, 2.0),
    ]
    merged = consolidate_events(events)
    assert len(merged) == 1
    assert merged[0].start == 1.0
    assert merged[0].end == 2.0


def test_consolidate_keeps_labels_apart() -> None:
    events = [
        event(EventLabel.LAUGHTER, 1.0, 1.4),
        event(EventLabel.MUSIC, 1.5, 1.9),
    ]
    assert len(consolidate_events(events)) == 2


def test_consolidate_drops_short_events() -> None:
    assert consolidate_events([event(EventLabel.COUGH, 1.0, 1.1)]) == []


def test_strip_event_tags_roundtrip() -> None:
    events = [
        event(EventLabel.MUSIC, 0.0, 0.3),
        event(EventLabel.LAUGHTER, 1.2, 2.1),
    ]
    merged = merge_transcript(WORDS, events)
    assert strip_event_tags(merged) == "Salom do'stlar, bugun boshlaymiz."


def test_deterministic_ordering() -> None:
    events = [
        event(EventLabel.MUSIC, 1.2, 2.0),
        event(EventLabel.LAUGHTER, 1.2, 2.0),
    ]
    first = merge_transcript(WORDS, events)
    second = merge_transcript(WORDS, list(reversed(events)))
    assert first == second
