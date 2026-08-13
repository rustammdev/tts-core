import pytest
from pydantic import ValidationError

from uztts_events import AudioEvent, EventLabel, Word


def test_event_rejects_reversed_interval() -> None:
    with pytest.raises(ValidationError):
        AudioEvent(label=EventLabel.LAUGHTER, start=2.0, end=1.0, score=0.5)


def test_event_rejects_zero_length() -> None:
    with pytest.raises(ValidationError):
        AudioEvent(label=EventLabel.MUSIC, start=1.0, end=1.0, score=0.5)


def test_event_rejects_score_above_one() -> None:
    with pytest.raises(ValidationError):
        AudioEvent(label=EventLabel.MUSIC, start=0.0, end=1.0, score=1.5)


def test_word_allows_zero_length() -> None:
    assert Word(text="a", start=1.0, end=1.0).end == 1.0


def test_word_rejects_empty_text() -> None:
    with pytest.raises(ValidationError):
        Word(text="", start=0.0, end=1.0)
