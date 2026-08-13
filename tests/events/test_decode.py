import numpy as np

from uztts_events.decode import frames_to_events, median_smooth
from uztts_events.labels import resolve_class_ids
from uztts_events.schema import EventLabel


def test_median_smooth_removes_single_spike() -> None:
    probs = np.array([0.0, 0.0, 0.9, 0.0, 0.0])
    assert median_smooth(probs, 3).max() == 0.0


def test_median_smooth_window_one_is_identity() -> None:
    probs = np.array([0.1, 0.9, 0.1])
    assert np.array_equal(median_smooth(probs, 1), probs)


def test_frames_to_events_extracts_run() -> None:
    probs = np.array([0.0] * 10 + [0.8] * 10 + [0.0] * 10)
    events = frames_to_events(
        EventLabel.LAUGHTER,
        probs,
        threshold=0.5,
        frame_seconds=0.04,
        max_seconds=1.2,
        median_window=1,
    )
    assert len(events) == 1
    assert events[0].start == 10 * 0.04
    assert events[0].end == 20 * 0.04
    assert events[0].score == 0.8


def test_frames_to_events_clips_to_audio_length() -> None:
    probs = np.array([0.9] * 250)
    events = frames_to_events(
        EventLabel.MUSIC,
        probs,
        threshold=0.5,
        frame_seconds=0.04,
        max_seconds=3.0,
        median_window=1,
    )
    assert len(events) == 1
    assert events[0].end == 3.0


def test_frames_to_events_run_reaching_last_frame_closes() -> None:
    probs = np.array([0.0] * 5 + [0.7] * 5)
    events = frames_to_events(
        EventLabel.COUGH,
        probs,
        threshold=0.5,
        frame_seconds=0.04,
        max_seconds=0.4,
        median_window=1,
    )
    assert len(events) == 1
    assert events[0].end == 0.4


def test_frames_to_events_nothing_above_threshold() -> None:
    probs = np.array([0.1] * 20)
    assert (
        frames_to_events(
            EventLabel.MUSIC,
            probs,
            threshold=0.5,
            frame_seconds=0.04,
            max_seconds=0.8,
            median_window=1,
        )
        == []
    )


def test_resolve_class_ids_finds_all_labels() -> None:
    names = [
        "Applause",
        "Baby laughter",
        "Belly laugh",
        "Chuckle, chortle",
        "Clapping",
        "Cough",
        "Giggle",
        "Laughter",
        "Music",
        "Snicker",
        "Speech",
    ]
    ids = resolve_class_ids(names)
    assert ids[EventLabel.LAUGHTER] == [7, 1, 6, 9, 2, 3]
    assert ids[EventLabel.MUSIC] == [8]
    assert ids[EventLabel.APPLAUSE] == [0, 4]
    assert ids[EventLabel.COUGH] == [5]


def test_resolve_class_ids_fails_fast_on_missing() -> None:
    import pytest

    with pytest.raises(ValueError, match="Music"):
        resolve_class_ids(["Laughter"])
