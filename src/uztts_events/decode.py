from __future__ import annotations

import numpy as np

from uztts_events.schema import AudioEvent, EventLabel


def median_smooth(probs: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return probs
    half = window // 2
    padded = np.pad(probs, half, mode="edge")
    frames = np.lib.stride_tricks.sliding_window_view(padded, window)
    return np.asarray(np.median(frames, axis=1)[: len(probs)])


def frames_to_events(
    label: EventLabel,
    probs: np.ndarray,
    *,
    threshold: float,
    frame_seconds: float,
    max_seconds: float,
    median_window: int = 9,
) -> list[AudioEvent]:
    smoothed = median_smooth(probs, median_window)
    active = smoothed >= threshold
    events: list[AudioEvent] = []
    run_start: int | None = None
    for index, flag in enumerate([*active.tolist(), False]):
        if flag and run_start is None:
            run_start = index
        elif not flag and run_start is not None:
            start = run_start * frame_seconds
            end = min(index * frame_seconds, max_seconds)
            if end > start:
                score = min(1.0, float(np.max(probs[run_start:index])))
                events.append(
                    AudioEvent(label=label, start=start, end=end, score=score)
                )
            run_start = None
    return events
