from __future__ import annotations

from pathlib import Path

import numpy as np

from uztts_events.config import EventsConfig
from uztts_events.localize import SAMPLE_RATE, FrameSedLocalizer
from uztts_events.merge import consolidate_events
from uztts_events.schema import AudioEvent, EventLabel
from uztts_events.screen import CedScreener


def load_audio(path: Path) -> np.ndarray:
    import soundfile

    waveform, sample_rate = soundfile.read(str(path), dtype="float32")
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    if sample_rate != SAMPLE_RATE:
        import torch
        from torchaudio.functional import resample

        resampled = resample(
            torch.from_numpy(waveform)[None, :], sample_rate, SAMPLE_RATE
        )
        waveform = resampled[0].numpy()
    return np.asarray(waveform, dtype=np.float32)


class EventTagger:
    def __init__(self, config: EventsConfig | None = None) -> None:
        self.config = config or EventsConfig()
        self.screener = CedScreener(
            model_name=self.config.ced_model,
            revision=self.config.ced_revision,
            device=self.config.device,
        )
        self.localizer = FrameSedLocalizer(
            checkpoint_url=self.config.sed_checkpoint_url,
            device=self.config.device,
        )

    def screen(self, waveform: np.ndarray) -> dict[EventLabel, float]:
        return self.screener.screen(waveform)

    def tag(self, waveform: np.ndarray) -> list[AudioEvent]:
        scores = self.screener.screen(waveform)
        flagged = {
            label
            for label, score in scores.items()
            if score >= self.config.screen_thresholds[label.value]
        }
        if not flagged:
            return []
        events = self.localizer.localize(
            waveform,
            flagged,
            frame_thresholds=self.config.frame_thresholds,
            median_window=self.config.median_window,
        )
        return consolidate_events(
            events,
            merge_gap=self.config.merge_gap,
            min_duration=self.config.min_duration,
        )
