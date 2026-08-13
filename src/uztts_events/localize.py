from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from uztts_data.paths import data_root
from uztts_events.config import SED_CHECKPOINT_URL
from uztts_events.decode import frames_to_events
from uztts_events.labels import resolve_class_ids
from uztts_events.schema import AudioEvent, EventLabel

SAMPLE_RATE = 16_000
CHUNK_SECONDS = 10.0
FRAMES_PER_CHUNK = 250
FRAME_SECONDS = CHUNK_SECONDS / FRAMES_PER_CHUNK


def checkpoint_path(url: str = SED_CHECKPOINT_URL) -> Path:
    filename = Path(urlparse(url).path).name
    target = data_root() / "models" / "psed" / filename
    if not target.is_file():
        from torch.hub import download_url_to_file

        target.parent.mkdir(parents=True, exist_ok=True)
        download_url_to_file(url, str(target))
    return target


class FrameSedLocalizer:
    def __init__(
        self,
        checkpoint_url: str = SED_CHECKPOINT_URL,
        device: str = "cpu",
    ) -> None:
        import torch

        from uztts_events._vendor.psed.audioset_classes import as_strong_train_classes
        from uztts_events._vendor.psed.wrapper import FramePredictor

        self._torch = torch
        model = FramePredictor()  # type: ignore[no-untyped-call]
        model.load_checkpoint(  # type: ignore[no-untyped-call]
            str(checkpoint_path(checkpoint_url))
        )
        self._model = model.to(device).eval()
        self._device = device
        self._class_ids = resolve_class_ids(list(as_strong_train_classes))

    def frame_scores(self, waveform: np.ndarray) -> dict[EventLabel, np.ndarray]:
        torch = self._torch
        chunk_samples = int(CHUNK_SECONDS * SAMPLE_RATE)
        chunks = []
        for offset in range(0, max(len(waveform), 1), chunk_samples):
            chunk = waveform[offset : offset + chunk_samples]
            padded = np.zeros(chunk_samples, dtype=np.float32)
            padded[: len(chunk)] = chunk
            chunks.append(padded)
        audio = torch.from_numpy(np.stack(chunks))
        with torch.no_grad():
            logits = self._model(audio.to(self._device))
        probs = torch.sigmoid(logits).cpu().numpy()
        joined = np.concatenate(list(probs), axis=1)
        return {
            label: joined[class_ids].max(axis=0)
            for label, class_ids in self._class_ids.items()
        }

    def localize(
        self,
        waveform: np.ndarray,
        labels: set[EventLabel],
        *,
        frame_thresholds: dict[str, float],
        median_window: int = 9,
    ) -> list[AudioEvent]:
        scores = self.frame_scores(waveform)
        max_seconds = len(waveform) / SAMPLE_RATE
        events: list[AudioEvent] = []
        for label in labels:
            events.extend(
                frames_to_events(
                    label,
                    scores[label],
                    threshold=frame_thresholds[label.value],
                    frame_seconds=FRAME_SECONDS,
                    max_seconds=max_seconds,
                    median_window=median_window,
                )
            )
        return sorted(events, key=lambda event: (event.start, event.end))
