from __future__ import annotations

from typing import Protocol

import pytest

from uztts_data import Segment


class SegmentFactory(Protocol):
    def __call__(self, index: int, **overrides: object) -> Segment: ...


@pytest.fixture
def make_segment() -> SegmentFactory:
    def factory(index: int, **overrides: object) -> Segment:
        payload: dict[str, object] = {
            "id": f"spk001_{index:06d}",
            "audio_path": f"data/processed/spk001/{index:06d}.wav",
            "speaker_id": "spk001",
            "duration": 2.0,
            "sample_rate": 24000,
            "source": "own_recording",
            "license": "owned",
        }
        return Segment.model_validate(payload | overrides)

    return factory
