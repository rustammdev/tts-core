from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

SED_CHECKPOINT_URL = (
    "https://github.com/fschmid56/PretrainedSED/releases/download/v0.0.1/"
    "frame_mn10_strong_1.pt"
)


def _default_screen_thresholds() -> dict[str, float]:
    return {"laughter": 0.05, "music": 0.2, "applause": 0.1, "cough": 0.1}


def _default_frame_thresholds() -> dict[str, float]:
    return {"laughter": 0.3, "music": 0.3, "applause": 0.3, "cough": 0.3}


@dataclass
class EventsConfig:
    ced_model: str = "mispeech/ced-small"
    ced_revision: str = ""
    sed_checkpoint_url: str = SED_CHECKPOINT_URL
    device: str = "cpu"
    screen_thresholds: dict[str, float] = field(
        default_factory=_default_screen_thresholds
    )
    frame_thresholds: dict[str, float] = field(
        default_factory=_default_frame_thresholds
    )
    median_window: int = 9
    merge_gap: float = 0.5
    min_duration: float = 0.2


def load_config(path: Path) -> EventsConfig:
    from omegaconf import OmegaConf

    merged = OmegaConf.merge(OmegaConf.structured(EventsConfig), OmegaConf.load(path))
    config = OmegaConf.to_object(merged)
    assert isinstance(config, EventsConfig)
    return config
