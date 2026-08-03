from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_serializer,
    field_validator,
    model_validator,
)


class QualityTag(StrEnum):
    CLEAN = "clean"
    MEDIUM = "medium"
    NOISY = "noisy"


class License(StrEnum):
    OWNED = "owned"
    LICENSED = "licensed"
    PUBLIC_DOMAIN = "public_domain"


Identifier = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9_-]*$")]
Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Duration = Annotated[float, Field(gt=0.0)]
SampleRate = Annotated[int, Field(gt=0)]
Cer = Annotated[float, Field(ge=0.0)]


class Segment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: Identifier
    audio_path: Path
    text: Text | None = None
    text_normalized: Text | None = None
    speaker_id: Identifier
    duration: Duration
    sample_rate: SampleRate
    quality_tag: QualityTag | None = None
    snr_db: float | None = None
    source: Text
    license: License
    style_caption: Text | None = None
    asr_cer: Cer | None = None

    @field_validator("audio_path", mode="before")
    @classmethod
    def _accept_windows_separators(cls, value: object) -> object:
        return value.replace("\\", "/") if isinstance(value, str) else value

    @field_serializer("audio_path")
    def _serialize_audio_path(self, value: Path) -> str:
        return value.as_posix()

    @model_validator(mode="after")
    def _normalized_text_needs_raw_text(self) -> Self:
        if self.text_normalized is not None and self.text is None:
            raise ValueError("text_normalized is set but text is missing")
        return self
