from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EventLabel(StrEnum):
    LAUGHTER = "laughter"
    MUSIC = "music"
    APPLAUSE = "applause"
    COUGH = "cough"


Seconds = Annotated[float, Field(ge=0.0)]
Score = Annotated[float, Field(ge=0.0, le=1.0)]


class AudioEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    label: EventLabel
    start: Seconds
    end: Seconds
    score: Score

    @model_validator(mode="after")
    def _end_after_start(self) -> Self:
        if self.end <= self.start:
            raise ValueError("end must be greater than start")
        return self


class Word(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    text: Annotated[str, Field(min_length=1)]
    start: Seconds
    end: Seconds

    @model_validator(mode="after")
    def _end_after_start(self) -> Self:
        if self.end < self.start:
            raise ValueError("end must not precede start")
        return self
