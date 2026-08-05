from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Protocol, Self
from urllib.parse import urlparse

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from uztts_data.manifest import (
    ManifestError,
    ManifestIssue,
    ManifestReport,
    describe_validation_error,
    iter_jsonl_lines,
)
from uztts_data.schema import Identifier, QualityTag, Text

RAW_HOURS_TARGET = 1000.0


class Genre(StrEnum):
    CONVERSATION = "conversation"
    NEWS = "news"
    EDUCATION = "education"
    VLOG = "vlog"
    AUDIOBOOK = "audiobook"
    OTHER = "other"


class Script(StrEnum):
    LATIN = "latin"
    CYRILLIC = "cyrillic"
    MIXED = "mixed"


class ChannelStatus(StrEnum):
    CANDIDATE = "candidate"
    APPROVED = "approved"
    REJECTED = "rejected"


class Channel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    channel_id: Identifier
    url: Text
    name: Text
    genre: Genre
    script: Script
    est_quality: QualityTag
    status: ChannelStatus
    reject_reason: Text | None = None
    notes: Text | None = None

    @field_validator("url")
    @classmethod
    def _require_youtube_url(cls, value: str) -> str:
        host = urlparse(value).hostname or ""
        if host != "youtu.be" and not host.endswith("youtube.com"):
            raise ValueError(f"not a YouTube URL: {value}")
        return value

    @model_validator(mode="after")
    def _reason_only_when_rejected(self) -> Self:
        rejected = self.status is ChannelStatus.REJECTED
        if rejected and self.reject_reason is None:
            raise ValueError("rejected channel needs reject_reason")
        if not rejected and self.reject_reason is not None:
            raise ValueError("reject_reason is set but status is not rejected")
        return self


class ChannelStat(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    channel_id: Identifier
    video_count: Annotated[int, Field(ge=0)]
    hours: Annotated[float, Field(ge=0.0)]


class VideoLister(Protocol):
    def video_durations(self, url: str) -> Sequence[float]: ...


def read_registry(path: Path) -> Iterator[Channel]:
    for line_number, line in iter_jsonl_lines(path):
        try:
            yield Channel.model_validate_json(line)
        except ValidationError as exc:
            raise ManifestError(
                f"{path}:{line_number}: {describe_validation_error(exc)}"
            ) from exc


def validate_registry(path: Path) -> ManifestReport:
    issues: list[ManifestIssue] = []
    seen: set[str] = set()
    total = 0
    for line_number, line in iter_jsonl_lines(path):
        total += 1
        try:
            channel = Channel.model_validate_json(line)
        except ValidationError as exc:
            issues.append(ManifestIssue(line_number, describe_validation_error(exc)))
            continue
        if channel.channel_id in seen:
            issues.append(
                ManifestIssue(
                    line_number, f"duplicate channel_id: {channel.channel_id}"
                )
            )
        seen.add(channel.channel_id)
    return ManifestReport(total=total, issues=tuple(issues))


def read_stats(path: Path) -> dict[str, ChannelStat]:
    stats: dict[str, ChannelStat] = {}
    for line_number, line in iter_jsonl_lines(path):
        try:
            stat = ChannelStat.model_validate_json(line)
        except ValidationError as exc:
            raise ManifestError(
                f"{path}:{line_number}: {describe_validation_error(exc)}"
            ) from exc
        stats[stat.channel_id] = stat
    return stats


def merge_stats(
    channels: Sequence[Channel],
    existing: Mapping[str, ChannelStat],
    lister: VideoLister,
    refresh: bool = False,
) -> tuple[list[ChannelStat], list[str]]:
    stats: list[ChannelStat] = []
    errors: list[str] = []
    for channel in channels:
        if channel.status is ChannelStatus.REJECTED:
            continue
        if not refresh and channel.channel_id in existing:
            stats.append(existing[channel.channel_id])
            continue
        try:
            durations = lister.video_durations(channel.url)
        except Exception as exc:
            errors.append(f"{channel.channel_id}: {type(exc).__name__}: {exc}")
            continue
        stats.append(
            ChannelStat(
                channel_id=channel.channel_id,
                video_count=len(durations),
                hours=round(sum(durations) / 3600, 2),
            )
        )
    return stats, errors


def render_report(
    channels: Sequence[Channel],
    stats: Mapping[str, ChannelStat],
    target_hours: float = RAW_HOURS_TARGET,
) -> str:
    active = [c for c in channels if c.status is not ChannelStatus.REJECTED]
    counted = [c for c in active if c.channel_id in stats]
    total_hours = sum(stats[c.channel_id].hours for c in counted)
    total_videos = sum(stats[c.channel_id].video_count for c in counted)

    lines = [f"{'genre':<13} {'channels':>8} {'videos':>7} {'hours':>8} {'share':>6}"]
    for genre in Genre:
        members = [c for c in counted if c.genre is genre]
        if not members:
            continue
        hours = sum(stats[c.channel_id].hours for c in members)
        videos = sum(stats[c.channel_id].video_count for c in members)
        share = hours / total_hours * 100 if total_hours else 0.0
        lines.append(
            f"{genre:<13} {len(members):>8} {videos:>7} {hours:>8.1f} {share:>5.1f}%"
        )
    total_share = 100.0 if total_hours else 0.0
    lines.append(
        f"{'total':<13} {len(counted):>8} {total_videos:>7}"
        f" {total_hours:>8.1f} {total_share:>5.1f}%"
    )

    missing = len(active) - len(counted)
    if missing:
        lines.append(f"missing stats: {missing} channel(s)")
    verdict = "met" if total_hours >= target_hours else "not met"
    lines.append(
        f"Gate-2: raw hours {total_hours:.1f} / {target_hours:.0f} — {verdict}"
    )
    return "\n".join(lines)


class YtDlpLister:
    def video_durations(self, url: str) -> list[float]:
        from yt_dlp import YoutubeDL

        options = {
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
            "extract_flat": "in_playlist",
            "skip_download": True,
        }
        with YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=False)
        if info is None:
            raise ManifestError(f"yt-dlp returned no metadata for {url}")
        return list(_durations(info))


def _durations(info: Mapping[str, Any]) -> Iterator[float]:
    for entry in info.get("entries") or []:
        if not isinstance(entry, Mapping):
            continue
        if entry.get("entries"):
            yield from _durations(entry)
        elif entry.get("duration"):
            yield float(entry["duration"])
