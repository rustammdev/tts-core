from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Protocol, TypeVar
from urllib.parse import parse_qs, urlparse

import typer
from pydantic import BaseModel, ConfigDict

from uztts_data.channels import Channel, ChannelStatus, read_registry
from uztts_data.paths import raw_root

SAMPLE_RATE = 24000
MAX_ATTEMPTS = 3
DONE_MARKER = ".done"
FILTERED_MARKER = ".filtered"
FAILURE_LOG = "_failed.jsonl"
ADHOC_CHANNEL = "adhoc"

_VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")
_PATH_PREFIXES = ("shorts", "embed", "live", "v")
_NON_AUDIO_SUFFIXES = frozenset({".vtt", ".srt", ".part", ".ytdl", ".temp", ".json"})

T = TypeVar("T")


class IngestError(Exception):
    pass


class SubtitleKind(StrEnum):
    MANUAL = "manual"
    AUTOMATIC = "automatic"


class Status(StrEnum):
    INGESTED = "ingested"
    SKIPPED = "skipped"
    FILTERED = "filtered"
    CAPPED = "capped"
    FAILED = "failed"


class VideoMeta(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    video_id: str
    url: str
    title: str
    channel: str | None
    channel_id: str | None = None
    duration: float
    language: str | None
    upload_date: str | None
    uz_subtitles: SubtitleKind | None


@dataclass(frozen=True, slots=True)
class FetchedMedia:
    audio: Path
    subtitles: Path | None


@dataclass(frozen=True, slots=True)
class Outcome:
    url: str
    status: Status
    video_id: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class DurationBounds:
    min_seconds: float | None = None
    max_seconds: float | None = None

    def rejects(self, duration: float) -> str | None:
        if self.min_seconds is not None and duration < self.min_seconds:
            return f"duration {duration:.0f}s < min {self.min_seconds:.0f}s"
        if self.max_seconds is not None and duration > self.max_seconds:
            return f"duration {duration:.0f}s > max {self.max_seconds:.0f}s"
        return None


class MediaSource(Protocol):
    def metadata(self, url: str) -> Mapping[str, Any]: ...

    def fetch(self, url: str, destination: Path) -> FetchedMedia: ...


class VideoSource(MediaSource, Protocol):
    def video_urls(self, url: str) -> Sequence[str]: ...


def read_urls(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    urls = (line.strip() for line in lines)
    return list(dict.fromkeys(url for url in urls if url and not url.startswith("#")))


def video_id_from_url(url: str) -> str | None:
    parsed = urlparse(url)
    if parsed.hostname == "youtu.be":
        candidate = parsed.path.lstrip("/")
    else:
        parts = parsed.path.strip("/").split("/")
        if len(parts) == 2 and parts[0] in _PATH_PREFIXES:
            candidate = parts[1]
        else:
            candidate = next(iter(parse_qs(parsed.query).get("v", [])), "")
    return candidate if _VIDEO_ID.match(candidate) else None


def parse_metadata(info: Mapping[str, Any], url: str) -> VideoMeta:
    video_id = info.get("id")
    duration = info.get("duration")
    if not video_id:
        raise IngestError(f"no video id in metadata for {url}")
    if duration is None:
        raise IngestError(f"no duration for {video_id} (live stream?)")
    return VideoMeta(
        video_id=str(video_id),
        url=str(info.get("webpage_url") or url),
        title=str(info.get("title") or ""),
        channel=_optional_str(info.get("channel") or info.get("uploader")),
        duration=float(duration),
        language=_optional_str(info.get("language")),
        upload_date=_iso_date(info.get("upload_date")),
        uz_subtitles=_uz_subtitle_kind(info),
    )


def ingest(
    urls: Sequence[str],
    root: Path,
    source: MediaSource,
    channel_id: str | None = None,
    bounds: DurationBounds | None = None,
) -> list[Outcome]:
    root.mkdir(parents=True, exist_ok=True)
    outcomes: list[Outcome] = []
    for url in urls:
        try:
            outcome = ingest_url(url, root, source, channel_id, bounds)
        except Exception as exc:
            outcome = Outcome(
                url=url,
                status=Status.FAILED,
                video_id=video_id_from_url(url),
                error=f"{type(exc).__name__}: {exc}",
            )
            _log_failure(root, outcome)
        outcomes.append(outcome)
    return outcomes


def ingest_channels(
    channels: Sequence[Channel],
    root: Path,
    source: VideoSource,
    bounds: DurationBounds | None = None,
    max_channel_hours: float | None = None,
) -> list[Outcome]:
    outcomes: list[Outcome] = []
    for channel in channels:
        if channel.status is not ChannelStatus.APPROVED:
            continue
        channel_root = root / channel.channel_id
        try:
            urls = _list_videos(source, channel.url)
        except Exception as exc:
            outcome = Outcome(
                url=channel.url,
                status=Status.FAILED,
                error=f"{type(exc).__name__}: {exc}",
            )
            channel_root.mkdir(parents=True, exist_ok=True)
            _log_failure(channel_root, outcome)
            outcomes.append(outcome)
            continue
        outcomes.extend(
            _ingest_channel_videos(
                urls,
                channel_root,
                source,
                channel.channel_id,
                bounds,
                max_channel_hours,
            )
        )
    return outcomes


def _ingest_channel_videos(
    urls: Sequence[str],
    channel_root: Path,
    source: MediaSource,
    channel_id: str,
    bounds: DurationBounds | None,
    max_channel_hours: float | None,
) -> list[Outcome]:
    cap_seconds = None if max_channel_hours is None else max_channel_hours * 3600
    seconds = _finished_seconds(channel_root) if cap_seconds is not None else 0.0
    channel_root.mkdir(parents=True, exist_ok=True)
    outcomes: list[Outcome] = []
    for index, url in enumerate(urls):
        if cap_seconds is not None and seconds >= cap_seconds:
            outcomes.append(
                Outcome(
                    url=url,
                    status=Status.CAPPED,
                    error=(
                        f"{channel_id}: {seconds / 3600:.2f}h >= "
                        f"{cap_seconds / 3600:.2f}h cap, "
                        f"{len(urls) - index} video(s) left"
                    ),
                )
            )
            break
        try:
            outcome = ingest_url(url, channel_root, source, channel_id, bounds)
        except Exception as exc:
            outcome = Outcome(
                url=url,
                status=Status.FAILED,
                video_id=video_id_from_url(url),
                error=f"{type(exc).__name__}: {exc}",
            )
            _log_failure(channel_root, outcome)
        outcomes.append(outcome)
        if (
            cap_seconds is not None
            and outcome.status is Status.INGESTED
            and outcome.video_id is not None
        ):
            seconds += _video_seconds(channel_root / outcome.video_id)
    return outcomes


def _finished_seconds(channel_root: Path) -> float:
    return sum(
        _video_seconds(marker.parent)
        for marker in channel_root.glob(f"*/{DONE_MARKER}")
    )


def _video_seconds(video_dir: Path) -> float:
    meta_path = video_dir / "meta.json"
    if not meta_path.is_file():
        return 0.0
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 0.0
    duration = payload.get("duration")
    return float(duration) if isinstance(duration, int | float) else 0.0


def ingest_url(
    url: str,
    root: Path,
    source: MediaSource,
    channel_id: str | None = None,
    bounds: DurationBounds | None = None,
) -> Outcome:
    known_id = video_id_from_url(url)
    if known_id is not None and _is_finished(root / known_id):
        return Outcome(url=url, status=Status.SKIPPED, video_id=known_id)

    meta = parse_metadata(_retry(lambda: source.metadata(url)), url)
    meta = meta.model_copy(update={"channel_id": channel_id})
    destination = root / meta.video_id
    if _is_finished(destination):
        return Outcome(url=url, status=Status.SKIPPED, video_id=meta.video_id)

    reason = bounds.rejects(meta.duration) if bounds is not None else None
    if reason is not None:
        destination.mkdir(parents=True, exist_ok=True)
        (destination / FILTERED_MARKER).write_text(reason + "\n", encoding="utf-8")
        return Outcome(
            url=url, status=Status.FILTERED, video_id=meta.video_id, error=reason
        )

    destination.mkdir(parents=True, exist_ok=True)
    (destination / "meta.json").write_text(
        meta.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    media = _retry(lambda: source.fetch(url, destination))
    if not media.audio.is_file():
        raise IngestError(f"{meta.video_id}: audio missing at {media.audio}")
    (destination / DONE_MARKER).write_text("ingest\n", encoding="utf-8")
    return Outcome(url=url, status=Status.INGESTED, video_id=meta.video_id)


def _list_videos(source: VideoSource, url: str) -> Sequence[str]:
    return _retry(lambda: source.video_urls(url))


@dataclass(frozen=True, slots=True)
class YtDlpSource:
    sample_rate: int = SAMPLE_RATE

    def metadata(self, url: str) -> Mapping[str, Any]:
        from yt_dlp import YoutubeDL

        with YoutubeDL(self._options()) as ydl:
            info = ydl.extract_info(url, download=False)
        if info is None:
            raise IngestError(f"yt-dlp returned no metadata for {url}")
        return dict(info)

    def video_urls(self, url: str) -> list[str]:
        from yt_dlp import YoutubeDL

        options = self._options() | {
            "noplaylist": False,
            "extract_flat": "in_playlist",
            "skip_download": True,
        }
        with YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=False)
        if info is None:
            raise IngestError(f"yt-dlp returned no video list for {url}")
        return watch_urls(dict(info))

    def fetch(self, url: str, destination: Path) -> FetchedMedia:
        from yt_dlp import YoutubeDL

        destination.mkdir(parents=True, exist_ok=True)
        options = self._options() | {
            "format": "bestaudio/best",
            "outtmpl": str(destination / "source.%(ext)s"),
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": ["uz"],
            "subtitlesformat": "vtt",
        }
        with YoutubeDL(options) as ydl:
            ydl.extract_info(url, download=True)

        audio = destination / "audio.wav"
        downloaded = _downloaded_audio(destination)
        to_pcm_wav(downloaded, audio, self.sample_rate)
        downloaded.unlink()
        return FetchedMedia(audio=audio, subtitles=_collect_subtitles(destination))

    def _options(self) -> dict[str, Any]:
        return {
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
            "noplaylist": True,
            "retries": MAX_ATTEMPTS,
        }


def watch_urls(info: Mapping[str, Any]) -> list[str]:
    ids = dict.fromkeys(_video_ids(info))
    return [f"https://www.youtube.com/watch?v={video_id}" for video_id in ids]


def _video_ids(info: Mapping[str, Any]) -> Iterator[str]:
    for entry in info.get("entries") or []:
        if not isinstance(entry, Mapping):
            continue
        if entry.get("entries"):
            yield from _video_ids(entry)
        else:
            video_id = str(entry.get("id") or "")
            if _VIDEO_ID.match(video_id):
                yield video_id


def require_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise IngestError("ffmpeg not found on PATH")
    return ffmpeg


def to_pcm_wav(source: Path, target: Path, sample_rate: int) -> None:
    ffmpeg = require_ffmpeg()
    result = subprocess.run(
        [
            ffmpeg,
            "-nostdin",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-acodec",
            "pcm_s16le",
            str(target),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise IngestError(f"ffmpeg failed on {source.name}: {result.stderr.strip()}")


app = typer.Typer(add_completion=False)


@app.command()
def main(
    urls: Annotated[
        Path | None,
        typer.Option("--urls", exists=True, dir_okay=False, readable=True),
    ] = None,
    channels: Annotated[
        Path | None,
        typer.Option("--channels", exists=True, dir_okay=False, readable=True),
    ] = None,
    only: Annotated[list[str] | None, typer.Option("--only")] = None,
    out: Annotated[Path | None, typer.Option("--out")] = None,
    sample_rate: Annotated[int, typer.Option("--sample-rate", min=8000)] = SAMPLE_RATE,
    min_duration: Annotated[float, typer.Option("--min-duration", min=0.0)] = 60.0,
    max_duration: Annotated[float, typer.Option("--max-duration", min=0.0)] = 14400.0,
    max_channel_hours: Annotated[
        float, typer.Option("--max-channel-hours", min=0.0)
    ] = 0.0,
) -> None:
    if (urls is None) == (channels is None):
        typer.echo("pass exactly one of --urls or --channels", err=True)
        raise typer.Exit(2)
    try:
        require_ffmpeg()
    except IngestError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc

    root = out if out is not None else raw_root()
    bounds = DurationBounds(
        min_seconds=min_duration or None, max_seconds=max_duration or None
    )
    source = YtDlpSource(sample_rate=sample_rate)

    if urls is not None:
        outcomes = ingest(
            read_urls(urls), root / ADHOC_CHANNEL, source, ADHOC_CHANNEL, bounds
        )
    else:
        assert channels is not None
        selected = list(read_registry(channels))
        if only:
            wanted = set(only)
            selected = [c for c in selected if c.channel_id in wanted]
        outcomes = ingest_channels(
            selected, root, source, bounds, max_channel_hours or None
        )

    for outcome in outcomes:
        label = outcome.video_id or outcome.url
        if outcome.status is Status.FAILED:
            typer.echo(f"failed: {label}: {outcome.error}", err=True)
        elif outcome.status is Status.CAPPED:
            typer.echo(f"capped: {outcome.error}")
        else:
            typer.echo(f"{outcome.status}: {label}")

    counts = Counter(outcome.status for outcome in outcomes)
    typer.echo(
        f"ingested={counts[Status.INGESTED]} "
        f"skipped={counts[Status.SKIPPED]} "
        f"filtered={counts[Status.FILTERED]} "
        f"capped={counts[Status.CAPPED]} "
        f"failed={counts[Status.FAILED]}"
    )
    if counts[Status.FAILED]:
        raise typer.Exit(1)


def _is_finished(destination: Path) -> bool:
    return (destination / DONE_MARKER).is_file() or (
        destination / FILTERED_MARKER
    ).is_file()


def _log_failure(root: Path, outcome: Outcome) -> None:
    record = {
        "url": outcome.url,
        "video_id": outcome.video_id,
        "error": outcome.error,
    }
    with (root / FAILURE_LOG).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _retry(operation: Callable[[], T]) -> T:
    failure: Exception | None = None
    for _ in range(MAX_ATTEMPTS):
        try:
            return operation()
        except Exception as exc:
            failure = exc
    raise IngestError(f"gave up after {MAX_ATTEMPTS} attempts: {failure}") from failure


def _downloaded_audio(destination: Path) -> Path:
    candidates = [
        path
        for path in destination.glob("source.*")
        if path.is_file() and path.suffix not in _NON_AUDIO_SUFFIXES
    ]
    if len(candidates) != 1:
        raise IngestError(f"expected one downloaded stream, found {len(candidates)}")
    return candidates[0]


def _collect_subtitles(destination: Path) -> Path | None:
    tracks = sorted(destination.glob("source.*.vtt"))
    if not tracks:
        return None
    target = destination / "subs.vtt"
    tracks[0].replace(target)
    for extra in tracks[1:]:
        extra.unlink()
    return target


def _uz_subtitle_kind(info: Mapping[str, Any]) -> SubtitleKind | None:
    if _has_uzbek(info.get("subtitles")):
        return SubtitleKind.MANUAL
    if _has_uzbek(info.get("automatic_captions")):
        return SubtitleKind.AUTOMATIC
    return None


def _has_uzbek(tracks: object) -> bool:
    if not isinstance(tracks, Mapping):
        return False
    return any(str(code).split("-")[0] == "uz" for code in tracks)


def _optional_str(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _iso_date(value: object) -> str | None:
    text = str(value or "")
    if len(text) != 8 or not text.isdigit():
        return None
    return f"{text[:4]}-{text[4:6]}-{text[6:]}"
