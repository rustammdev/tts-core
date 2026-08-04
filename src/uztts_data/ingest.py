from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Protocol, TypeVar
from urllib.parse import parse_qs, urlparse

import typer
from pydantic import BaseModel, ConfigDict

SAMPLE_RATE = 24000
MAX_ATTEMPTS = 3
DONE_MARKER = ".done"
FAILURE_LOG = "_failed.jsonl"

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
    FAILED = "failed"


class VideoMeta(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    video_id: str
    url: str
    title: str
    channel: str | None
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


class MediaSource(Protocol):
    def metadata(self, url: str) -> Mapping[str, Any]: ...

    def fetch(self, url: str, destination: Path) -> FetchedMedia: ...


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


def ingest(urls: Sequence[str], root: Path, source: MediaSource) -> list[Outcome]:
    root.mkdir(parents=True, exist_ok=True)
    outcomes: list[Outcome] = []
    for url in urls:
        try:
            outcome = ingest_url(url, root, source)
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


def ingest_url(url: str, root: Path, source: MediaSource) -> Outcome:
    known_id = video_id_from_url(url)
    if known_id is not None and _is_done(root / known_id):
        return Outcome(url=url, status=Status.SKIPPED, video_id=known_id)

    meta = parse_metadata(_retry(lambda: source.metadata(url)), url)
    destination = root / meta.video_id
    if _is_done(destination):
        return Outcome(url=url, status=Status.SKIPPED, video_id=meta.video_id)

    destination.mkdir(parents=True, exist_ok=True)
    (destination / "meta.json").write_text(
        meta.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    media = _retry(lambda: source.fetch(url, destination))
    if not media.audio.is_file():
        raise IngestError(f"{meta.video_id}: audio missing at {media.audio}")
    (destination / DONE_MARKER).write_text("ingest\n", encoding="utf-8")
    return Outcome(url=url, status=Status.INGESTED, video_id=meta.video_id)


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
        Path, typer.Option("--urls", exists=True, dir_okay=False, readable=True)
    ],
    out: Annotated[Path, typer.Option("--out")] = Path("data/raw"),
    sample_rate: Annotated[int, typer.Option("--sample-rate", min=8000)] = SAMPLE_RATE,
) -> None:
    try:
        require_ffmpeg()
    except IngestError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc

    outcomes = ingest(read_urls(urls), out, YtDlpSource(sample_rate=sample_rate))
    for outcome in outcomes:
        label = outcome.video_id or outcome.url
        if outcome.status is Status.FAILED:
            typer.echo(f"failed: {label}: {outcome.error}", err=True)
        else:
            typer.echo(f"{outcome.status}: {label}")

    counts = Counter(outcome.status for outcome in outcomes)
    typer.echo(
        f"ingested={counts[Status.INGESTED]} "
        f"skipped={counts[Status.SKIPPED]} "
        f"failed={counts[Status.FAILED]}"
    )
    if counts[Status.FAILED]:
        raise typer.Exit(1)


def _is_done(destination: Path) -> bool:
    return (destination / DONE_MARKER).is_file()


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
