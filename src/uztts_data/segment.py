from __future__ import annotations

import math
import wave
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Protocol

import typer

if TYPE_CHECKING:
    import torch

from uztts_data.ingest import DONE_MARKER
from uztts_data.manifest import read_manifest, write_manifest
from uztts_data.paths import data_root, manifests_root
from uztts_data.schema import Segment

MIN_SEGMENT_SECONDS = 2.0
MAX_SEGMENT_SECONDS = 20.0


@dataclass(frozen=True, slots=True)
class Span:
    start: float
    end: float

    @property
    def seconds(self) -> float:
        return self.end - self.start


class SpeechDetector(Protocol):
    def speech_spans(self, audio: Path) -> Sequence[Span]: ...


@dataclass(frozen=True, slots=True)
class SegmentOutcome:
    video_id: str
    status: str
    error: str | None = None


def clamp_spans(
    spans: Sequence[Span], min_seconds: float, max_seconds: float
) -> tuple[list[Span], float]:
    kept: list[Span] = []
    dropped = 0.0
    for span in spans:
        if span.seconds < min_seconds:
            dropped += span.seconds
            continue
        if span.seconds <= max_seconds:
            kept.append(span)
            continue
        pieces = math.ceil(span.seconds / max_seconds)
        step = span.seconds / pieces
        kept.extend(
            Span(span.start + index * step, span.start + (index + 1) * step)
            for index in range(pieces)
        )
    return kept, dropped


def cut_wav(source: Path, spans: Sequence[Span], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    with wave.open(str(source), "rb") as reader:
        rate = reader.getframerate()
        total = reader.getnframes()
        for index, span in enumerate(spans):
            start = min(int(span.start * rate), total)
            end = min(int(span.end * rate), total)
            if end <= start:
                continue
            reader.setpos(start)
            frames = reader.readframes(end - start)
            target = out_dir / f"{index:04d}.wav"
            with wave.open(str(target), "wb") as writer:
                writer.setnchannels(reader.getnchannels())
                writer.setsampwidth(reader.getsampwidth())
                writer.setframerate(rate)
                writer.writeframes(frames)
            written.append(target)
    return written


def video_out_dir(out_root: Path, video: Segment) -> Path | None:
    if video.channel_id is None:
        return None
    key = video.id.removeprefix(f"{video.channel_id}_")
    if not key or key == video.id:
        return None
    return out_root / video.channel_id / key


def segment_videos(
    videos: Sequence[Segment],
    root: Path,
    out_root: Path,
    detector: SpeechDetector,
    min_seconds: float = MIN_SEGMENT_SECONDS,
    max_seconds: float = MAX_SEGMENT_SECONDS,
) -> list[SegmentOutcome]:
    outcomes: list[SegmentOutcome] = []
    for video in videos:
        out_dir = video_out_dir(out_root, video)
        if out_dir is None:
            outcomes.append(
                SegmentOutcome(video.id, "failed", "cannot derive output dir")
            )
            continue
        if (out_dir / DONE_MARKER).is_file():
            outcomes.append(SegmentOutcome(video.id, "skipped"))
            continue
        audio = root / video.audio_path
        if not audio.is_file():
            outcomes.append(SegmentOutcome(video.id, "failed", f"missing {audio}"))
            continue
        try:
            spans, _ = clamp_spans(
                detector.speech_spans(audio), min_seconds, max_seconds
            )
            for stale in out_dir.glob("*.wav"):
                stale.unlink()
            cut_wav(audio, spans, out_dir)
        except Exception as exc:
            outcomes.append(
                SegmentOutcome(video.id, "failed", f"{type(exc).__name__}: {exc}")
            )
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / DONE_MARKER).write_text("segment\n", encoding="utf-8")
        outcomes.append(SegmentOutcome(video.id, "segmented"))
    return outcomes


def scan_segments(out_root: Path, root: Path) -> tuple[list[Segment], list[str]]:
    segments: list[Segment] = []
    issues: list[str] = []
    for marker in sorted(out_root.glob(f"*/*/{DONE_MARKER}")):
        video_dir = marker.parent
        channel_id = video_dir.parent.name
        key = video_dir.name
        for wav_path in sorted(video_dir.glob("*.wav")):
            try:
                with wave.open(str(wav_path), "rb") as handle:
                    rate = handle.getframerate()
                    frames = handle.getnframes()
            except (OSError, wave.Error) as exc:
                issues.append(f"{wav_path}: {exc}")
                continue
            if frames == 0 or rate == 0:
                issues.append(f"{wav_path}: empty audio")
                continue
            segments.append(
                Segment(
                    id=f"{channel_id}_{key}_{wav_path.stem}",
                    audio_path=wav_path.relative_to(root),
                    speaker_id=f"{channel_id}_c0",
                    channel_id=channel_id,
                    duration=frames / rate,
                    sample_rate=rate,
                    source="youtube",
                    license="web_scraped",
                )
            )
    return segments, issues


class SileroDetector:
    def __init__(self, max_seconds: float = MAX_SEGMENT_SECONDS) -> None:
        self._max_seconds = max_seconds
        self._model: Any = None

    def speech_spans(self, audio: Path) -> list[Span]:
        from silero_vad import get_speech_timestamps, load_silero_vad

        if self._model is None:
            self._model = load_silero_vad()
        samples = _load_mono_16k(audio)
        stamps = get_speech_timestamps(
            samples,
            self._model,
            sampling_rate=16000,
            min_speech_duration_ms=250,
            min_silence_duration_ms=500,
            speech_pad_ms=100,
            max_speech_duration_s=self._max_seconds,
            return_seconds=True,
        )
        return [Span(float(s["start"]), float(s["end"])) for s in stamps]


def _load_mono_16k(audio: Path) -> torch.Tensor:
    import torch
    import torchaudio

    with wave.open(str(audio), "rb") as handle:
        if handle.getsampwidth() != 2:
            raise ValueError(f"{audio}: expected 16-bit PCM")
        rate = handle.getframerate()
        channels = handle.getnchannels()
        frames = handle.readframes(handle.getnframes())
    samples = torch.frombuffer(bytearray(frames), dtype=torch.int16).float() / 32768.0
    if channels > 1:
        samples = samples.view(-1, channels).mean(dim=1)
    if rate != 16000:
        samples = torchaudio.functional.resample(samples, rate, 16000)
    return samples


app = typer.Typer(add_completion=False)


@app.command()
def main(
    manifest: Annotated[Path | None, typer.Option("--manifest")] = None,
    out_root: Annotated[Path | None, typer.Option("--out-root")] = None,
    out: Annotated[Path | None, typer.Option("--out")] = None,
    min_seconds: Annotated[
        float, typer.Option("--min-seconds", min=0.1)
    ] = MIN_SEGMENT_SECONDS,
    max_seconds: Annotated[
        float, typer.Option("--max-seconds", min=1.0)
    ] = MAX_SEGMENT_SECONDS,
) -> None:
    root = data_root()
    manifest_path = manifest if manifest is not None else manifests_root() / "raw.jsonl"
    if not manifest_path.is_file():
        typer.echo(f"manifest not found: {manifest_path}", err=True)
        raise typer.Exit(2)
    segments_root = out_root if out_root is not None else root / "interim" / "segments"
    target = out if out is not None else manifests_root() / "segments.jsonl"

    videos = list(read_manifest(manifest_path))
    detector = SileroDetector(max_seconds=max_seconds)
    outcomes = segment_videos(
        videos, root, segments_root, detector, min_seconds, max_seconds
    )
    for outcome in outcomes:
        if outcome.status == "failed":
            typer.echo(f"failed: {outcome.video_id}: {outcome.error}", err=True)

    segments, issues = scan_segments(segments_root, root)
    write_manifest(target, segments)
    for issue in issues:
        typer.echo(f"skipped: {issue}", err=True)

    done_videos = {
        outcome.video_id
        for outcome in outcomes
        if outcome.status in {"segmented", "skipped"}
    }
    raw_seconds = sum(video.duration for video in videos if video.id in done_videos)
    speech_seconds = sum(segment.duration for segment in segments)
    share = speech_seconds / raw_seconds * 100 if raw_seconds else 0.0
    counts = {
        status: sum(1 for outcome in outcomes if outcome.status == status)
        for status in ("segmented", "skipped", "failed")
    }
    typer.echo(
        f"segmented={counts['segmented']} "
        f"skipped={counts['skipped']} "
        f"failed={counts['failed']}"
    )
    typer.echo(
        f"raw {raw_seconds / 3600:.2f} h -> speech {speech_seconds / 3600:.2f} h"
        f" ({share:.0f}%) in {len(segments)} segment(s) -> {target}"
    )
    if counts["failed"]:
        raise typer.Exit(1)
