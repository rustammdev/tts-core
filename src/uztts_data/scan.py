from __future__ import annotations

import hashlib
import wave
from pathlib import Path

from uztts_data.ingest import DONE_MARKER, VideoMeta
from uztts_data.schema import License, Segment


def scan_raw(raw: Path, data_root: Path) -> tuple[list[Segment], list[str]]:
    segments: list[Segment] = []
    issues: list[str] = []
    for meta_path in sorted(raw.glob("*/*/meta.json")):
        video_dir = meta_path.parent
        audio = video_dir / "audio.wav"
        if not (video_dir / DONE_MARKER).is_file() or not audio.is_file():
            continue
        try:
            segments.append(_segment_from_video(meta_path, audio, data_root))
        except Exception as exc:
            issues.append(f"{video_dir}: {type(exc).__name__}: {exc}")
    return segments, issues


def _segment_from_video(meta_path: Path, audio: Path, data_root: Path) -> Segment:
    meta = VideoMeta.model_validate_json(meta_path.read_text(encoding="utf-8"))
    channel_id = meta.channel_id or meta_path.parent.parent.name
    duration, sample_rate = _wav_info(audio)
    return Segment(
        id=f"{channel_id}_{video_key(meta.video_id)}",
        audio_path=_relative_to(audio, data_root),
        speaker_id=f"{channel_id}_c0",
        channel_id=channel_id,
        duration=duration,
        sample_rate=sample_rate,
        source="youtube",
        license=License.WEB_SCRAPED,
    )


def video_key(video_id: str) -> str:
    return hashlib.sha1(video_id.encode("utf-8")).hexdigest()[:10]


def _wav_info(path: Path) -> tuple[float, int]:
    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        return handle.getnframes() / rate, rate


def _relative_to(path: Path, base: Path) -> Path:
    resolved = path.resolve()
    try:
        return resolved.relative_to(base.resolve())
    except ValueError:
        return resolved
