from __future__ import annotations

import re
import wave
from pathlib import Path

import pytest
from typer.testing import CliRunner

from uztts_data.cli import app
from uztts_data.ingest import DONE_MARKER, SubtitleKind, VideoMeta
from uztts_data.paths import DATA_ROOT_ENV
from uztts_data.scan import scan_raw, video_key
from uztts_data.schema import License

runner = CliRunner()


def write_wav(path: Path, seconds: float = 1.0, rate: int = 24000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x00" * int(seconds * rate))


def make_video_dir(
    raw: Path,
    channel_id: str,
    video_id: str,
    seconds: float = 1.0,
    done: bool = True,
    meta_channel_id: str | None = "unset",
) -> Path:
    video_dir = raw / channel_id / video_id
    video_dir.mkdir(parents=True, exist_ok=True)
    meta = VideoMeta(
        video_id=video_id,
        url=f"https://youtu.be/{video_id}",
        title="Video",
        channel="Kanal",
        channel_id=channel_id if meta_channel_id == "unset" else meta_channel_id,
        duration=seconds,
        language="uz",
        upload_date=None,
        uz_subtitles=SubtitleKind.AUTOMATIC,
    )
    (video_dir / "meta.json").write_text(meta.model_dump_json(), encoding="utf-8")
    write_wav(video_dir / "audio.wav", seconds=seconds)
    if done:
        (video_dir / DONE_MARKER).write_text("ingest\n", encoding="utf-8")
    return video_dir


def test_scan_builds_video_level_segments(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    make_video_dir(raw, "ch_001", "dQw4w9WgXcQ", seconds=2.0)
    make_video_dir(raw, "ch_002", "aQw4w9WgXcQ", seconds=1.0)

    segments, issues = scan_raw(raw, tmp_path)
    assert not issues
    assert len(segments) == 2

    first = segments[0]
    assert first.id == f"ch_001_{video_key('dQw4w9WgXcQ')}"
    assert first.speaker_id == "ch_001_c0"
    assert first.channel_id == "ch_001"
    assert first.duration == pytest.approx(2.0)
    assert first.sample_rate == 24000
    assert first.license is License.WEB_SCRAPED
    assert first.audio_path == Path("raw/ch_001/dQw4w9WgXcQ/audio.wav")


def test_scan_skips_unfinished_and_audioless_videos(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    make_video_dir(raw, "ch_001", "dQw4w9WgXcQ", done=False)
    incomplete = make_video_dir(raw, "ch_001", "aQw4w9WgXcQ")
    (incomplete / "audio.wav").unlink()

    segments, issues = scan_raw(raw, tmp_path)
    assert not segments
    assert not issues


def test_scan_reports_zero_length_audio(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    make_video_dir(raw, "ch_001", "dQw4w9WgXcQ", seconds=0.0)
    segments, issues = scan_raw(raw, tmp_path)
    assert not segments
    assert len(issues) == 1


def test_scan_falls_back_to_directory_channel_id(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    make_video_dir(raw, "ch_001", "dQw4w9WgXcQ", meta_channel_id=None)
    segments, _ = scan_raw(raw, tmp_path)
    assert segments[0].channel_id == "ch_001"


def test_video_key_is_deterministic_hex() -> None:
    assert video_key("dQw4w9WgXcQ") == video_key("dQw4w9WgXcQ")
    assert re.fullmatch(r"[0-9a-f]{10}", video_key("dQw4w9WgXcQ"))
    assert video_key("dQw4w9WgXcQ") != video_key("aQw4w9WgXcQ")


def test_cli_scan_raw_then_stats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path))
    raw = tmp_path / "raw"
    make_video_dir(raw, "ch_001", "dQw4w9WgXcQ", seconds=2.0)

    result = runner.invoke(app, ["scan-raw"])
    assert result.exit_code == 0
    manifest = tmp_path / "manifests" / "raw.jsonl"
    assert manifest.is_file()

    stats_result = runner.invoke(app, ["stats", str(manifest)])
    assert stats_result.exit_code == 0
    assert "ch_001" in stats_result.stdout
    assert "total: 1 segment(s)" in stats_result.stdout
