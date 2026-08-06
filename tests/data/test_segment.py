from __future__ import annotations

import wave
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from typer.testing import CliRunner

import uztts_data.segment as segment_module
from uztts_data.ingest import DONE_MARKER
from uztts_data.manifest import read_manifest, write_manifest
from uztts_data.paths import DATA_ROOT_ENV
from uztts_data.schema import License, Segment
from uztts_data.segment import (
    Span,
    app,
    clamp_spans,
    cut_wav,
    scan_segments,
    segment_videos,
    video_out_dir,
)

runner = CliRunner()


def write_wav(path: Path, seconds: float, rate: int = 24000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x01\x02" * int(seconds * rate))


def make_video(root: Path, channel_id: str, key: str, seconds: float) -> Segment:
    audio = root / "raw" / channel_id / key / "audio.wav"
    write_wav(audio, seconds)
    return Segment(
        id=f"{channel_id}_{key}",
        audio_path=audio.relative_to(root),
        speaker_id=f"{channel_id}_c0",
        channel_id=channel_id,
        duration=seconds,
        sample_rate=24000,
        source="youtube",
        license=License.WEB_SCRAPED,
    )


@dataclass
class FakeDetector:
    spans: Sequence[Span] = ((Span(1.0, 5.0)),)
    calls: list[Path] = field(default_factory=list)

    def speech_spans(self, audio: Path) -> Sequence[Span]:
        self.calls.append(audio)
        return list(self.spans)


def test_clamp_spans_drops_short_and_splits_long() -> None:
    spans = [Span(0.0, 1.0), Span(10.0, 15.0), Span(20.0, 65.0)]
    kept, dropped = clamp_spans(spans, min_seconds=2.0, max_seconds=20.0)
    assert dropped == pytest.approx(1.0)
    assert kept[0] == Span(10.0, 15.0)
    assert len(kept) == 4
    lengths = [span.seconds for span in kept[1:]]
    assert all(length == pytest.approx(15.0) for length in lengths)
    assert kept[1].start == pytest.approx(20.0)
    assert kept[-1].end == pytest.approx(65.0)


def test_cut_wav_writes_expected_durations(tmp_path: Path) -> None:
    source = tmp_path / "audio.wav"
    write_wav(source, seconds=10.0)
    written = cut_wav(source, [Span(1.0, 3.5), Span(4.0, 9.0)], tmp_path / "out")
    assert [path.name for path in written] == ["0000.wav", "0001.wav"]
    with wave.open(str(written[0]), "rb") as handle:
        assert handle.getframerate() == 24000
        assert handle.getnframes() == pytest.approx(2.5 * 24000)


def test_cut_wav_clips_span_past_end_of_file(tmp_path: Path) -> None:
    source = tmp_path / "audio.wav"
    write_wav(source, seconds=5.0)
    written = cut_wav(source, [Span(4.0, 30.0), Span(10.0, 12.0)], tmp_path / "out")
    assert len(written) == 1
    with wave.open(str(written[0]), "rb") as handle:
        assert handle.getnframes() == pytest.approx(1.0 * 24000)


def test_video_out_dir_derives_channel_and_key(tmp_path: Path) -> None:
    video = make_video(tmp_path, "ch_sokin_qalb", "ab12cd34ef", 6.0)
    assert video_out_dir(tmp_path / "seg", video) == (
        tmp_path / "seg" / "ch_sokin_qalb" / "ab12cd34ef"
    )


def test_segment_videos_writes_segments_and_marker(tmp_path: Path) -> None:
    video = make_video(tmp_path, "ch_001", "ab12cd34ef", 10.0)
    out_root = tmp_path / "interim" / "segments"
    detector = FakeDetector(spans=[Span(0.5, 4.5), Span(5.0, 6.0)])

    outcomes = segment_videos([video], tmp_path, out_root, detector)
    assert [outcome.status for outcome in outcomes] == ["segmented"]

    video_dir = out_root / "ch_001" / "ab12cd34ef"
    assert (video_dir / DONE_MARKER).is_file()
    assert sorted(path.name for path in video_dir.glob("*.wav")) == ["0000.wav"]

    segments, issues = scan_segments(out_root, tmp_path)
    assert not issues
    assert len(segments) == 1
    first = segments[0]
    assert first.id == "ch_001_ab12cd34ef_0000"
    assert first.speaker_id == "ch_001_c0"
    assert first.channel_id == "ch_001"
    assert first.duration == pytest.approx(4.0)
    assert first.audio_path == Path("interim/segments/ch_001/ab12cd34ef/0000.wav")


def test_segment_videos_skips_done_video(tmp_path: Path) -> None:
    video = make_video(tmp_path, "ch_001", "ab12cd34ef", 10.0)
    out_root = tmp_path / "interim" / "segments"
    detector = FakeDetector(spans=[Span(0.0, 5.0)])
    segment_videos([video], tmp_path, out_root, detector)
    outcomes = segment_videos([video], tmp_path, out_root, detector)
    assert [outcome.status for outcome in outcomes] == ["skipped"]
    assert len(detector.calls) == 1


def test_segment_videos_reports_missing_audio(tmp_path: Path) -> None:
    video = make_video(tmp_path, "ch_001", "ab12cd34ef", 10.0)
    (tmp_path / video.audio_path).unlink()
    outcomes = segment_videos([video], tmp_path, tmp_path / "seg", FakeDetector())
    assert outcomes[0].status == "failed"
    assert outcomes[0].error is not None and "missing" in outcomes[0].error


def test_segment_videos_removes_stale_wavs_before_redo(tmp_path: Path) -> None:
    video = make_video(tmp_path, "ch_001", "ab12cd34ef", 10.0)
    out_root = tmp_path / "seg"
    video_dir = out_root / "ch_001" / "ab12cd34ef"
    video_dir.mkdir(parents=True)
    write_wav(video_dir / "0007.wav", seconds=1.0)

    segment_videos([video], tmp_path, out_root, FakeDetector(spans=[Span(0.0, 5.0)]))
    assert sorted(path.name for path in video_dir.glob("*.wav")) == ["0000.wav"]


def test_cli_segment_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path))
    video = make_video(tmp_path, "ch_001", "ab12cd34ef", 10.0)
    write_manifest(tmp_path / "manifests" / "raw.jsonl", [video])
    monkeypatch.setattr(
        segment_module,
        "SileroDetector",
        lambda max_seconds: FakeDetector(spans=[Span(0.0, 4.0), Span(5.0, 8.0)]),
    )

    result = runner.invoke(app, [])

    assert result.exit_code == 0
    assert "segmented=1" in result.stdout
    manifest = tmp_path / "manifests" / "segments.jsonl"
    rows = list(read_manifest(manifest))
    assert len(rows) == 2
    assert "raw 0.00 h" in result.stdout or "speech" in result.stdout


def test_cli_segment_requires_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path))
    result = runner.invoke(app, [])
    assert result.exit_code == 2
