from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest
from typer.testing import CliRunner

import uztts_data.transcribe as transcribe_module
from uztts_data.manifest import read_manifest, write_manifest
from uztts_data.paths import DATA_ROOT_ENV
from uztts_data.schema import License, Segment
from uztts_data.transcribe import (
    Transcription,
    app,
    apply_transcription,
    transcribed_ids,
)

runner = CliRunner()


def make_segment(index: int, root: Path | None = None) -> Segment:
    audio = Path(f"interim/segments/ch_001/abc/{index:04d}.wav")
    if root is not None:
        target = root / audio
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"RIFF")
    return Segment(
        id=f"ch_001_abc_{index:04d}",
        audio_path=audio,
        speaker_id="ch_001_c0",
        channel_id="ch_001",
        duration=5.0,
        sample_rate=24000,
        source="youtube",
        license=License.WEB_SCRAPED,
    )


@dataclass
class FakeTranscriber:
    result: Transcription = field(
        default_factory=lambda: Transcription(
            text="salom dunyo",
            avg_logprob=-0.31,
            compression_ratio=1.4,
            lang_prob=0.95,
        )
    )
    calls: list[Path] = field(default_factory=list)
    fail_on: str | None = None

    def preload(self) -> None:
        pass

    def transcribe(self, audio: Path) -> Transcription:
        self.calls.append(audio)
        if self.fail_on and self.fail_on in str(audio):
            raise RuntimeError("decoder blew up")
        return self.result


def test_apply_transcription_fills_asr_fields() -> None:
    segment = make_segment(0)
    result = Transcription(
        text=" salom ", avg_logprob=-0.4, compression_ratio=1.2, lang_prob=0.9
    )
    updated = apply_transcription(segment, result)
    assert updated.text == "salom"
    assert updated.asr_avg_logprob == -0.4
    assert updated.asr_compression_ratio == 1.2
    assert updated.lang_prob == 0.9
    assert segment.text is None


def test_apply_transcription_keeps_empty_text_as_none() -> None:
    segment = make_segment(0)
    result = Transcription(
        text="", avg_logprob=-10.0, compression_ratio=1.0, lang_prob=0.0
    )
    updated = apply_transcription(segment, result)
    assert updated.text is None
    assert updated.asr_avg_logprob == -10.0


def test_apply_transcription_clamps_out_of_range_diagnostics() -> None:
    segment = make_segment(0)
    result = Transcription(
        text="x", avg_logprob=-1.0, compression_ratio=0.0, lang_prob=1.5
    )
    updated = apply_transcription(segment, result)
    assert updated.asr_compression_ratio == 0.01
    assert updated.lang_prob == 1.0


def test_transcribed_ids_reads_existing_manifest(tmp_path: Path) -> None:
    out = tmp_path / "transcripts.jsonl"
    assert transcribed_ids(out) == set()
    segment = apply_transcription(
        make_segment(0),
        Transcription(
            text="salom", avg_logprob=-0.3, compression_ratio=1.2, lang_prob=0.9
        ),
    )
    write_manifest(out, [segment])
    assert transcribed_ids(out) == {segment.id}


def test_cli_transcribe_resumes_and_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path))
    segments = [make_segment(index, tmp_path) for index in range(3)]
    write_manifest(tmp_path / "manifests" / "segments.jsonl", segments)
    out = tmp_path / "manifests" / "transcripts.jsonl"
    write_manifest(
        out,
        [
            apply_transcription(
                segments[0],
                Transcription(
                    text="bor", avg_logprob=-0.2, compression_ratio=1.1, lang_prob=0.9
                ),
            )
        ],
    )
    fake = FakeTranscriber()
    monkeypatch.setattr(
        transcribe_module,
        "FasterWhisperTranscriber",
        lambda **kwargs: fake,
    )

    result = runner.invoke(app, [])

    assert result.exit_code == 0
    assert "transcribed=2 skipped=1 failed=0" in result.stdout
    assert len(fake.calls) == 2
    rows = list(read_manifest(out))
    assert len(rows) == 3
    assert [row.id for row in rows] == sorted(row.id for row in rows)
    assert rows[1].text == "salom dunyo"


def test_cli_transcribe_logs_failures_and_exits_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path))
    segments = [make_segment(index, tmp_path) for index in range(2)]
    write_manifest(tmp_path / "manifests" / "segments.jsonl", segments)
    fake = FakeTranscriber(fail_on="0000")
    monkeypatch.setattr(
        transcribe_module,
        "FasterWhisperTranscriber",
        lambda **kwargs: fake,
    )

    result = runner.invoke(app, [])

    assert result.exit_code == 1
    assert "transcribed=1" in result.stdout
    assert "failed=1" in result.stdout


def test_cli_transcribe_requires_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path))
    result = runner.invoke(app, [])
    assert result.exit_code == 2


def test_cli_transcribe_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path))
    segments = [make_segment(index, tmp_path) for index in range(5)]
    write_manifest(tmp_path / "manifests" / "segments.jsonl", segments)
    fake = FakeTranscriber()
    monkeypatch.setattr(
        transcribe_module,
        "FasterWhisperTranscriber",
        lambda **kwargs: fake,
    )
    result = runner.invoke(app, ["--limit", "2"])
    assert result.exit_code == 0
    assert len(fake.calls) == 2
