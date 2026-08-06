from __future__ import annotations

import io
import json
import wave
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from uztts_asr.evaluate import (
    GigaAmTranscriber,
    TurboTranscriber,
    make_transcriber,
    resolve_audio,
    score,
    select_rows,
)


def wav_bytes(seconds: float = 2.0, rate: int = 16000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x10\x00" * int(seconds * rate))
    return buffer.getvalue()


def make_parquet(path: Path, rows: list[tuple[bytes | None, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "audio": [
                {"bytes": payload, "path": f"{index}.wav"} if payload else None
                for index, (payload, _) in enumerate(rows)
            ],
            "sentence": [text for _, text in rows],
        }
    )
    pq.write_table(table, path)


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_select_rows_filters_source_and_limit(tmp_path: Path) -> None:
    manifest = tmp_path / "test_manifest.jsonl"
    write_manifest(
        manifest,
        [
            {"source": "fleurs", "text": "a"},
            {"source": "usc", "text": "b"},
            {"source": "fleurs", "text": "c"},
        ],
    )
    rows = select_rows(manifest, {"fleurs"})
    assert [row["text"] for row in rows] == ["a", "c"]
    assert len(select_rows(manifest, set(), limit=2)) == 2


def test_resolve_audio_handles_files_and_parquet(tmp_path: Path) -> None:
    asr_root = tmp_path / "asr"
    (asr_root / "fleurs_audio").mkdir(parents=True)
    wav = asr_root / "fleurs_audio" / "s1.wav"
    wav.write_bytes(wav_bytes(1.0))
    corpora = tmp_path / "corpora"
    make_parquet(
        corpora / "usc" / "data" / "train-00000.parquet",
        [(wav_bytes(2.0), "salom dunyo")],
    )
    rows: list[dict[str, object]] = [
        {"source": "fleurs", "audio_filepath": "fleurs_audio/s1.wav"},
        {"source": "usc", "parquet": "usc/data/train-00000.parquet", "row": 0},
    ]

    samples = list(resolve_audio(rows, asr_root, corpora, tmp_path / "work"))
    assert len(samples) == 2
    assert samples[0].audio_path == wav
    assert samples[1].audio_path.is_file()
    assert samples[1].audio_path.suffix == ".wav"


def test_score_computes_wer_and_cer() -> None:
    pairs = [
        ("salom dunyo", "salom dunyo", 2.0),
        ("yaxshi kun boʻlsin", "yaxshi tun boʻlsin", 3.0),
    ]
    summary = score("turbo", pairs)
    assert summary.samples == 2
    assert summary.hours == pytest.approx(5.0 / 3600)
    assert summary.wer == pytest.approx(1 / 5)
    assert 0 < summary.cer < summary.wer


def test_quietest_cut_finds_silence_gap() -> None:
    import numpy as np

    from uztts_asr.evaluate import quietest_cut

    rate = 16000
    loud = np.sin(np.linspace(0, 2000, rate * 10, dtype=np.float32))
    audio = np.concatenate(
        [loud[: rate * 6], np.zeros(rate, dtype=np.float32), loud[: rate * 3]]
    )
    cut = quietest_cut(audio, rate)
    assert rate * 6 <= cut <= rate * 7


def test_make_transcriber_picks_backend() -> None:
    assert isinstance(make_transcriber("gigaam"), GigaAmTranscriber)
    assert isinstance(make_transcriber("gigaam-large"), GigaAmTranscriber)
    assert isinstance(make_transcriber("turbo"), TurboTranscriber)
