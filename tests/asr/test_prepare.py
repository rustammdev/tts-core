from __future__ import annotations

import io
import json
import struct
import tarfile
import wave
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from uztts_asr.prepare import (
    SourceSpec,
    _wav_duration,
    app,
    clean_raw_text,
    merge_manifests,
    normalize_text,
    prepare_fleurs,
    prepare_parquet_source,
    probe_duration,
    reject_reason,
    split_of,
    to_okina,
)

runner = CliRunner()


def wav_bytes(seconds: float = 2.0, rate: int = 16000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x10\x00" * int(seconds * rate))
    return buffer.getvalue()


def float_wav_bytes(seconds: float = 2.0, rate: int = 16000) -> bytes:
    frames = int(seconds * rate)
    data = struct.pack(f"<{frames}f", *([0.01] * frames))
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + len(data),
        b"WAVE",
        b"fmt ",
        16,
        3,
        1,
        rate,
        rate * 4,
        4,
        32,
        b"data",
        len(data),
    )
    return header + data


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


def test_to_okina_normalizes_apostrophe_variants() -> None:
    assert to_okina("o'zbek g’oz Gʻalaba") == "oʻzbek gʻoz Gʻalaba"
    assert to_okina("ta'lim va'da") == "taʼlim vaʼda"


def test_normalize_text_lowercases_and_strips_punctuation() -> None:
    assert normalize_text("Salom, Dunyo! O'zbekiston — vatanim.") == (
        "salom dunyo oʻzbekiston vatanim"
    )


def test_clean_raw_text_keeps_punctuation() -> None:
    assert clean_raw_text("Salom,  Dunyo!") == "Salom, Dunyo!"


@pytest.mark.parametrize(
    ("raw", "duration", "expected"),
    [
        ("salom dunyo", 2.0, None),
        ("", 2.0, "empty"),
        ("500 soʻm", 2.0, "digits"),
        ("привет salom", 2.0, "charset"),
        ("salom dunyo", 0.2, "duration"),
        ("salom dunyo azizlar yaxshimisiz hammangiz", 1.0, "char_rate"),
        ("salom", 10.0, "word_rate"),
    ],
)
def test_reject_reason(raw: str, duration: float, expected: str | None) -> None:
    assert reject_reason(raw, normalize_text(raw), duration) == expected


def test_split_of_is_deterministic_and_mostly_train() -> None:
    keys = [f"speaker_{index}" for index in range(500)]
    splits = [split_of(key) for key in keys]
    assert splits == [split_of(key) for key in keys]
    assert splits.count("train") > 400
    assert set(splits) == {"train", "val", "test"}


def test_wav_duration_handles_ieee_float_wav(tmp_path: Path) -> None:
    pcm = tmp_path / "pcm.wav"
    pcm.write_bytes(wav_bytes(2.0))
    assert _wav_duration(pcm) == pytest.approx(2.0, abs=0.1)

    ieee = tmp_path / "float.wav"
    ieee.write_bytes(float_wav_bytes(3.0))
    assert _wav_duration(ieee) == pytest.approx(3.0, abs=0.1)

    assert _wav_duration(tmp_path / "missing.wav") is None


def test_probe_duration_reads_wav_header() -> None:
    assert probe_duration(wav_bytes(3.0)) == pytest.approx(3.0, abs=0.1)
    assert probe_duration(b"not audio") is None


def test_prepare_parquet_source_writes_shards(tmp_path: Path) -> None:
    corpora = tmp_path / "corpora"
    make_parquet(
        corpora / "usc" / "data" / "train-00000.parquet",
        [
            (wav_bytes(2.0), "Salom dunyo azizlar"),
            (wav_bytes(2.0), "500 soʻm"),
            (None, "audio yoʻq"),
        ],
    )
    out = tmp_path / "asr"
    spec = SourceSpec("usc", "usc", "audio", "sentence")

    stats = prepare_parquet_source(spec, corpora, out)
    assert stats.kept == 1
    assert stats.rejected == {"digits": 1, "missing_audio": 1}

    shard = out / "shards" / "usc" / "train-00000.jsonl"
    rows = [json.loads(line) for line in shard.open(encoding="utf-8")]
    assert len(rows) == 1
    assert rows[0]["parquet"] == "usc/data/train-00000.parquet"
    assert rows[0]["row"] == 0
    assert rows[0]["text"] == "salom dunyo azizlar"
    assert rows[0]["split"] in {"train", "val", "test"}

    rerun = prepare_parquet_source(spec, corpora, out)
    assert rerun.kept == 1


def test_prepare_parquet_source_skips_excluded_rows(tmp_path: Path) -> None:
    corpora = tmp_path / "corpora"
    make_parquet(
        corpora / "usc" / "data" / "train-00000.parquet",
        [
            (wav_bytes(2.0), "Salom dunyo azizlar"),
            (wav_bytes(2.0), "Etalonga olingan satr bu"),
        ],
    )
    (corpora / "usc" / "exclude_rows.json").write_text(
        json.dumps({"data/train-00000.parquet": [1]}), encoding="utf-8"
    )
    spec = SourceSpec("usc", "usc", "audio", "sentence")

    stats = prepare_parquet_source(spec, corpora, tmp_path / "asr")
    assert stats.kept == 1
    assert stats.rejected == {"excluded": 1}

    shard = tmp_path / "asr" / "shards" / "usc" / "train-00000.jsonl"
    rows = [json.loads(line) for line in shard.open(encoding="utf-8")]
    assert [row["row"] for row in rows] == [0]


def test_prepare_parquet_source_respects_file_glob(tmp_path: Path) -> None:
    corpora = tmp_path / "corpora"
    make_parquet(
        corpora / "cv" / "data" / "validated-00000.parquet",
        [(wav_bytes(2.0), "Salom dunyo azizlar")],
    )
    make_parquet(
        corpora / "cv" / "data" / "train-00000.parquet",
        [(wav_bytes(2.0), "Takror satr bu yerda")],
    )
    spec = SourceSpec(
        "cv", "cv", "audio", "sentence", file_glob="data/validated-*.parquet"
    )

    stats = prepare_parquet_source(spec, corpora, tmp_path / "asr")
    assert stats.kept == 1
    shards = list((tmp_path / "asr" / "shards" / "cv").glob("*.jsonl"))
    assert [shard.name for shard in shards] == ["validated-00000.jsonl"]


def test_prepare_fleurs_extracts_and_maps_splits(tmp_path: Path) -> None:
    base = tmp_path / "corpora" / "fleurs_uz" / "data" / "uz_uz"
    (base / "audio").mkdir(parents=True)
    inner = io.BytesIO()
    with tarfile.open(fileobj=inner, mode="w:gz") as archive:
        payload = wav_bytes(2.0)
        info = tarfile.TarInfo("dev/sample1.wav")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    (base / "audio" / "dev.tar.gz").write_bytes(inner.getvalue())
    (base / "dev.tsv").write_text(
        "1\tsample1.wav\tSalom dunyo azizlar\tsalom dunyo azizlar\n",
        encoding="utf-8",
    )

    out = tmp_path / "asr"
    stats = prepare_fleurs(tmp_path / "corpora", out)
    assert stats.kept == 1

    shard = out / "shards" / "fleurs" / "dev.jsonl"
    row = json.loads(shard.read_text(encoding="utf-8"))
    assert row["split"] == "val"
    assert row["audio_filepath"].endswith("sample1.wav")
    assert (out / row["audio_filepath"]).is_file()


def test_merge_manifests_groups_by_split(tmp_path: Path) -> None:
    shard_dir = tmp_path / "shards" / "x"
    shard_dir.mkdir(parents=True)
    lines = [
        {"duration": 1.0, "text": "a", "split": "train"},
        {"duration": 1.0, "text": "b", "split": "val"},
        {"duration": 1.0, "text": "c", "split": "train"},
    ]
    (shard_dir / "s.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in lines), encoding="utf-8"
    )
    counts = merge_manifests(tmp_path)
    assert counts == {"train": 2, "val": 1, "test": 0}
    assert (tmp_path / "train_manifest.jsonl").is_file()


def test_cli_prepare_with_synthetic_source(tmp_path: Path) -> None:
    corpora = tmp_path / "corpora"
    make_parquet(
        corpora / "usc" / "data" / "train-00000.parquet",
        [(wav_bytes(2.0), "Salom dunyo azizlar")],
    )
    out = tmp_path / "asr"
    result = runner.invoke(
        app,
        [
            "--corpora-root",
            str(corpora),
            "--out-root",
            str(out),
            "--only",
            "usc",
        ],
    )
    assert result.exit_code == 0
    assert "usc: kept=1" in result.stdout
    assert (out / "train_manifest.jsonl").is_file() or (
        out / "val_manifest.jsonl"
    ).is_file()


def test_cli_prepare_rejects_unknown_source(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "--corpora-root",
            str(tmp_path),
            "--out-root",
            str(tmp_path / "o"),
            "--only",
            "nomalum",
        ],
    )
    assert result.exit_code == 2
