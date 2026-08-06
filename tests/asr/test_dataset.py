from __future__ import annotations

import io
import json
import wave
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from uztts_asr.dataset import (
    ManifestSampleIterator,
    decode_bytes,
    load_rows,
    punct_target,
)


def wav_bytes(seconds: float = 1.0, rate: int = 16000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x10\x00" * int(seconds * rate))
    return buffer.getvalue()


def test_punct_target_keeps_sentence_punctuation() -> None:
    assert punct_target("Salom, Dunyo! Qalaysiz?") == "salom, dunyo! qalaysiz?"
    assert punct_target("Uch — to'rt; besh: (olti)") == "uch to'rt besh olti"
    assert punct_target("Ha .  Yo'q !") == "ha. yo'q!"


def test_load_rows_reads_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(
        json.dumps({"text": "a"}) + "\n\n" + json.dumps({"text": "b"}) + "\n",
        encoding="utf-8",
    )
    assert [row["text"] for row in load_rows(manifest)] == ["a", "b"]


def test_decode_bytes_returns_16k_mono() -> None:
    audio = decode_bytes(wav_bytes(2.0))
    assert audio.shape == (32000,)


def make_corpus(tmp_path: Path) -> tuple[Path, Path, list[dict[str, object]]]:
    corpora = tmp_path / "corpora"
    parquet = corpora / "usc" / "data" / "train-00000.parquet"
    parquet.parent.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "audio": [
                    {"bytes": wav_bytes(1.0), "path": "0.wav"},
                    {"bytes": wav_bytes(2.0), "path": "1.wav"},
                ]
            }
        ),
        parquet,
    )
    asr_root = tmp_path / "asr"
    (asr_root / "fleurs_audio").mkdir(parents=True)
    (asr_root / "fleurs_audio" / "f.wav").write_bytes(wav_bytes(1.5))
    rows: list[dict[str, object]] = [
        {
            "parquet": "usc/data/train-00000.parquet",
            "row": 0,
            "text": "bir",
            "text_raw": "Bir.",
            "duration": 1.0,
            "source": "usc",
        },
        {
            "parquet": "usc/data/train-00000.parquet",
            "row": 1,
            "text": "ikki",
            "text_raw": "Ikki!",
            "duration": 2.0,
            "source": "usc",
        },
        {
            "audio_filepath": "fleurs_audio/f.wav",
            "text": "uch",
            "text_raw": "Uch?",
            "duration": 1.5,
            "source": "fleurs",
        },
    ]
    return corpora, asr_root, rows


def test_iterator_yields_all_samples(tmp_path: Path) -> None:
    corpora, asr_root, rows = make_corpus(tmp_path)
    iterator = ManifestSampleIterator(rows, asr_root, corpora, buffer_size=2)
    samples = list(iterator)
    assert sorted(sample.text for sample in samples) == ["bir", "ikki", "uch"]
    assert all(sample.audio.ndim == 1 for sample in samples)
    by_text = {sample.text: sample for sample in samples}
    assert by_text["ikki"].audio.shape == (32000,)


def test_iterator_punctuated_uses_text_raw(tmp_path: Path) -> None:
    corpora, asr_root, rows = make_corpus(tmp_path)
    iterator = ManifestSampleIterator(rows, asr_root, corpora, punctuated=True)
    assert sorted(sample.text for sample in iterator) == ["bir.", "ikki!", "uch?"]


def test_iterator_epoch_changes_order(tmp_path: Path) -> None:
    corpora, asr_root, rows = make_corpus(tmp_path)
    rows = rows * 5
    iterator = ManifestSampleIterator(rows, asr_root, corpora, buffer_size=4)
    first = [sample.duration for sample in iterator]
    again = [sample.duration for sample in iterator]
    assert first == again
    iterator.set_epoch(1)
    shuffled = [sample.duration for sample in iterator]
    assert sorted(shuffled) == sorted(first)
    assert shuffled != first


def test_iterator_skips_undecodable_audio(tmp_path: Path) -> None:
    corpora, asr_root, rows = make_corpus(tmp_path)
    rows.append(
        {
            "audio_filepath": "fleurs_audio/yoq.wav",
            "text": "toʻrt",
            "duration": 1.0,
            "source": "fleurs",
        }
    )
    samples = list(ManifestSampleIterator(rows, asr_root, corpora))
    assert len(samples) == 3


def test_iterator_reads_across_row_groups(tmp_path: Path) -> None:
    corpora = tmp_path / "corpora"
    parquet = corpora / "usc" / "data" / "train-00000.parquet"
    parquet.parent.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "audio": [
                    {"bytes": wav_bytes(1.0), "path": f"{index}.wav"}
                    for index in range(4)
                ]
            }
        ),
        parquet,
        row_group_size=1,
    )
    rows: list[dict[str, object]] = [
        {
            "parquet": "usc/data/train-00000.parquet",
            "row": index,
            "text": f"matn {index}",
            "duration": 1.0,
            "source": "usc",
        }
        for index in range(4)
    ]
    samples = list(ManifestSampleIterator(rows, tmp_path / "asr", corpora))
    assert sorted(sample.text for sample in samples) == [
        f"matn {index}" for index in range(4)
    ]


def test_sample_rate_is_16k() -> None:
    from uztts_asr.dataset import SAMPLE_RATE

    assert SAMPLE_RATE == 16000


@pytest.mark.parametrize("punctuated", [False, True])
def test_iterator_is_reiterable(tmp_path: Path, punctuated: bool) -> None:
    corpora, asr_root, rows = make_corpus(tmp_path)
    iterator = ManifestSampleIterator(rows, asr_root, corpora, punctuated=punctuated)
    assert len(list(iterator)) == len(list(iterator)) == 3
