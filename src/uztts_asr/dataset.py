from __future__ import annotations

import importlib
import io
import json
import random
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from uztts_asr.prepare import PARQUET_SOURCES

if TYPE_CHECKING:
    import numpy
    from numpy.typing import NDArray

SAMPLE_RATE = 16000

_PUNCT_KEEP = ".,?!"
_PUNCT_DROP_RE = re.compile(rf"[^\w\s{re.escape(_PUNCT_KEEP)}ʻʼ'-]")
_PUNCT_SPACE_RE = re.compile(r"\s+")

_AUDIO_COLUMNS = {spec.name: spec.audio_column for spec in PARQUET_SOURCES}


def punct_target(text_raw: str) -> str:
    lowered = _PUNCT_DROP_RE.sub(" ", text_raw.lower())
    spaced = _PUNCT_SPACE_RE.sub(" ", lowered)
    return re.sub(rf" ?([{re.escape(_PUNCT_KEEP)}])", r"\1", spaced).strip()


@dataclass(frozen=True, slots=True)
class Sample:
    audio: NDArray[numpy.float32]
    text: str
    duration: float
    source: str


def preload_audio_decoder() -> None:
    importlib.import_module("faster_whisper.audio")


def load_rows(manifest: Path) -> list[dict[str, Any]]:
    with manifest.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def decode_bytes(payload: bytes) -> NDArray[numpy.float32]:
    from typing import cast

    from faster_whisper.audio import decode_audio

    return cast(
        "NDArray[numpy.float32]",
        decode_audio(io.BytesIO(payload), sampling_rate=SAMPLE_RATE),
    )


class ManifestSampleIterator:
    def __init__(
        self,
        rows: list[dict[str, Any]],
        asr_root: Path,
        corpora_root: Path,
        punctuated: bool = False,
        seed: int = 0,
        buffer_size: int = 1024,
    ) -> None:
        self._rows = rows
        self._asr_root = asr_root
        self._corpora_root = corpora_root
        self._punctuated = punctuated
        self._seed = seed
        self._buffer_size = buffer_size
        self._epoch = 0
        self._cached_file: str | None = None
        self._reader: Any = None
        self._cached_column: Any = None
        self._group_start = 0
        self._group_end = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __iter__(self) -> Iterator[Sample]:
        rng = random.Random(self._seed + self._epoch + 1)
        buffer: list[Sample] = []
        for sample in self._sequential():
            buffer.append(sample)
            if len(buffer) >= self._buffer_size:
                index = rng.randrange(len(buffer))
                buffer[index], buffer[-1] = buffer[-1], buffer[index]
                yield buffer.pop()
        rng.shuffle(buffer)
        yield from buffer

    def _sequential(self) -> Iterator[Sample]:
        for row in self._shuffled():
            audio = self._audio_of(row)
            if audio is None:
                continue
            text = (
                punct_target(str(row.get("text_raw") or row["text"]))
                if self._punctuated
                else str(row["text"])
            )
            yield Sample(
                audio=audio,
                text=text,
                duration=float(row["duration"]),
                source=str(row["source"]),
            )

    def _shuffled(self) -> list[dict[str, Any]]:
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in self._rows:
            key = str(row.get("parquet") or row.get("audio_filepath"))
            groups.setdefault(key, []).append(row)
        rng = random.Random(self._seed + self._epoch)
        keys = sorted(groups)
        rng.shuffle(keys)
        ordered: list[dict[str, Any]] = []
        for key in keys:
            rows = sorted(groups[key], key=lambda row: int(row.get("row", 0)))
            ordered.extend(rows)
        return ordered

    def _audio_of(self, row: dict[str, Any]) -> NDArray[numpy.float32] | None:
        if "audio_filepath" in row:
            path = self._asr_root / str(row["audio_filepath"])
            try:
                return decode_bytes(path.read_bytes())
            except Exception:
                return None
        payload = self._parquet_payload(row)
        if payload is None:
            return None
        try:
            return decode_bytes(payload)
        except Exception:
            return None

    def _parquet_payload(self, row: dict[str, Any]) -> bytes | None:
        import pyarrow.parquet as pq

        relative = str(row["parquet"])
        index = int(row["row"])
        if relative != self._cached_file:
            self._reader = pq.ParquetFile(self._corpora_root / relative)
            self._cached_file = relative
            self._cached_column = None
            self._group_start = 0
            self._group_end = 0
        if self._cached_column is None or not (
            self._group_start <= index < self._group_end
        ):
            self._load_row_group(str(row["source"]), index)
        if self._cached_column is None:
            return None
        cell = self._cached_column[index - self._group_start].as_py()
        payload = cell.get("bytes") if isinstance(cell, dict) else cell
        return payload if isinstance(payload, bytes) else None

    def _load_row_group(self, source: str, index: int) -> None:
        start = 0
        metadata = self._reader.metadata
        for group in range(metadata.num_row_groups):
            count = metadata.row_group(group).num_rows
            if index < start + count:
                column = _AUDIO_COLUMNS[source]
                table = self._reader.read_row_group(group, columns=[column])
                self._cached_column = table.column(column)
                self._group_start = start
                self._group_end = start + count
                return
            start += count
        self._cached_column = None
