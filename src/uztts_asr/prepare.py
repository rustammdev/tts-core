from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import tarfile
import wave
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import typer

MIN_SECONDS = 0.5
MAX_SECONDS = 30.0
MAX_CHARS_PER_SECOND = 18.0
MIN_WORDS_PER_SECOND = 0.3
MAX_WORDS_PER_SECOND = 2.67
SAMPLE_RATE = 16000

OKINA = "ʻ"
TUTUQ = "ʼ"
_APOSTROPHES = "'`´’ʼʻ‘"
_OKINA_RE = re.compile(f"([ogOG])[{_APOSTROPHES}{OKINA}]")
_TUTUQ_RE = re.compile(f"[{_APOSTROPHES}]")
_PUNCT_RE = re.compile(rf"[^a-z{OKINA}{TUTUQ} ]")
_ALLOWED_RAW_RE = re.compile(rf"^[a-zA-Z{OKINA}{TUTUQ} .,!?—–:;\"'()…-]*$")
_SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class SourceSpec:
    name: str
    directory: str
    audio_column: str
    text_column: str
    speaker_column: str | None = None
    duration_column: str | None = None
    file_glob: str = "data/*.parquet"


PARQUET_SOURCES = (
    SourceSpec("usc", "usc", "audio", "sentence"),
    SourceSpec(
        "common_voice",
        "common_voice_uz",
        "audio",
        "sentence",
        "client_id",
        file_glob="data/validated-*.parquet",
    ),
    SourceSpec(
        "uzbekvoice", "uzbekvoice_filtered", "path", "sentence", "client_id", "duration"
    ),
    SourceSpec("yt_news", "yt_news", "audio", "text"),
    SourceSpec("yt_it", "yt_it", "audio", "text"),
    SourceSpec("yt_podcasts", "yt_podcasts", "audio", "text"),
    SourceSpec("yt_gemini", "yt_gemini", "audio", "transcription"),
)
FLEURS_NAME = "fleurs"
ALL_SOURCE_NAMES = (*(spec.name for spec in PARQUET_SOURCES), FLEURS_NAME)


def to_okina(text: str) -> str:
    marked = _OKINA_RE.sub("\\1\x00", text)
    return _TUTUQ_RE.sub(TUTUQ, marked).replace("\x00", OKINA)


def normalize_text(text: str) -> str:
    lowered = to_okina(text).lower()
    stripped = _PUNCT_RE.sub(" ", lowered)
    return _SPACE_RE.sub(" ", stripped).strip()


def clean_raw_text(text: str) -> str:
    return _SPACE_RE.sub(" ", to_okina(text)).strip()


def reject_reason(raw: str, normalized: str, duration: float) -> str | None:
    if not normalized:
        return "empty"
    if any(ch.isdigit() for ch in raw):
        return "digits"
    if not _ALLOWED_RAW_RE.match(to_okina(raw)):
        return "charset"
    if not MIN_SECONDS <= duration <= MAX_SECONDS:
        return "duration"
    if len(normalized) / duration > MAX_CHARS_PER_SECOND:
        return "char_rate"
    words_per_second = len(normalized.split()) / duration
    if not MIN_WORDS_PER_SECOND <= words_per_second <= MAX_WORDS_PER_SECOND:
        return "word_rate"
    return None


def split_of(key: str) -> str:
    bucket = int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) % 100
    if bucket < 96:
        return "train"
    return "val" if bucket < 98 else "test"


def probe_duration(payload: bytes) -> float | None:
    import av

    try:
        opened: Any = av.open(io.BytesIO(payload), metadata_errors="ignore")
        with opened as container:
            if container.duration:
                return float(container.duration) / 1_000_000
            streams = container.streams.audio
            if streams and streams[0].duration and streams[0].time_base:
                return float(streams[0].duration * streams[0].time_base)
    except Exception:
        return None
    return None


def decode_duration(payload: bytes) -> float | None:
    from faster_whisper.audio import decode_audio

    try:
        samples = decode_audio(io.BytesIO(payload), sampling_rate=SAMPLE_RATE)
    except Exception:
        return None
    return len(samples) / SAMPLE_RATE


@dataclass(slots=True)
class SourceStats:
    kept: int = 0
    seconds: float = 0.0
    rejected: dict[str, int] | None = None

    def reject(self, reason: str) -> None:
        if self.rejected is None:
            self.rejected = {}
        self.rejected[reason] = self.rejected.get(reason, 0) + 1


def _sample_row(
    source: str,
    raw_text: str,
    duration: float | None,
    split_key: str,
    stats: SourceStats,
    location: dict[str, Any],
    forced_split: str | None = None,
) -> str | None:
    normalized = normalize_text(raw_text)
    if duration is None:
        stats.reject("no_duration")
        return None
    reason = reject_reason(raw_text, normalized, duration)
    if reason is not None:
        stats.reject(reason)
        return None
    stats.kept += 1
    stats.seconds += duration
    row = {
        **location,
        "duration": round(duration, 3),
        "text": normalized,
        "text_raw": clean_raw_text(raw_text),
        "source": source,
        "split": forced_split or split_of(split_key),
    }
    return json.dumps(row, ensure_ascii=False) + "\n"


def load_excluded_rows(source_dir: Path) -> dict[str, frozenset[int]]:
    target = source_dir / "exclude_rows.json"
    if not target.is_file():
        return {}
    payload: dict[str, list[int]] = json.loads(target.read_text(encoding="utf-8"))
    return {name: frozenset(rows) for name, rows in payload.items()}


def prepare_parquet_source(
    spec: SourceSpec,
    corpora_root: Path,
    out_root: Path,
    limit: int = 0,
) -> SourceStats:
    import pyarrow.parquet as pq

    stats = SourceStats()
    shard_dir = out_root / "shards" / spec.name
    shard_dir.mkdir(parents=True, exist_ok=True)
    excluded = load_excluded_rows(corpora_root / spec.directory)
    files = sorted((corpora_root / spec.directory).glob(spec.file_glob))
    for parquet_file in files:
        shard = shard_dir / f"{parquet_file.stem}.jsonl"
        if shard.is_file():
            for line in shard.open(encoding="utf-8"):
                stats.kept += 1
                stats.seconds += json.loads(line)["duration"]
            continue
        columns = [spec.audio_column, spec.text_column]
        if spec.speaker_column:
            columns.append(spec.speaker_column)
        if spec.duration_column:
            columns.append(spec.duration_column)
        table = pq.read_table(parquet_file, columns=columns)
        relative = parquet_file.relative_to(corpora_root)
        banned = excluded.get(
            str(parquet_file.relative_to(corpora_root / spec.directory)), frozenset()
        )
        rows: list[str] = []
        for row_index in range(table.num_rows):
            if row_index in banned:
                stats.reject("excluded")
                continue
            raw_text = str(table.column(spec.text_column)[row_index].as_py() or "")
            speaker = (
                str(table.column(spec.speaker_column)[row_index].as_py() or "")
                if spec.speaker_column
                else ""
            )
            if spec.duration_column:
                cell = table.column(spec.duration_column)[row_index].as_py()
                duration = float(cell) if cell else None
            else:
                audio_cell = table.column(spec.audio_column)[row_index].as_py()
                payload = audio_cell["bytes"] if audio_cell else None
                if not payload:
                    stats.reject("missing_audio")
                    continue
                duration = probe_duration(payload) or decode_duration(payload)
            sample_id = f"{spec.name}:{relative}:{row_index}"
            row_line = _sample_row(
                spec.name,
                raw_text,
                duration,
                f"{spec.name}:{speaker or sample_id}",
                stats,
                {"parquet": str(relative), "row": row_index},
            )
            if row_line is not None:
                rows.append(row_line)
            if limit and stats.kept >= limit:
                break
        staged = shard.with_suffix(".tmp")
        staged.write_text("".join(rows), encoding="utf-8")
        staged.replace(shard)
        if limit and stats.kept >= limit:
            break
    return stats


def prepare_fleurs(corpora_root: Path, out_root: Path, limit: int = 0) -> SourceStats:
    stats = SourceStats()
    base = corpora_root / "fleurs_uz" / "data" / "uz_uz"
    shard_dir = out_root / "shards" / FLEURS_NAME
    shard_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = out_root / "fleurs_audio"
    for tar_path in sorted(base.glob("audio/*.tar.gz")):
        marker = audio_dir / f".{tar_path.stem}.done"
        if marker.is_file():
            continue
        with tarfile.open(tar_path) as archive:
            archive.extractall(audio_dir, filter="data")
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("ok\n", encoding="utf-8")
    for tsv_path in sorted(base.glob("*.tsv")):
        shard = shard_dir / f"{tsv_path.stem}.jsonl"
        if shard.is_file():
            for line in shard.open(encoding="utf-8"):
                stats.kept += 1
                stats.seconds += json.loads(line)["duration"]
            continue
        rows: list[str] = []
        with tsv_path.open(encoding="utf-8") as handle:
            reader = csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
            for record in reader:
                if len(record) < 3:
                    continue
                file_name, raw_text = record[1], record[2]
                matches = sorted(audio_dir.rglob(file_name))
                if not matches:
                    stats.reject("missing_audio")
                    continue
                duration = _wav_duration(matches[0])
                row_line = _sample_row(
                    FLEURS_NAME,
                    raw_text,
                    duration,
                    f"{FLEURS_NAME}:{file_name}",
                    stats,
                    {"audio_filepath": str(matches[0].relative_to(out_root))},
                    forced_split={"dev": "val", "test": "test"}.get(
                        tsv_path.stem, "train"
                    ),
                )
                if row_line is not None:
                    rows.append(row_line)
                if limit and stats.kept >= limit:
                    break
        staged = shard.with_suffix(".tmp")
        staged.write_text("".join(rows), encoding="utf-8")
        staged.replace(shard)
        if limit and stats.kept >= limit:
            break
    return stats


def _wav_duration(path: Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as handle:
            rate = handle.getframerate()
            return handle.getnframes() / rate if rate else None
    except (OSError, wave.Error):
        try:
            return probe_duration(path.read_bytes())
        except OSError:
            return None


def merge_manifests(out_root: Path) -> dict[str, int]:
    counts = {"train": 0, "val": 0, "test": 0}
    handles = {
        split: (out_root / f"{split}_manifest.jsonl.tmp").open("w", encoding="utf-8")
        for split in counts
    }
    try:
        for shard in sorted((out_root / "shards").rglob("*.jsonl")):
            with shard.open(encoding="utf-8") as reader:
                for line in reader:
                    split = str(json.loads(line)["split"])
                    handles[split].write(line)
                    counts[split] += 1
    finally:
        for handle in handles.values():
            handle.close()
    for split in counts:
        staged = out_root / f"{split}_manifest.jsonl.tmp"
        staged.replace(out_root / f"{split}_manifest.jsonl")
    return counts


def iter_manifest(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def prepare(
    corpora_root: Annotated[Path | None, typer.Option("--corpora-root")] = None,
    out_root: Annotated[Path | None, typer.Option("--out-root")] = None,
    only: Annotated[list[str] | None, typer.Option("--only")] = None,
    limit: Annotated[int, typer.Option("--limit", min=0)] = 0,
) -> None:
    from uztts_data.paths import data_root

    corpora = (
        corpora_root if corpora_root is not None else data_root() / "train_corpora"
    )
    target = out_root if out_root is not None else data_root() / "asr"
    wanted = set(only) if only else set(ALL_SOURCE_NAMES)
    unknown = wanted - set(ALL_SOURCE_NAMES)
    if unknown:
        typer.echo(f"unknown source(s): {', '.join(sorted(unknown))}", err=True)
        raise typer.Exit(2)

    total_seconds = 0.0
    for spec in PARQUET_SOURCES:
        if spec.name not in wanted:
            continue
        stats = prepare_parquet_source(spec, corpora, target, limit)
        total_seconds += stats.seconds
        _echo_stats(spec.name, stats)
    if FLEURS_NAME in wanted:
        stats = prepare_fleurs(corpora, target, limit)
        total_seconds += stats.seconds
        _echo_stats(FLEURS_NAME, stats)

    counts = merge_manifests(target)
    typer.echo(
        f"manifests: train={counts['train']} val={counts['val']}"
        f" test={counts['test']} ({total_seconds / 3600:.1f} h) -> {target}"
    )


def _echo_stats(name: str, stats: SourceStats) -> None:
    rejected = sum((stats.rejected or {}).values())
    detail = ", ".join(
        f"{reason}={count}" for reason, count in sorted((stats.rejected or {}).items())
    )
    typer.echo(
        f"{name}: kept={stats.kept} ({stats.seconds / 3600:.1f} h)"
        f" rejected={rejected}" + (f" [{detail}]" if detail else "")
    )
