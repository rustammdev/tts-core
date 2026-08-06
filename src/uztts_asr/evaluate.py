from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Protocol

import typer

from uztts_asr.prepare import (
    PARQUET_SOURCES,
    SourceSpec,
    iter_manifest,
    normalize_text,
)

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

MODEL_GIGAAM_REPO = "ai-sage/GigaAM-Multilingual"
MODEL_TURBO = "hostmepanda/whisper-large-v3-turbo-uzbek-ct2"

MODEL_CHOICES = ("gigaam", "gigaam-large", "turbo")

_SPEC_BY_NAME = {spec.name: spec for spec in PARQUET_SOURCES}


class Transcriber(Protocol):
    def transcribe(self, audio_path: Path) -> str: ...


class GigaAmTranscriber:
    def __init__(self, revision: str) -> None:
        self._revision = revision
        self._model: Any = None

    def _ensure(self) -> None:
        if self._model is None:
            from transformers import AutoModel

            self._model = AutoModel.from_pretrained(
                MODEL_GIGAAM_REPO,
                revision=self._revision,
                trust_remote_code=True,
            )

    def transcribe(self, audio_path: Path) -> str:
        self._ensure()
        result = self._model.transcribe(str(audio_path))
        return str(getattr(result, "text", result))


class TurboTranscriber:
    def __init__(self) -> None:
        self._model: WhisperModel | None = None

    def _load(self) -> WhisperModel:
        if self._model is None:
            from faster_whisper import WhisperModel

            from uztts_data.transcribe import _preload_cuda_libraries

            _preload_cuda_libraries()
            try:
                self._model = WhisperModel(MODEL_TURBO, device="cuda")
            except Exception:
                self._model = WhisperModel(MODEL_TURBO, device="cpu")
        return self._model

    def transcribe(self, audio_path: Path) -> str:
        segments, _ = self._load().transcribe(
            str(audio_path), language="uz", beam_size=5
        )
        return " ".join(segment.text.strip() for segment in segments).strip()


def make_transcriber(model: str) -> Transcriber:
    if model == "gigaam":
        return GigaAmTranscriber("ctc")
    if model == "gigaam-large":
        return GigaAmTranscriber("large_ctc")
    return TurboTranscriber()


def select_rows(
    manifest: Path, sources: set[str], limit: int = 0
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in iter_manifest(manifest):
        if sources and row["source"] not in sources:
            continue
        rows.append(row)
        if limit and len(rows) >= limit:
            break
    return rows


@dataclass(frozen=True, slots=True)
class ResolvedSample:
    row: dict[str, Any]
    audio_path: Path


def resolve_audio(
    rows: list[dict[str, Any]],
    asr_root: Path,
    corpora_root: Path,
    work_dir: Path,
) -> Iterator[ResolvedSample]:
    by_parquet: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if "audio_filepath" in row:
            yield ResolvedSample(row, asr_root / str(row["audio_filepath"]))
        else:
            by_parquet.setdefault(str(row["parquet"]), []).append(row)
    for relative, grouped in by_parquet.items():
        yield from _extract_parquet_rows(grouped, corpora_root / relative, work_dir)


def _extract_parquet_rows(
    rows: list[dict[str, Any]], parquet_file: Path, work_dir: Path
) -> Iterator[ResolvedSample]:
    import pyarrow.parquet as pq

    spec = _spec_for(str(rows[0]["source"]))
    table = pq.read_table(parquet_file, columns=[spec.audio_column])
    work_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        cell = table.column(spec.audio_column)[int(row["row"])].as_py()
        payload = cell.get("bytes") if isinstance(cell, dict) else cell
        if not isinstance(payload, bytes):
            continue
        suffix = Path(str(cell.get("path") or "clip.wav")).suffix or ".wav"
        target = work_dir / f"{parquet_file.stem}_{row['row']}{suffix}"
        target.write_bytes(payload)
        yield ResolvedSample(row, target)


def _spec_for(source: str) -> SourceSpec:
    if source not in _SPEC_BY_NAME:
        raise ValueError(f"unknown parquet source: {source}")
    return _SPEC_BY_NAME[source]


@dataclass(frozen=True, slots=True)
class EvalSummary:
    model: str
    samples: int
    hours: float
    wer: float
    cer: float


def score(model: str, pairs: list[tuple[str, str, float]]) -> EvalSummary:
    import jiwer

    references = [reference for reference, _, _ in pairs]
    hypotheses = [hypothesis for _, hypothesis, _ in pairs]
    return EvalSummary(
        model=model,
        samples=len(pairs),
        hours=sum(duration for _, _, duration in pairs) / 3600,
        wer=float(jiwer.wer(references, hypotheses)),
        cer=float(jiwer.cer(references, hypotheses)),
    )


app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def evaluate(
    model: Annotated[str, typer.Option("--model")],
    source: Annotated[list[str] | None, typer.Option("--source")] = None,
    split: Annotated[str, typer.Option("--split")] = "test",
    limit: Annotated[int, typer.Option("--limit", min=0)] = 0,
    asr_root: Annotated[Path | None, typer.Option("--asr-root")] = None,
    corpora_root: Annotated[Path | None, typer.Option("--corpora-root")] = None,
) -> None:
    from uztts_data.paths import data_root

    if model not in MODEL_CHOICES:
        typer.echo(
            f"unknown model: {model} (bor: {', '.join(MODEL_CHOICES)})", err=True
        )
        raise typer.Exit(2)
    root = asr_root if asr_root is not None else data_root() / "asr"
    corpora = (
        corpora_root if corpora_root is not None else data_root() / "train_corpora"
    )
    manifest = root / f"{split}_manifest.jsonl"
    sources = set(source) if source else {"fleurs"}
    rows = select_rows(manifest, sources, limit)
    if not rows:
        typer.echo("no matching rows", err=True)
        raise typer.Exit(1)

    transcriber = make_transcriber(model)
    out_dir = root / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model}_{split}_{'-'.join(sorted(sources))}.jsonl"
    pairs: list[tuple[str, str, float]] = []
    with out_path.open("w", encoding="utf-8") as sink:
        for sample in resolve_audio(rows, root, corpora, out_dir / "tmp"):
            hypothesis = normalize_text(transcriber.transcribe(sample.audio_path))
            reference = str(sample.row["text"])
            pairs.append((reference, hypothesis, float(sample.row["duration"])))
            sink.write(
                json.dumps(
                    {
                        "source": sample.row["source"],
                        "duration": sample.row["duration"],
                        "ref": reference,
                        "hyp": hypothesis,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            if len(pairs) % 50 == 0:
                partial = score(model, pairs)
                typer.echo(f"... {partial.samples} ta, WER={partial.wer:.3f}")

    summary = score(model, pairs)
    typer.echo(
        f"{summary.model}: samples={summary.samples} ({summary.hours:.2f} h)"
        f" WER={summary.wer:.3f} CER={summary.cer:.3f} -> {out_path}"
    )
