from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Protocol

import typer

if TYPE_CHECKING:
    from faster_whisper import WhisperModel
    from faster_whisper.transcribe import TranscriptionInfo

from uztts_data.manifest import read_manifest, validate_manifest, write_manifest
from uztts_data.paths import data_root, manifests_root
from uztts_data.schema import Segment

MODEL_DEFAULT = "large-v3"
PROGRESS_EVERY = 250


@dataclass(frozen=True, slots=True)
class Transcription:
    text: str
    avg_logprob: float
    compression_ratio: float
    lang_prob: float


class Transcriber(Protocol):
    def transcribe(self, audio: Path) -> Transcription: ...


def apply_transcription(segment: Segment, result: Transcription) -> Segment:
    return segment.model_copy(
        update={
            "text": result.text.strip() or None,
            "asr_avg_logprob": result.avg_logprob,
            "asr_compression_ratio": max(result.compression_ratio, 0.01),
            "lang_prob": min(max(result.lang_prob, 0.0), 1.0),
        }
    )


def transcribed_ids(out: Path) -> set[str]:
    if not out.is_file():
        return set()
    return {segment.id for segment in read_manifest(out)}


class FasterWhisperTranscriber:
    def __init__(
        self,
        model: str = MODEL_DEFAULT,
        device: str = "auto",
        compute_type: str = "default",
        beam_size: int = 5,
    ) -> None:
        self._model_name = model
        self._device = device
        self._compute_type = compute_type
        self._beam_size = beam_size
        self._model: WhisperModel | None = None

    def transcribe(self, audio: Path) -> Transcription:
        model = self._load_model()
        segments, info = model.transcribe(
            str(audio),
            beam_size=self._beam_size,
            without_timestamps=True,
            condition_on_previous_text=False,
        )
        parts: list[str] = []
        weighted_logprob = 0.0
        total_seconds = 0.0
        compression = 0.0
        for part in segments:
            text = part.text.strip()
            if text:
                parts.append(text)
            seconds = max(part.end - part.start, 0.01)
            weighted_logprob += part.avg_logprob * seconds
            total_seconds += seconds
            compression = max(compression, part.compression_ratio)
        return Transcription(
            text=" ".join(parts),
            avg_logprob=weighted_logprob / total_seconds if total_seconds else -10.0,
            compression_ratio=compression if compression > 0 else 1.0,
            lang_prob=_uzbek_probability(info),
        )

    def _load_model(self) -> WhisperModel:
        if self._model is not None:
            return self._model
        from faster_whisper import WhisperModel

        if self._device != "auto":
            self._model = WhisperModel(
                self._model_name, device=self._device, compute_type=self._compute_type
            )
            return self._model
        try:
            self._model = WhisperModel(
                self._model_name, device="cuda", compute_type="float16"
            )
        except Exception as exc:
            typer.echo(f"cuda unavailable ({exc}); falling back to cpu int8", err=True)
            self._model = WhisperModel(
                self._model_name, device="cpu", compute_type="int8"
            )
        return self._model


def _uzbek_probability(info: TranscriptionInfo) -> float:
    for language, probability in info.all_language_probs or []:
        if language == "uz":
            return float(probability)
    return float(info.language_probability) if info.language == "uz" else 0.0


app = typer.Typer(add_completion=False)


@app.command()
def main(
    manifest: Annotated[Path | None, typer.Option("--manifest")] = None,
    out: Annotated[Path | None, typer.Option("--out")] = None,
    model: Annotated[str, typer.Option("--model")] = MODEL_DEFAULT,
    device: Annotated[str, typer.Option("--device")] = "auto",
    compute_type: Annotated[str, typer.Option("--compute-type")] = "default",
    beam_size: Annotated[int, typer.Option("--beam-size", min=1)] = 5,
    limit: Annotated[int, typer.Option("--limit", min=0)] = 0,
) -> None:
    root = data_root()
    manifest_path = (
        manifest if manifest is not None else manifests_root() / "segments.jsonl"
    )
    if not manifest_path.is_file():
        typer.echo(f"manifest not found: {manifest_path}", err=True)
        raise typer.Exit(2)
    target = out if out is not None else manifests_root() / "transcripts.jsonl"

    segments = list(read_manifest(manifest_path))
    done = transcribed_ids(target)
    pending = [segment for segment in segments if segment.id not in done]
    if limit:
        pending = pending[:limit]

    transcriber = FasterWhisperTranscriber(
        model=model, device=device, compute_type=compute_type, beam_size=beam_size
    )
    transcribed = 0
    failed = 0
    started = time.monotonic()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        for segment in pending:
            audio = root / segment.audio_path
            if not audio.is_file():
                typer.echo(f"failed: {segment.id}: missing {audio}", err=True)
                failed += 1
                continue
            try:
                result = transcriber.transcribe(audio)
            except Exception as exc:
                typer.echo(
                    f"failed: {segment.id}: {type(exc).__name__}: {exc}", err=True
                )
                failed += 1
                continue
            handle.write(apply_transcription(segment, result).model_dump_json() + "\n")
            handle.flush()
            transcribed += 1
            if transcribed % PROGRESS_EVERY == 0:
                elapsed = time.monotonic() - started
                rate = transcribed / elapsed if elapsed else 0.0
                remaining = (len(pending) - transcribed) / rate if rate else 0.0
                typer.echo(
                    f"{transcribed}/{len(pending)} segment(s), "
                    f"{rate:.1f}/s, ~{remaining / 60:.0f} min left"
                )

    report = validate_manifest(target)
    rows = list(read_manifest(target)) if report.ok else []
    if report.ok:
        write_manifest(target, sorted(rows, key=lambda row: row.id))
    typer.echo(
        f"transcribed={transcribed} skipped={len(done)} failed={failed} -> {target}"
    )
    if failed or not report.ok:
        raise typer.Exit(1)
