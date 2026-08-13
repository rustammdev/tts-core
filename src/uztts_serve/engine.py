from __future__ import annotations

import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from uztts_data.paths import data_root
from uztts_data.segment import SileroDetector, clamp_spans, cut_wav

if TYPE_CHECKING:
    from collections.abc import Iterator

    from uztts_asr.evaluate import Transcriber
    from uztts_events.tagger import EventTagger

SAMPLE_RATE = 16_000
MIN_CHUNK_SECONDS = 0.6
MAX_CHUNK_SECONDS = 12.0
MAX_AUDIO_SECONDS = 30 * 60.0

MODEL_LABELS = {
    "uz-stt": "UzSTT (bizniki, gemini_full_220m)",
    "gigaam": "GigaAM 220M (baza)",
    "gigaam-large": "GigaAM-large 600M (baza)",
}


def model_source(key: str) -> str:
    if key == "uz-stt":
        return str(data_root() / "asr" / "runs" / "gemini_full_220m" / "best.pt")
    if key in ("gigaam", "gigaam-large"):
        return key
    raise ValueError(f"unknown model: {key}")


def extract_audio(source: Path, target: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-i",
            str(source),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            str(target),
        ],
        capture_output=True,
        check=True,
    )


def download_url(url: str, out_dir: Path) -> Path:
    import yt_dlp

    options = {
        "format": "bestaudio/best",
        "outtmpl": str(out_dir / "source.%(ext)s"),
        "quiet": True,
        "noplaylist": True,
    }
    with yt_dlp.YoutubeDL(options) as downloader:
        downloader.download([url])
    files = [p for p in out_dir.iterdir() if p.name.startswith("source.")]
    if not files:
        raise RuntimeError("yuklab olinmadi")
    return files[0]


@dataclass
class Engine:
    _transcribers: dict[str, Transcriber] = field(default_factory=dict)
    _tagger: EventTagger | None = None
    _detector: SileroDetector | None = None

    def transcriber(self, key: str) -> Transcriber:
        if key not in self._transcribers:
            from uztts_asr.evaluate import make_transcriber

            self._transcribers[key] = make_transcriber(model_source(key))
        return self._transcribers[key]

    def tagger(self) -> EventTagger:
        if self._tagger is None:
            import torch

            from uztts_events.config import EventsConfig
            from uztts_events.tagger import EventTagger

            config = EventsConfig()
            config.device = "cuda" if torch.cuda.is_available() else "cpu"
            self._tagger = EventTagger(config)
        return self._tagger

    def detector(self) -> SileroDetector:
        if self._detector is None:
            self._detector = SileroDetector(max_seconds=MAX_CHUNK_SECONDS)
        return self._detector

    def transcribe_stream(
        self, media: Path, model_key: str, with_events: bool
    ) -> Iterator[dict[str, Any]]:
        import soundfile

        timings: dict[str, float] = {}
        with tempfile.TemporaryDirectory() as workdir:
            work = Path(workdir)
            wav = work / "audio.wav"
            started = time.time()
            extract_audio(media, wav)
            timings["decode"] = time.time() - started

            waveform, _ = soundfile.read(str(wav), dtype="float32")
            duration = len(waveform) / SAMPLE_RATE
            if duration > MAX_AUDIO_SECONDS:
                raise ValueError(
                    f"audio {duration/60:.0f} daqiqa — chegara "
                    f"{MAX_AUDIO_SECONDS/60:.0f} daqiqa"
                )

            started = time.time()
            spans, _ = clamp_spans(
                self.detector().speech_spans(wav),
                MIN_CHUNK_SECONDS,
                MAX_CHUNK_SECONDS,
            )
            timings["vad"] = time.time() - started

            yield {
                "type": "meta",
                "model": model_key,
                "model_label": MODEL_LABELS[model_key],
                "duration": round(duration, 1),
                "chunks": len(spans),
            }

            started = time.time()
            chunk_files = cut_wav(wav, spans, work / "chunks")
            transcriber = self.transcriber(model_key)
            segments: list[dict[str, Any]] = []
            for span, chunk in zip(spans, chunk_files, strict=True):
                text = transcriber.transcribe(chunk).strip()
                if not text:
                    continue
                segment = {
                    "start": round(span.start, 2),
                    "end": round(span.end, 2),
                    "text": text,
                }
                segments.append(segment)
                yield {"type": "segment", "done": len(segments), **segment}
            timings["asr"] = time.time() - started

            events: list[dict[str, Any]] = []
            if with_events:
                started = time.time()
                events = [
                    {
                        "label": event.label.value,
                        "start": round(event.start, 2),
                        "end": round(event.end, 2),
                        "score": round(event.score, 3),
                    }
                    for event in self.tagger().tag(waveform)
                ]
                timings["events"] = time.time() - started
                yield {"type": "events", "events": events}

        yield {
            "type": "done",
            "text": merged_text(segments, events),
            "timings": {name: round(value, 1) for name, value in timings.items()},
        }


def merged_text(
    segments: list[dict[str, Any]], events: list[dict[str, Any]]
) -> str:
    from uztts_events.merge import merge_transcript
    from uztts_events.schema import AudioEvent, EventLabel, Word

    words = [
        Word(text=str(seg["text"]), start=float(seg["start"]), end=float(seg["end"]))
        for seg in segments
    ]
    parsed = [
        AudioEvent(
            label=EventLabel(event["label"]),
            start=float(event["start"]),
            end=float(event["end"]),
            score=float(event["score"]),
        )
        for event in events
    ]
    return merge_transcript(words, parsed)
