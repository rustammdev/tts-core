from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from uztts_events.config import EventsConfig, load_config

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.callback()
def main() -> None:
    pass


@app.command("tag")
def tag(
    audio_paths: Annotated[list[Path], typer.Argument(exists=True, dir_okay=False)],
    config_path: Annotated[Path | None, typer.Option("--config")] = None,
    device: Annotated[str | None, typer.Option()] = None,
) -> None:
    from uztts_events.tagger import EventTagger, load_audio

    config = load_config(config_path) if config_path else EventsConfig()
    if device:
        config.device = device
    tagger = EventTagger(config)
    for path in audio_paths:
        waveform = load_audio(path)
        scores = tagger.screen(waveform)
        events = tagger.tag(waveform)
        print(
            json.dumps(
                {
                    "audio_path": str(path),
                    "screen_scores": {
                        label.value: round(score, 4)
                        for label, score in sorted(scores.items())
                    },
                    "events": [
                        {
                            "label": event.label.value,
                            "start": round(event.start, 2),
                            "end": round(event.end, 2),
                            "score": round(event.score, 3),
                        }
                        for event in events
                    ],
                },
                ensure_ascii=False,
            )
        )
