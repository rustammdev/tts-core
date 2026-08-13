from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse

from uztts_serve.engine import MODEL_LABELS, Engine, download_url
from uztts_serve.page import INDEX_HTML

app = FastAPI(title="UzSTT demo")
engine = Engine()


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML


def _event_lines(media: Path, work: Path, model: str, events: bool) -> Iterator[str]:
    try:
        for event in engine.transcribe_stream(media, model, events):
            yield json.dumps(event, ensure_ascii=False) + "\n"
    except Exception as error:
        yield json.dumps({"type": "error", "detail": str(error)}) + "\n"
    finally:
        shutil.rmtree(work, ignore_errors=True)


@app.post("/api/transcribe")
def transcribe(
    model: Annotated[str, Form()] = "uz-stt",
    events: Annotated[str, Form()] = "false",
    url: Annotated[str, Form()] = "",
    file: UploadFile | None = None,
) -> StreamingResponse:
    if model not in MODEL_LABELS:
        raise HTTPException(400, f"noma'lum model: {model}")
    if not url and file is None:
        raise HTTPException(400, "fayl yoki havola kerak")
    work = Path(tempfile.mkdtemp(prefix="uzstt_"))
    try:
        if url:
            media = download_url(url, work)
        else:
            assert file is not None
            media = work / (Path(file.filename or "upload").name or "upload")
            with media.open("wb") as target:
                shutil.copyfileobj(file.file, target)
    except Exception as error:
        shutil.rmtree(work, ignore_errors=True)
        raise HTTPException(400, f"kirish xatosi: {error}") from error
    return StreamingResponse(
        _event_lines(media, work, model, events.lower() == "true"),
        media_type="application/x-ndjson",
    )


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=7860)
