from __future__ import annotations

import sys
from typing import Annotated

import typer

from uztts_text.normalize import normalize

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.callback()
def main() -> None:
    pass


@app.command("normalize")
def normalize_command(
    text: Annotated[str | None, typer.Argument()] = None,
) -> None:
    source = text if text is not None else sys.stdin.read()
    typer.echo(normalize(source))
