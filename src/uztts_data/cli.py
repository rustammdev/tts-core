from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from uztts_data.manifest import manifest_hash, validate_manifest

app = typer.Typer(add_completion=False, no_args_is_help=True)

ManifestArgument = Annotated[
    Path, typer.Argument(exists=True, dir_okay=False, readable=True)
]


@app.command()
def validate(manifest: ManifestArgument) -> None:
    report = validate_manifest(manifest)
    for issue in report.issues:
        typer.echo(f"{manifest}:{issue.line}: {issue.message}", err=True)
    if not report.ok:
        typer.echo(f"{len(report.issues)} issue(s) in {report.total} line(s)", err=True)
        raise typer.Exit(1)
    typer.echo(f"{report.total} segment(s) ok")


@app.command("hash")
def hash_command(manifest: ManifestArgument) -> None:
    typer.echo(manifest_hash(manifest))
