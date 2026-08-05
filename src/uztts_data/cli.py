from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from uztts_data.channels import (
    YtDlpLister,
    merge_stats,
    read_registry,
    read_stats,
    render_report,
    validate_registry,
)
from uztts_data.manifest import (
    ManifestError,
    ManifestReport,
    manifest_hash,
    summarize_manifest,
    validate_manifest,
    write_jsonl,
)
from uztts_data.paths import data_root, manifests_root, raw_root
from uztts_data.scan import scan_raw

app = typer.Typer(add_completion=False, no_args_is_help=True)
channels_app = typer.Typer(add_completion=False, no_args_is_help=True)
app.add_typer(channels_app, name="channels")

ManifestArgument = Annotated[
    Path, typer.Argument(exists=True, dir_okay=False, readable=True)
]
RegistryArgument = Annotated[
    Path, typer.Argument(exists=True, dir_okay=False, readable=True)
]
DEFAULT_REGISTRY = Path("configs/channels.jsonl")


@app.command()
def validate(manifest: ManifestArgument) -> None:
    report = validate_manifest(manifest)
    _echo_issues(manifest, report)
    typer.echo(f"{report.total} segment(s) ok")


@app.command("hash")
def hash_command(manifest: ManifestArgument) -> None:
    typer.echo(manifest_hash(manifest))


@app.command("scan-raw")
def scan_raw_command(
    raw: Annotated[Path | None, typer.Option("--raw", file_okay=False)] = None,
    out: Annotated[Path | None, typer.Option("--out")] = None,
) -> None:
    raw_dir = raw if raw is not None else raw_root()
    if not raw_dir.is_dir():
        typer.echo(f"raw directory not found: {raw_dir}", err=True)
        raise typer.Exit(2)
    segments, issues = scan_raw(raw_dir, data_root())
    target = out if out is not None else manifests_root() / "raw.jsonl"
    write_jsonl(target, segments)
    typer.echo(f"{len(segments)} segment(s) -> {target}")
    for issue in issues:
        typer.echo(f"skipped: {issue}", err=True)
    if issues:
        raise typer.Exit(1)


@app.command()
def stats(manifest: ManifestArgument) -> None:
    try:
        summary = summarize_manifest(manifest)
    except ManifestError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    ranked = sorted(
        summary.hours_by_channel.items(), key=lambda item: item[1], reverse=True
    )
    for channel, hours in ranked:
        typer.echo(f"{channel:<24} {hours:>8.2f} h")
    typer.echo(f"total: {summary.segments} segment(s), {summary.hours:.2f} h")


@channels_app.command("validate")
def channels_validate(registry: RegistryArgument = DEFAULT_REGISTRY) -> None:
    report = validate_registry(registry)
    _echo_issues(registry, report)
    typer.echo(f"{report.total} channel(s) ok")


@channels_app.command("stats")
def channels_stats(
    registry: RegistryArgument = DEFAULT_REGISTRY,
    out: Annotated[Path | None, typer.Option("--out")] = None,
    refresh: Annotated[bool, typer.Option("--refresh")] = False,
    target_hours: Annotated[float, typer.Option("--target-hours", min=1.0)] = 1000.0,
) -> None:
    report = validate_registry(registry)
    _echo_issues(registry, report)

    channels = list(read_registry(registry))
    stats_path = out if out is not None else manifests_root() / "channel_stats.jsonl"
    existing = read_stats(stats_path) if stats_path.is_file() else {}
    stats, errors = merge_stats(channels, existing, YtDlpLister(), refresh=refresh)
    write_jsonl(stats_path, stats)

    by_id = {stat.channel_id: stat for stat in stats}
    typer.echo(render_report(channels, by_id, target_hours=target_hours))
    for error in errors:
        typer.echo(f"failed: {error}", err=True)
    if errors:
        raise typer.Exit(1)


def _echo_issues(path: Path, report: ManifestReport) -> None:
    for issue in report.issues:
        typer.echo(f"{path}:{issue.line}: {issue.message}", err=True)
    if not report.ok:
        typer.echo(f"{len(report.issues)} issue(s) in {report.total} line(s)", err=True)
        raise typer.Exit(1)
