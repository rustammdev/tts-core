from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from tests.conftest import SegmentFactory
from uztts_data import manifest_hash, write_manifest
from uztts_data.cli import app

runner = CliRunner()


def test_validate_accepts_clean_manifest(
    tmp_path: Path, make_segment: SegmentFactory
) -> None:
    path = tmp_path / "train.jsonl"
    write_manifest(path, [make_segment(1), make_segment(2)])
    result = runner.invoke(app, ["validate", str(path)])
    assert result.exit_code == 0
    assert "2 segment(s) ok" in result.stdout


def test_validate_fails_on_duplicate_ids(
    tmp_path: Path, make_segment: SegmentFactory
) -> None:
    path = tmp_path / "train.jsonl"
    write_manifest(path, [make_segment(1), make_segment(1)])
    assert runner.invoke(app, ["validate", str(path)]).exit_code == 1


def test_validate_fails_on_missing_manifest(tmp_path: Path) -> None:
    assert runner.invoke(app, ["validate", str(tmp_path / "nope.jsonl")]).exit_code != 0


def test_hash_matches_library(tmp_path: Path, make_segment: SegmentFactory) -> None:
    path = tmp_path / "train.jsonl"
    write_manifest(path, [make_segment(1)])
    result = runner.invoke(app, ["hash", str(path)])
    assert result.exit_code == 0
    assert result.stdout.strip() == manifest_hash(path)
