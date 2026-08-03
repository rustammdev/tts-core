from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.conftest import SegmentFactory
from uztts_data import (
    ManifestError,
    Segment,
    manifest_hash,
    read_manifest,
    validate_manifest,
    write_manifest,
)


def test_roundtrip_preserves_segments(
    tmp_path: Path, make_segment: SegmentFactory
) -> None:
    segments = [make_segment(1), make_segment(2, text="Salom dunyo")]
    path = tmp_path / "nested" / "train.jsonl"
    assert write_manifest(path, segments) == 2
    assert list(read_manifest(path)) == segments


def test_write_keeps_unicode_and_field_order(
    tmp_path: Path, make_segment: SegmentFactory
) -> None:
    path = tmp_path / "train.jsonl"
    write_manifest(path, [make_segment(1, text="O'zbek tili — g'alaba")])
    line = path.read_text(encoding="utf-8").splitlines()[0]
    assert "O'zbek tili — g'alaba" in line
    assert list(json.loads(line)) == list(Segment.model_fields)


def test_write_leaves_no_staging_file(
    tmp_path: Path, make_segment: SegmentFactory
) -> None:
    path = tmp_path / "train.jsonl"
    write_manifest(path, [make_segment(1)])
    assert [entry.name for entry in tmp_path.iterdir()] == ["train.jsonl"]


def test_read_skips_blank_lines(tmp_path: Path, make_segment: SegmentFactory) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text(f"\n{make_segment(1).model_dump_json()}\n\n", encoding="utf-8")
    assert len(list(read_manifest(path))) == 1


def test_read_reports_line_number(tmp_path: Path, make_segment: SegmentFactory) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text(
        f'{make_segment(1).model_dump_json()}\n{{"id": "broken"}}\n', encoding="utf-8"
    )
    with pytest.raises(ManifestError, match=r"train\.jsonl:2:"):
        list(read_manifest(path))


def test_validate_accepts_clean_manifest(
    tmp_path: Path, make_segment: SegmentFactory
) -> None:
    path = tmp_path / "train.jsonl"
    write_manifest(path, [make_segment(1), make_segment(2)])
    report = validate_manifest(path)
    assert report.ok
    assert report.total == 2


def test_validate_flags_duplicate_ids(
    tmp_path: Path, make_segment: SegmentFactory
) -> None:
    path = tmp_path / "train.jsonl"
    write_manifest(path, [make_segment(1), make_segment(1)])
    report = validate_manifest(path)
    assert report.total == 2
    assert [issue.line for issue in report.issues] == [2]
    assert "duplicate id" in report.issues[0].message


def test_validate_collects_every_broken_line(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text('{"id": "a"}\nnot json\n', encoding="utf-8")
    report = validate_manifest(path)
    assert [issue.line for issue in report.issues] == [1, 2]


def test_hash_tracks_content(tmp_path: Path, make_segment: SegmentFactory) -> None:
    first = tmp_path / "a.jsonl"
    same = tmp_path / "b.jsonl"
    other = tmp_path / "c.jsonl"
    write_manifest(first, [make_segment(1)])
    write_manifest(same, [make_segment(1)])
    write_manifest(other, [make_segment(2)])
    assert manifest_hash(first) == manifest_hash(same)
    assert manifest_hash(first) != manifest_hash(other)
