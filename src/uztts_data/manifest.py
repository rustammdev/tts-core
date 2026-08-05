from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ValidationError

from uztts_data.schema import Segment

_CHUNK_SIZE = 1 << 20


class ManifestError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class ManifestIssue:
    line: int
    message: str


@dataclass(frozen=True, slots=True)
class ManifestReport:
    total: int
    issues: tuple[ManifestIssue, ...]

    @property
    def ok(self) -> bool:
        return not self.issues


def read_manifest(path: Path) -> Iterator[Segment]:
    for line_number, line in iter_jsonl_lines(path):
        try:
            yield Segment.model_validate_json(line)
        except ValidationError as exc:
            raise ManifestError(
                f"{path}:{line_number}: {describe_validation_error(exc)}"
            ) from exc


def write_jsonl(path: Path, rows: Iterable[BaseModel]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(path.name + ".tmp")
    written = 0
    with staged.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row.model_dump_json() + "\n")
            written += 1
    staged.replace(path)
    return written


def write_manifest(path: Path, segments: Iterable[Segment]) -> int:
    return write_jsonl(path, segments)


def validate_manifest(path: Path) -> ManifestReport:
    issues: list[ManifestIssue] = []
    seen: set[str] = set()
    total = 0
    for line_number, line in iter_jsonl_lines(path):
        total += 1
        try:
            segment = Segment.model_validate_json(line)
        except ValidationError as exc:
            issues.append(ManifestIssue(line_number, describe_validation_error(exc)))
            continue
        if segment.id in seen:
            issues.append(ManifestIssue(line_number, f"duplicate id: {segment.id}"))
        seen.add(segment.id)
    return ManifestReport(total=total, issues=tuple(issues))


def manifest_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def iter_jsonl_lines(path: Path) -> Iterator[tuple[int, str]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if line:
                yield line_number, line


def describe_validation_error(exc: ValidationError) -> str:
    messages = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"])
        messages.append(f"{location}: {error['msg']}" if location else error["msg"])
    return "; ".join(messages)
