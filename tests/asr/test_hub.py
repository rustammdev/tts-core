from __future__ import annotations

import json
from pathlib import Path

from uztts_asr.hub import (
    MANIFEST_NAMES,
    dataset_card,
    manifest_stats,
    model_card,
    sha256_of,
)


def write_asr_root(tmp_path: Path) -> Path:
    rows = {
        "train_manifest.jsonl": [
            {"source": "usc", "duration": 3600.0},
            {"source": "fleurs", "duration": 1800.0},
        ],
        "val_manifest.jsonl": [{"source": "usc", "duration": 360.0}],
        "test_manifest.jsonl": [{"source": "fleurs", "duration": 720.0}],
    }
    for name, content in rows.items():
        (tmp_path / name).write_text(
            "".join(json.dumps(row) + "\n" for row in content), encoding="utf-8"
        )
    return tmp_path


def test_sha256_of_is_stable(tmp_path: Path) -> None:
    target = tmp_path / "x.jsonl"
    target.write_text("salom\n", encoding="utf-8")
    assert sha256_of(target) == sha256_of(target)
    assert len(sha256_of(target)) == 64


def test_manifest_stats_counts_rows_and_hours(tmp_path: Path) -> None:
    stats = manifest_stats(write_asr_root(tmp_path))
    assert stats["train"]["rows"] == 2
    assert stats["train"]["hours"] == 1.5
    assert stats["train"]["hours_usc"] == 1.0
    assert stats["test"]["hours"] == 0.2


def test_dataset_card_lists_splits_and_hashes(tmp_path: Path) -> None:
    root = write_asr_root(tmp_path)
    stats = manifest_stats(root)
    hashes = {name: sha256_of(root / name) for name in MANIFEST_NAMES}
    card = dataset_card(stats, hashes)
    assert "| train | 2 | 1.5 |" in card
    assert hashes["train_manifest.jsonl"] in card
    assert card.startswith("---\nlicense:")


def test_model_card_mentions_baseline() -> None:
    card = model_card()
    assert "6.7%" in card
    assert "base_model: ai-sage/GigaAM-Multilingual" in card
