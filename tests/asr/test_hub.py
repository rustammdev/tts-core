from __future__ import annotations

import json
import re
from pathlib import Path

from uztts_asr.hub import (
    BENCHMARK_ASSETS,
    DATA_ASSETS,
    MANIFEST_NAMES,
    MODEL_ASSETS,
    MODEL_EXTRA,
    hub_dir,
    manifest_stats,
    sha256_of,
)

ALL_ASSETS = (*MODEL_ASSETS, *MODEL_EXTRA, *DATA_ASSETS, *BENCHMARK_ASSETS)


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


def test_manifest_names_cover_every_split() -> None:
    assert {name.split("_")[0] for name in MANIFEST_NAMES} == {"train", "val", "test"}


def test_every_published_asset_exists() -> None:
    for local_name, _ in ALL_ASSETS:
        assert (hub_dir() / local_name).is_file(), local_name


def test_cards_carry_hub_front_matter() -> None:
    for local_name, target_name in ALL_ASSETS:
        if not target_name.endswith(".md"):
            continue
        text = (hub_dir() / local_name).read_text(encoding="utf-8")
        assert text.startswith("---\n"), local_name
        assert "\nlanguage:\n- uz\n" in text, local_name


def test_model_card_documents_the_shipped_checkpoints() -> None:
    script = (hub_dir() / "inference.py").read_text(encoding="utf-8")
    card = (hub_dir() / "model_card.md").read_text(encoding="utf-8")
    offered = set(re.findall(r'"(checkpoints/[^"]+\.pt)"', script))
    assert offered
    assert "base_model: ai-sage/GigaAM-Multilingual" in card
    for relative in sorted(offered):
        assert relative in card, relative


def test_every_published_config_names_a_run_in_the_model_card() -> None:
    card = (hub_dir() / "model_card.md").read_text(encoding="utf-8")
    configs = sorted((hub_dir() / "configs").glob("*.yaml"))
    assert configs
    for config in configs:
        assert f"`{config.stem}`" in card, config.stem


def test_benchmark_source_map_matches_the_published_clip_count() -> None:
    payload = json.loads((hub_dir() / "source_map.json").read_text(encoding="utf-8"))
    clips = [clip for group in payload["clips"].values() for clip in group]
    assert len(clips) == 311
    assert len({clip["id"] for clip in clips}) == 311
