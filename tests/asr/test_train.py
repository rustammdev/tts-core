from __future__ import annotations

import io
import json
import wave
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from uztts_asr.train import (
    TrainConfig,
    Trainer,
    apply_spec_augment,
    filter_rows,
    keep_row,
    latest_checkpoint,
    load_config,
    lr_at,
    needs_whole_manifest,
    owns_group,
    prune_checkpoints,
    select_rows,
    strip_punct,
)
from uztts_asr.vocab import TextEncoder

VOCAB = [" ", "'", "b", "i", "k", "r", "u"]


def wav_bytes(seconds: float) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x10\x00" * int(seconds * 16000))
    return buffer.getvalue()


class StubPreprocessor(torch.nn.Module):
    def forward(
        self, signal: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = torch.nn.functional.avg_pool1d(signal.unsqueeze(1), 160, 160)
        return features.repeat(1, 4, 1), torch.div(lengths, 160, rounding_mode="floor")


class StubEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv1d(4, 8, kernel_size=1)

    def forward(
        self, features: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.conv(features), lengths


class StubHead(torch.nn.Module):
    def __init__(self, classes: int) -> None:
        super().__init__()
        self.decoder_layers = torch.nn.Sequential(
            torch.nn.Conv1d(8, classes, kernel_size=1)
        )

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.log_softmax(
            self.decoder_layers(encoder_output).transpose(1, 2), dim=-1
        )


class StubDecoding:
    def decode(
        self, head: torch.nn.Module, encoded: torch.Tensor, lengths: torch.Tensor
    ) -> list[tuple[str, list[int], list[int]]]:
        return [("bir", [], [])] * encoded.shape[0]


class StubAsr(torch.nn.Module):
    def __init__(self, classes: int) -> None:
        super().__init__()
        self.preprocessor = StubPreprocessor()
        self.encoder = StubEncoder()
        self.head = StubHead(classes)
        self.decoding: Any = StubDecoding()


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def make_run_env(tmp_path: Path) -> tuple[Path, Path]:
    asr_root = tmp_path / "asr"
    (asr_root / "audio").mkdir(parents=True)
    corpora = tmp_path / "corpora"
    corpora.mkdir()
    train_rows: list[dict[str, object]] = []
    for index, text in enumerate(["bir", "ikki", "bir", "ikki", "bir", "ikki"]):
        name = f"audio/{index}.wav"
        (asr_root / name).write_bytes(wav_bytes(1.0))
        train_rows.append(
            {"audio_filepath": name, "text": text, "duration": 1.0, "source": "fleurs"}
        )
    write_manifest(asr_root / "train_manifest.jsonl", train_rows)
    write_manifest(asr_root / "val_manifest.jsonl", [train_rows[0], train_rows[2]])
    return asr_root, corpora


def make_trainer(tmp_path: Path, config: TrainConfig) -> Trainer:
    asr_root, corpora = make_run_env(tmp_path)
    trainer = Trainer(config, asr_root, corpora)
    trainer.device = torch.device("cpu")
    trainer.model = StubAsr(len(VOCAB) + 1)
    trainer.encoder = TextEncoder(VOCAB)
    trainer.optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=1e-3)
    trainer.run_dir.mkdir(parents=True, exist_ok=True)
    return trainer


def small_config() -> TrainConfig:
    return TrainConfig(
        run_name="t",
        accum_steps=1,
        num_workers=0,
        shuffle_buffer=4,
        max_steps=4,
        warmup_steps=2,
        checkpoint_every=2,
        keep_checkpoints=1,
        val_every=3,
        val_samples=2,
        batch_seconds=5.0,
        spec_augment=False,
        precision="fp32",
    )


def test_trainer_runs_checkpoints_and_validates(tmp_path: Path) -> None:
    trainer = make_trainer(tmp_path, small_config())
    trainer.train()
    assert trainer.step == 4
    assert (trainer.run_dir / "best.pt").exists()
    assert [p.name for p in sorted(trainer.run_dir.glob("ckpt_*.pt"))] == [
        "ckpt_0000004.pt"
    ]
    events = [
        json.loads(line)["event"]
        for line in (trainer.run_dir / "journal.jsonl").read_text().splitlines()
    ]
    assert "val" in events
    assert trainer.best_wer == 0.0


def test_trainer_restore_resumes_state(tmp_path: Path) -> None:
    trainer = make_trainer(tmp_path, small_config())
    trainer.train()
    resumed = Trainer(small_config(), trainer.asr_root, trainer.corpora_root)
    resumed.device = torch.device("cpu")
    resumed.model = StubAsr(len(VOCAB) + 1)
    resumed.encoder = TextEncoder(VOCAB)
    resumed.optimizer = torch.optim.AdamW(resumed.model.parameters(), lr=1e-3)
    resumed._restore()
    assert resumed.step == 4
    assert resumed.stream_epoch == trainer.stream_epoch + 1
    assert resumed.best_wer == 0.0


def test_lr_schedule_warmup_and_cosine() -> None:
    config = TrainConfig(max_steps=100, warmup_steps=10, lr=1e-4, min_lr=1e-5)
    assert lr_at(0, config) == 1e-5
    assert lr_at(10, config) == 1e-4
    assert abs(lr_at(100, config) - 1e-5) < 1e-12
    assert lr_at(55, config) < lr_at(20, config)


def test_select_rows_filters_and_limits() -> None:
    rows = [
        {"source": "usc", "duration": 3600.0},
        {"source": "fleurs", "duration": 3600.0},
        {"source": "usc", "duration": 3600.0},
    ]
    config = TrainConfig(sources=["usc"])
    assert len(select_rows(rows, config)) == 2
    limited = select_rows(rows, TrainConfig(limit_hours=1.0))
    assert len(limited) == 1
    assert select_rows(rows, TrainConfig(limit_hours=1.0)) == limited


def test_select_rows_caps_source_hours() -> None:
    rows = [
        {"source": "yt_gemini", "duration": 1800.0},
        {"source": "yt_gemini", "duration": 1800.0},
        {"source": "yt_gemini", "duration": 1800.0},
        {"source": "usc", "duration": 1800.0},
    ]
    config = TrainConfig(source_hours={"yt_gemini": 1.0})
    picked = select_rows(rows, config)
    assert sum(1 for row in picked if row["source"] == "yt_gemini") == 2
    assert sum(1 for row in picked if row["source"] == "usc") == 1
    assert select_rows(rows, config) == picked


def test_owns_group_partitions_every_key_once() -> None:
    keys = [f"yt_gemini/data/shard-{index:05d}.parquet" for index in range(200)]
    workers = 4
    owners = [
        [worker for worker in range(workers) if owns_group(key, worker, workers)]
        for key in keys
    ]
    assert all(len(owner) == 1 for owner in owners)
    covered = {owner[0] for owner in owners}
    assert covered == set(range(workers))


def test_needs_whole_manifest_only_for_global_budgets() -> None:
    assert not needs_whole_manifest(TrainConfig(sources=["usc"], require_punct=True))
    assert needs_whole_manifest(TrainConfig(limit_hours=1.0))
    assert needs_whole_manifest(TrainConfig(source_hours={"usc": 1.0}))


def test_keep_row_matches_filter_rows() -> None:
    rows = [
        {"source": "usc", "text_raw": "Salom, dunyo."},
        {"source": "usc", "text_raw": "salom dunyo"},
        {"source": "fleurs", "text_raw": "Salom, dunyo."},
    ]
    config = TrainConfig(sources=["usc"], require_punct=True)
    assert [row for row in rows if keep_row(row, config)] == filter_rows(rows, config)
    assert len(filter_rows(rows, config)) == 1


def test_strip_punct() -> None:
    assert strip_punct("salom, dunyo! qalaysiz?") == "salom dunyo qalaysiz"


def test_filter_rows_require_punct() -> None:
    rows: list[dict[str, object]] = [
        {"source": "fleurs", "text_raw": "Salom, dunyo."},
        {"source": "fleurs", "text_raw": "salom dunyo"},
        {"source": "fleurs"},
    ]
    kept = filter_rows(rows, TrainConfig(require_punct=True))
    assert kept == [rows[0]]
    assert filter_rows(rows, TrainConfig()) == rows


def test_init_from_loads_weights_then_extends_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = StubAsr(len(VOCAB) + 1)
    checkpoint = tmp_path / "best.pt"
    torch.save(
        {"model": base.state_dict(), "config": {"punctuated": False}}, checkpoint
    )

    fresh = StubAsr(len(VOCAB) + 1)
    fresh.decoding = SimpleNamespace(
        tokenizer=SimpleNamespace(vocab=list(VOCAB)), blank_id=len(VOCAB)
    )

    def fake_from_pretrained(*args: object, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(model=fresh)

    monkeypatch.setattr("transformers.AutoModel.from_pretrained", fake_from_pretrained)
    config = TrainConfig(punctuated=True, init_from=str(checkpoint))
    trainer = Trainer(config, tmp_path / "asr", tmp_path / "corpora")
    trainer.device = torch.device("cpu")
    trainer._load_model()

    extended = trainer.model.head.decoder_layers[0]
    base_conv = cast(torch.nn.Conv1d, base.head.decoder_layers[0])
    assert extended.out_channels == len(VOCAB) + 1 + 4
    assert trainer.encoder.vocab[-4:] == [".", ",", "?", "!"]
    assert torch.equal(extended.weight[: len(VOCAB)], base_conv.weight[: len(VOCAB)])
    assert torch.equal(extended.weight[-1], base_conv.weight[len(VOCAB)])
    assert torch.all(extended.bias[len(VOCAB) : len(VOCAB) + 4] == -8.0)


def test_prune_and_latest_checkpoint(tmp_path: Path) -> None:
    for step in (2, 4, 6):
        (tmp_path / f"ckpt_{step:07d}.pt").write_bytes(b"x")
    prune_checkpoints(tmp_path, keep=2)
    remaining = sorted(p.name for p in tmp_path.glob("ckpt_*.pt"))
    assert remaining == ["ckpt_0000004.pt", "ckpt_0000006.pt"]
    latest = latest_checkpoint(tmp_path)
    assert latest is not None and latest.name == "ckpt_0000006.pt"
    assert latest_checkpoint(tmp_path / "missing") is None


def test_load_config_merges_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "c.yaml"
    config_path.write_text("run_name: demo\nlr: 1.0e-4\n", encoding="utf-8")
    config = load_config(config_path)
    assert config.run_name == "demo"
    assert config.lr == 1e-4
    assert config.accum_steps == 4


def test_spec_augment_masks_within_lengths() -> None:
    features = torch.ones(2, 16, 40)
    lens = torch.tensor([40, 30])
    apply_spec_augment(features, lens, TrainConfig(freq_masks=2, time_masks=2))
    assert features.shape == (2, 16, 40)
    assert (features == 0).any()
    assert torch.all(features[1, :, 35:] == 1.0)
