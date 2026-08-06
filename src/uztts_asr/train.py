from __future__ import annotations

import json
import math
import random
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer

from uztts_asr.batching import Batch, batches_of
from uztts_asr.dataset import ManifestSampleIterator, load_rows
from uztts_asr.hub import sha256_of
from uztts_asr.vocab import PUNCT_TOKENS, TextEncoder, extend_ctc_conv, extended_vocab

if TYPE_CHECKING:
    from collections.abc import Iterator

    import torch
    from torch.utils.data import DataLoader

GIGAAM_REPO = "ai-sage/GigaAM-Multilingual"


@dataclass
class TrainConfig:
    run_name: str = "run"
    revision: str = "ctc"
    punctuated: bool = False
    sources: list[str] = field(default_factory=list)
    limit_hours: float = 0.0
    batch_seconds: float = 80.0
    accum_steps: int = 4
    num_workers: int = 2
    shuffle_buffer: int = 4096
    max_steps: int = 20000
    warmup_steps: int = 1000
    lr: float = 5e-5
    min_lr: float = 5e-6
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    seed: int = 0
    precision: str = "bf16"
    freeze_encoder: bool = False
    spec_augment: bool = True
    freq_masks: int = 2
    freq_width: int = 8
    time_masks: int = 2
    time_ratio: float = 0.05
    checkpoint_every: int = 500
    keep_checkpoints: int = 3
    val_every: int = 1000
    val_samples: int = 300
    push_best_to_hub: bool = False


def load_config(path: Path) -> TrainConfig:
    from omegaconf import OmegaConf

    merged = OmegaConf.merge(OmegaConf.structured(TrainConfig), OmegaConf.load(path))
    config = OmegaConf.to_object(merged)
    assert isinstance(config, TrainConfig)
    return config


def lr_at(step: int, config: TrainConfig) -> float:
    if step < config.warmup_steps:
        return config.lr * (step + 1) / config.warmup_steps
    span = max(1, config.max_steps - config.warmup_steps)
    progress = min(1.0, (step - config.warmup_steps) / span)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.min_lr + (config.lr - config.min_lr) * cosine


def select_rows(
    rows: list[dict[str, Any]], config: TrainConfig
) -> list[dict[str, Any]]:
    if config.sources:
        allowed = set(config.sources)
        rows = [row for row in rows if row["source"] in allowed]
    if config.limit_hours <= 0:
        return rows
    shuffled = list(rows)
    random.Random(config.seed).shuffle(shuffled)
    budget = config.limit_hours * 3600
    picked: list[dict[str, Any]] = []
    total = 0.0
    for row in shuffled:
        if total >= budget:
            break
        picked.append(row)
        total += float(row["duration"])
    return picked


def strip_punct(text: str) -> str:
    cleaned = text.translate({ord(token): " " for token in PUNCT_TOKENS})
    return " ".join(cleaned.split())


def apply_spec_augment(
    features: torch.Tensor, feature_lens: torch.Tensor, config: TrainConfig
) -> None:
    import torch

    mels = features.shape[1]
    for row in range(features.shape[0]):
        frames = int(feature_lens[row].item())
        for _ in range(config.freq_masks):
            width = int(torch.randint(1, config.freq_width + 1, (1,)).item())
            start = int(torch.randint(0, max(1, mels - width), (1,)).item())
            features[row, start : start + width, :frames] = 0.0
        widest = max(1, int(frames * config.time_ratio))
        for _ in range(config.time_masks):
            width = int(torch.randint(1, widest + 1, (1,)).item())
            start = int(torch.randint(0, max(1, frames - width), (1,)).item())
            features[row, :, start : start + width] = 0.0


def _batch_stream_class() -> type[Any]:
    import torch.utils.data

    class BatchStream(torch.utils.data.IterableDataset[Batch]):
        def __init__(
            self,
            manifest: Path,
            asr_root: Path,
            corpora_root: Path,
            encoder: TextEncoder,
            config: TrainConfig,
        ) -> None:
            self.manifest = manifest
            self.asr_root = asr_root
            self.corpora_root = corpora_root
            self.encoder = encoder
            self.config = config
            self.epoch = 0

        def _shard(self) -> list[dict[str, Any]]:
            rows = select_rows(load_rows(self.manifest), self.config)
            info = torch.utils.data.get_worker_info()
            if info is None or info.num_workers <= 1:
                return rows
            groups: dict[str, list[dict[str, Any]]] = {}
            for row in rows:
                key = str(row.get("parquet") or row.get("audio_filepath"))
                groups.setdefault(key, []).append(row)
            shard: list[dict[str, Any]] = []
            for index, key in enumerate(sorted(groups)):
                if index % info.num_workers == info.id:
                    shard.extend(groups[key])
            return shard

        def __iter__(self) -> Iterator[Batch]:
            samples = ManifestSampleIterator(
                self._shard(),
                self.asr_root,
                self.corpora_root,
                punctuated=self.config.punctuated,
                seed=self.config.seed,
                buffer_size=self.config.shuffle_buffer,
            )
            samples.set_epoch(self.epoch)
            return batches_of(samples, self.encoder, self.config.batch_seconds)

    return BatchStream


def git_commit() -> str:
    try:
        probe = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return probe.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


class Trainer:
    def __init__(self, config: TrainConfig, asr_root: Path, corpora_root: Path) -> None:
        self.config = config
        self.asr_root = asr_root
        self.corpora_root = corpora_root
        self.run_dir = asr_root / "runs" / config.run_name
        self.model: Any = None
        self.optimizer: Any = None
        self.encoder = TextEncoder([])
        self.device: Any = None
        self.step = 0
        self.stream_epoch = 0
        self.best_wer = math.inf

    def setup(self, resume: bool) -> None:
        import torch

        torch.manual_seed(self.config.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if resume:
            self._restore()
        else:
            self._write_run_metadata()

    def _load_model(self) -> None:
        from transformers import AutoModel

        wrapper = AutoModel.from_pretrained(
            GIGAAM_REPO, revision=self.config.revision, trust_remote_code=True
        )
        self.model = wrapper.model
        base_vocab = list(self.model.decoding.tokenizer.vocab)
        if self.config.punctuated:
            vocab = extended_vocab(base_vocab)
            self.model.head.decoder_layers[0] = extend_ctc_conv(
                self.model.head.decoder_layers[0], len(PUNCT_TOKENS)
            )
            self.model.decoding.tokenizer.vocab = vocab
            self.model.decoding.blank_id = len(vocab)
        else:
            vocab = base_vocab
        self.encoder = TextEncoder(vocab)
        if self.config.freeze_encoder:
            for module in (self.model.preprocessor, self.model.encoder):
                for parameter in module.parameters():
                    parameter.requires_grad = False
        self.model.to(self.device)
        self.model.train()

    def _write_run_metadata(self) -> None:
        hashes = {}
        for name in ("train_manifest.jsonl", "val_manifest.jsonl"):
            target = self.asr_root / name
            if target.is_file():
                hashes[name] = sha256_of(target)
        metadata = {
            "config": asdict(self.config),
            "git_commit": git_commit(),
            "manifests": hashes,
            "vocab_size": len(self.encoder.vocab),
            "trainable_params": sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            ),
            "started_at": time.time(),
        }
        (self.run_dir / "run.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

    def _journal(self, record: dict[str, Any]) -> None:
        record["time"] = time.time()
        with (self.run_dir / "journal.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def _encode_batch(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        import contextlib

        import torch

        with torch.no_grad():
            features, feature_lens = self.model.preprocessor(
                batch.audio, batch.audio_lens
            )
            if self.model.training and self.config.spec_augment:
                apply_spec_augment(features, feature_lens, self.config)
        autocast = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if self.device.type == "cuda" and self.config.precision == "bf16"
            else contextlib.nullcontext()
        )
        with autocast:
            encoded, encoded_lens = self.model.encoder(features, feature_lens)
        return encoded, encoded_lens

    def _loss_of(self, batch: Batch) -> torch.Tensor:
        import torch

        encoded, encoded_lens = self._encode_batch(batch)
        log_probs = self.model.head(encoded.float())
        return torch.nn.functional.ctc_loss(
            log_probs.transpose(0, 1),
            batch.targets,
            encoded_lens,
            batch.target_lens,
            blank=self.encoder.blank_id,
            zero_infinity=True,
        )

    def _optimizer_update(self) -> float:
        import torch

        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )
        )
        lr = lr_at(self.step, self.config)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        return grad_norm

    def _loader(self) -> DataLoader[Batch]:
        import torch.utils.data

        stream = _batch_stream_class()(
            self.asr_root / "train_manifest.jsonl",
            self.asr_root,
            self.corpora_root,
            self.encoder,
            self.config,
        )
        stream.epoch = self.stream_epoch
        return torch.utils.data.DataLoader(
            stream, batch_size=None, num_workers=self.config.num_workers
        )

    def train(self) -> None:
        try:
            while self.step < self.config.max_steps:
                finished = self._stream_pass()
                self.stream_epoch += 1
                if not finished:
                    self._journal({"event": "stream_epoch_end", "step": self.step})
        except KeyboardInterrupt:
            self._journal({"event": "interrupted", "step": self.step})
        finally:
            if self.step > 0:
                path = self._save_checkpoint()
                typer.echo(f"saved {path}")
        if math.isfinite(self.best_wer):
            typer.echo(f"best val wer: {self.best_wer:.4f}")

    def _stream_pass(self) -> bool:
        import torch

        micro = 0
        window: list[float] = []
        started = time.time()
        seconds_done = 0.0
        for batch in self._loader():
            on_device = batch.to(self.device)
            loss = self._loss_of(on_device) / self.config.accum_steps
            if not torch.isfinite(loss):
                self.optimizer.zero_grad(set_to_none=True)
                self._journal({"event": "nonfinite_loss", "step": self.step})
                micro = 0
                continue
            loss.backward()  # type: ignore[no-untyped-call]
            window.append(float(loss.item()) * self.config.accum_steps)
            seconds_done += float(on_device.audio_lens.sum().item()) / 16000
            micro += 1
            if micro < self.config.accum_steps:
                continue
            micro = 0
            grad_norm = self._optimizer_update()
            self.step += 1
            if self.step % 20 == 0:
                elapsed = time.time() - started
                self._journal(
                    {
                        "event": "train",
                        "step": self.step,
                        "loss": sum(window) / len(window),
                        "grad_norm": grad_norm,
                        "lr": lr_at(self.step, self.config),
                        "audio_hours_per_hour": seconds_done / max(elapsed, 1e-6),
                    }
                )
                typer.echo(
                    f"step {self.step} loss {sum(window) / len(window):.3f} "
                    f"lr {lr_at(self.step, self.config):.2e}"
                )
                window = []
            if self.step % self.config.val_every == 0:
                self._run_validation()
            if self.step % self.config.checkpoint_every == 0:
                self._save_checkpoint()
            if self.step >= self.config.max_steps:
                return True
        return False

    def _val_batches(self) -> list[Batch]:
        rows = load_rows(self.asr_root / "val_manifest.jsonl")
        if self.config.sources:
            allowed = set(self.config.sources)
            rows = [row for row in rows if row["source"] in allowed]
        random.Random(self.config.seed).shuffle(rows)
        rows = rows[: self.config.val_samples]
        samples = ManifestSampleIterator(
            rows,
            self.asr_root,
            self.corpora_root,
            punctuated=self.config.punctuated,
            seed=self.config.seed,
            buffer_size=1,
        )
        return list(batches_of(samples, self.encoder, self.config.batch_seconds))

    def _run_validation(self) -> None:
        import jiwer
        import torch

        self.model.eval()
        references: list[str] = []
        hypotheses: list[str] = []
        with torch.inference_mode():
            for batch in self._val_batches():
                on_device = batch.to(self.device)
                encoded, encoded_lens = self._encode_batch(on_device)
                decoded = self.model.decoding.decode(
                    self.model.head, encoded, encoded_lens
                )
                references.extend(self.encoder.clean(text) for text in batch.texts)
                hypotheses.extend(text for text, _, _ in decoded)
        self.model.train()
        wer = float(jiwer.wer(references, hypotheses))
        record = {"event": "val", "step": self.step, "wer": wer}
        if self.config.punctuated:
            record["wer_bare"] = float(
                jiwer.wer(
                    [strip_punct(text) for text in references],
                    [strip_punct(text) for text in hypotheses],
                )
            )
        self._journal(record)
        typer.echo(f"step {self.step} val wer {wer:.4f}")
        if wer < self.best_wer:
            self.best_wer = wer
            best = self._save_checkpoint(name="best.pt")
            if self.config.push_best_to_hub:
                self._push(best)

    def _save_checkpoint(self, name: str | None = None) -> Path:
        import torch

        payload = {
            "model": self.model.state_dict(),
            "step": self.step,
            "stream_epoch": self.stream_epoch,
            "best_wer": self.best_wer,
            "vocab": self.encoder.vocab,
            "config": asdict(self.config),
        }
        if name is None:
            payload["optimizer"] = self.optimizer.state_dict()
        target = self.run_dir / (name or f"ckpt_{self.step:07d}.pt")
        staging = target.with_suffix(".tmp")
        torch.save(payload, staging)
        staging.replace(target)
        if name is None:
            prune_checkpoints(self.run_dir, self.config.keep_checkpoints)
        return target

    def _restore(self) -> None:
        import torch

        latest = latest_checkpoint(self.run_dir)
        if latest is None:
            self._write_run_metadata()
            return
        payload = torch.load(latest, map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.step = int(payload["step"])
        self.stream_epoch = int(payload["stream_epoch"]) + 1
        self.best_wer = float(payload["best_wer"])
        self._journal({"event": "resumed", "step": self.step, "from": latest.name})
        typer.echo(f"resumed from {latest.name} at step {self.step}")

    def _push(self, checkpoint: Path) -> None:
        from uztts_asr.hub import ensure_repos, hub_api, push_checkpoint

        try:
            api = hub_api()
            model_id, _ = ensure_repos(api)
            push_checkpoint(api, checkpoint, model_id, self.config.run_name)
            self._journal({"event": "pushed", "step": self.step})
        except Exception as error:
            self._journal({"event": "push_failed", "error": str(error)})


def latest_checkpoint(run_dir: Path) -> Path | None:
    candidates = sorted(run_dir.glob("ckpt_*.pt"))
    return candidates[-1] if candidates else None


def prune_checkpoints(run_dir: Path, keep: int) -> None:
    candidates = sorted(run_dir.glob("ckpt_*.pt"))
    for stale in candidates[:-keep]:
        stale.unlink()


def train(
    config_path: Annotated[Path, typer.Option("--config")],
    resume: Annotated[bool, typer.Option("--resume")] = False,
    asr_root: Annotated[Path | None, typer.Option("--asr-root")] = None,
    corpora_root: Annotated[Path | None, typer.Option("--corpora-root")] = None,
) -> None:
    from uztts_data.paths import data_root

    root = asr_root if asr_root is not None else data_root() / "asr"
    corpora = (
        corpora_root if corpora_root is not None else data_root() / "train_corpora"
    )
    config = load_config(config_path)
    trainer = Trainer(config, root, corpora)
    trainer.setup(resume=resume)
    trainer.train()
