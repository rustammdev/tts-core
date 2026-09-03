from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

if TYPE_CHECKING:
    from huggingface_hub import HfApi

MODEL_REPO = "uzbek-asr-gigaam"
DATA_REPO = "uzbek-asr-train-manifests"
BENCHMARK_REPO = "uzbek-asr-benchmark-spontaneous"

MANIFEST_NAMES = ("train_manifest.jsonl", "val_manifest.jsonl", "test_manifest.jsonl")

MODEL_ASSETS = (("model_card.md", "README.md"), ("inference.py", "inference.py"))
MODEL_EXTRA = (("requirements.txt", "requirements.txt"),)
MODEL_CONFIG_DIR = "configs"
DATA_ASSETS = (("manifests_card.md", "README.md"),)
BENCHMARK_ASSETS = (
    ("benchmark_card.md", "README.md"),
    ("reconstruct_benchmark_audio.py", "reconstruct_benchmark_audio.py"),
    ("scoring.py", "scoring.py"),
    ("source_map.json", "source_map.json"),
)


def hub_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "hub"


def hub_api() -> HfApi:
    from huggingface_hub import HfApi

    return HfApi()


def repo_ids(api: HfApi) -> tuple[str, str, str]:
    user = str(api.whoami()["name"])
    return f"{user}/{MODEL_REPO}", f"{user}/{DATA_REPO}", f"{user}/{BENCHMARK_REPO}"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_stats(asr_root: Path) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for name in MANIFEST_NAMES:
        split = name.split("_")[0]
        rows = 0
        seconds = Counter[str]()
        with (asr_root / name).open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                rows += 1
                seconds[row["source"]] += row["duration"]
        stats[split] = {"rows": rows, "hours": sum(seconds.values()) / 3600}
        stats[split].update(
            {f"hours_{source}": value / 3600 for source, value in seconds.items()}
        )
    return stats


def ensure_repos(api: HfApi) -> tuple[str, str, str]:
    model_id, data_id, benchmark_id = repo_ids(api)
    api.create_repo(model_id, repo_type="model", exist_ok=True)
    api.create_repo(data_id, repo_type="dataset", exist_ok=True)
    api.create_repo(benchmark_id, repo_type="dataset", exist_ok=True)
    return model_id, data_id, benchmark_id


def upload_assets(
    api: HfApi,
    repo_id: str,
    repo_type: str,
    assets: tuple[tuple[str, str], ...],
    message: str,
) -> list[str]:
    source_dir = hub_dir()
    uploaded: list[str] = []
    for local_name, target_name in assets:
        api.upload_file(
            path_or_fileobj=str(source_dir / local_name),
            path_in_repo=target_name,
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=message,
        )
        uploaded.append(target_name)
    return uploaded


def push_data_snapshot(api: HfApi, asr_root: Path, data_id: str) -> dict[str, str]:
    hashes = {name: sha256_of(asr_root / name) for name in MANIFEST_NAMES}
    message = "snapshot: " + ", ".join(
        f"{name}={digest[:12]}" for name, digest in sorted(hashes.items())
    )
    stats = {"splits": manifest_stats(asr_root), "sha256": hashes}
    api.upload_file(
        path_or_fileobj=json.dumps(stats, indent=1).encode("utf-8"),
        path_in_repo="stats.json",
        repo_id=data_id,
        repo_type="dataset",
        commit_message=message,
    )
    for name in MANIFEST_NAMES:
        api.upload_file(
            path_or_fileobj=str(asr_root / name),
            path_in_repo=f"manifests/{name}",
            repo_id=data_id,
            repo_type="dataset",
            commit_message=message,
        )
    eval_dir = asr_root / "eval"
    if eval_dir.is_dir():
        api.upload_folder(
            folder_path=str(eval_dir),
            path_in_repo="eval",
            repo_id=data_id,
            repo_type="dataset",
            commit_message=message,
            allow_patterns=["*.jsonl"],
        )
    return hashes


def push_checkpoint(api: HfApi, checkpoint: Path, model_id: str, tag: str) -> str:
    target = f"checkpoints/{tag}/{checkpoint.name}"
    api.upload_file(
        path_or_fileobj=str(checkpoint),
        path_in_repo=target,
        repo_id=model_id,
        repo_type="model",
        commit_message=f"checkpoint {tag}",
    )
    return target


app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def setup() -> None:
    api = hub_api()
    model_id, data_id, benchmark_id = ensure_repos(api)
    typer.echo(f"model repo: https://huggingface.co/{model_id}")
    typer.echo(f"data repo: https://huggingface.co/datasets/{data_id}")
    typer.echo(f"benchmark repo: https://huggingface.co/datasets/{benchmark_id}")


@app.command("push-cards")
def push_cards(
    message: Annotated[str, typer.Option("--message")] = "docs: refresh hub cards",
) -> None:
    api = hub_api()
    model_id, data_id, benchmark_id = ensure_repos(api)
    targets = (
        (model_id, "model", MODEL_ASSETS + MODEL_EXTRA),
        (data_id, "dataset", DATA_ASSETS),
        (benchmark_id, "dataset", BENCHMARK_ASSETS),
    )
    for repo_id, repo_type, assets in targets:
        uploaded = upload_assets(api, repo_id, repo_type, assets, message)
        typer.echo(f"{repo_id}: {', '.join(uploaded)}")
    api.upload_folder(
        folder_path=str(hub_dir() / MODEL_CONFIG_DIR),
        path_in_repo=MODEL_CONFIG_DIR,
        repo_id=model_id,
        repo_type="model",
        commit_message=message,
        allow_patterns=["*.yaml"],
    )
    typer.echo(f"{model_id}: {MODEL_CONFIG_DIR}/")


@app.command("push-data")
def push_data(
    asr_root: Annotated[Path | None, typer.Option("--asr-root")] = None,
) -> None:
    from uztts_data.paths import data_root

    root = asr_root if asr_root is not None else data_root() / "asr"
    api = hub_api()
    _, data_id, _ = ensure_repos(api)
    hashes = push_data_snapshot(api, root, data_id)
    for name, digest in sorted(hashes.items()):
        typer.echo(f"{name}: {digest[:12]}")
    typer.echo(f"pushed -> https://huggingface.co/datasets/{data_id}")


@app.command("push-checkpoint")
def push_checkpoint_cmd(
    checkpoint: Annotated[Path, typer.Argument()],
    tag: Annotated[str, typer.Option("--tag")],
) -> None:
    api = hub_api()
    model_id, _, _ = ensure_repos(api)
    target = push_checkpoint(api, checkpoint, model_id, tag)
    typer.echo(f"{target} -> https://huggingface.co/{model_id}")
