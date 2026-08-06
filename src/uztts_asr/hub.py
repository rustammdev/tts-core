from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

if TYPE_CHECKING:
    from huggingface_hub import HfApi

MODEL_REPO = "uz-stt"
DATA_REPO = "uz-stt-data"

MANIFEST_NAMES = ("train_manifest.jsonl", "val_manifest.jsonl", "test_manifest.jsonl")


def hub_api() -> HfApi:
    from huggingface_hub import HfApi

    return HfApi()


def repo_ids(api: HfApi) -> tuple[str, str]:
    user = str(api.whoami()["name"])
    return f"{user}/{MODEL_REPO}", f"{user}/{DATA_REPO}"


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


def dataset_card(stats: dict[str, dict[str, float]], hashes: dict[str, str]) -> str:
    lines = [
        "---",
        "license: other",
        "license_name: mixed-upstream",
        "license_link: https://huggingface.co/datasets/uzinfocom-edu-ai/uzbek-asr-curated-701h",
        "language:",
        "- uz",
        "---",
        "",
        "# uz-stt-data — o'zbekcha STT trening manifestlari (private)",
        "",
        "Audio nusxalanmaydi: har satr manba parquet fayli va satr raqamiga",
        "(`{parquet, row}`) yoki lokal faylga (`audio_filepath`) ishora qiladi.",
        "Tayyorlash retsepti: `uztts_asr.prepare` (uz-tts repo).",
        "",
        "| Split | Satrlar | Soat |",
        "|---|---|---|",
    ]
    for split in ("train", "val", "test"):
        block = stats[split]
        lines.append(f"| {split} | {int(block['rows'])} | {block['hours']:.1f} |")
    lines += ["", "## Manifest hashlari (sha256)", ""]
    lines += [f"- `{name}`: `{hashes[name]}`" for name in sorted(hashes)]
    lines += [
        "",
        "Manbalar: uzbekvoice, common_voice (validated), usc,",
        "yt_news / yt_it / yt_podcasts (islomov), fleurs.",
        "Matn: lotin, okina/tutuq normallashgan; `text` — kichik harf",
        "punktuatsiyasiz, `text_raw` — punktuatsiyali.",
        "",
    ]
    return "\n".join(lines)


def model_card() -> str:
    return "\n".join(
        [
            "---",
            "license: mit",
            "language:",
            "- uz",
            "base_model: ai-sage/GigaAM-Multilingual",
            "---",
            "",
            "# uz-stt — o'zbekcha STT (GigaAM fine-tune, private)",
            "",
            "Checkpointlar `checkpoints/` papkasida, har biri o'z konfigi bilan.",
            "",
            "Baseline (FLEURS test, 650 namuna, normallashgan matn):",
            "",
            "| Model | WER | CER |",
            "|---|---|---|",
            "| GigaAM-large (600M) | 6.7% | 1.2% |",
            "| GigaAM (220M) | 9.5% | 1.7% |",
            "| whisper-turbo-uzbek | 19.6% | 4.4% |",
            "",
            "Maqsad: fine-tune shu raqamlardan yaxshi + `. , ? !` punktuatsiya.",
            "Reja: uz-tts repo, `docs/decisions/007-gigaam-fine-tune.md`.",
            "",
        ]
    )


def ensure_repos(api: HfApi) -> tuple[str, str]:
    model_id, data_id = repo_ids(api)
    api.create_repo(model_id, repo_type="model", private=True, exist_ok=True)
    api.create_repo(data_id, repo_type="dataset", private=True, exist_ok=True)
    return model_id, data_id


def push_data_snapshot(api: HfApi, asr_root: Path, data_id: str) -> dict[str, str]:
    hashes = {name: sha256_of(asr_root / name) for name in MANIFEST_NAMES}
    stats = manifest_stats(asr_root)
    card = dataset_card(stats, hashes)
    message = "snapshot: " + ", ".join(
        f"{name}={digest[:12]}" for name, digest in sorted(hashes.items())
    )
    api.upload_file(
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo="README.md",
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
    model_id, data_id = ensure_repos(api)
    api.upload_file(
        path_or_fileobj=model_card().encode("utf-8"),
        path_in_repo="README.md",
        repo_id=model_id,
        repo_type="model",
        commit_message="model card",
    )
    typer.echo(f"model repo: https://huggingface.co/{model_id} (private)")
    typer.echo(f"data repo: https://huggingface.co/datasets/{data_id} (private)")


@app.command("push-data")
def push_data(
    asr_root: Annotated[Path | None, typer.Option("--asr-root")] = None,
) -> None:
    from uztts_data.paths import data_root

    root = asr_root if asr_root is not None else data_root() / "asr"
    api = hub_api()
    _, data_id = ensure_repos(api)
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
    model_id, _ = ensure_repos(api)
    target = push_checkpoint(api, checkpoint, model_id, tag)
    typer.echo(f"{target} -> https://huggingface.co/{model_id}")
