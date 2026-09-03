from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import hf_hub_download
from torch import nn
from transformers import AutoModel

REPO_ID = "rustam1221/uzbek-asr-gigaam"
BASE_MODEL = "ai-sage/GigaAM-Multilingual"
PUNCT_TOKENS = (".", ",", "?", "!")
NEW_ROW_BIAS = -8.0
MAX_CHUNK_SECONDS = 30.0

CHECKPOINTS = {
    "large-600m": "checkpoints/large_full_600m/best.pt",
    "220m": "checkpoints/gemini_full_220m/best.pt",
    "220m-no-youtube-llm": "checkpoints/punct_220m/best.pt",
}

_SENTENCE_START_RE = re.compile(r"(^\s*|[.?!]\s+)([a-z])")


def extend_ctc_conv(conv: nn.Conv1d, extra: int) -> nn.Conv1d:
    old_classes = conv.out_channels
    extended = nn.Conv1d(conv.in_channels, old_classes + extra, kernel_size=1)
    with torch.no_grad():
        extended.weight.zero_()
        assert extended.bias is not None and conv.bias is not None
        extended.bias.fill_(NEW_ROW_BIAS)
        extended.weight[: old_classes - 1] = conv.weight[: old_classes - 1]
        extended.bias[: old_classes - 1] = conv.bias[: old_classes - 1]
        extended.weight[-1] = conv.weight[old_classes - 1]
        extended.bias[-1] = conv.bias[old_classes - 1]
    return extended


def capitalize_sentences(text: str) -> str:
    return _SENTENCE_START_RE.sub(
        lambda match: match.group(1) + match.group(2).upper(), text
    )


class UzbekAsr:
    def __init__(self, checkpoint: Path, device: str | None = None) -> None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        config = payload["config"]
        wrapper: Any = AutoModel.from_pretrained(
            BASE_MODEL, revision=str(config["revision"]), trust_remote_code=True
        )
        self.punctuated = bool(config["punctuated"])
        if self.punctuated:
            vocab = list(payload["vocab"])
            wrapper.model.head.decoder_layers[0] = extend_ctc_conv(
                wrapper.model.head.decoder_layers[0], len(PUNCT_TOKENS)
            )
            wrapper.model.decoding.tokenizer.vocab = vocab
            wrapper.model.decoding.blank_id = len(vocab)
        wrapper.model.load_state_dict(payload["model"])
        wrapper.model.eval()
        resolved = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = wrapper.to(resolved) if resolved != "cpu" else wrapper
        self.device = resolved

    @classmethod
    def from_hub(
        cls, variant: str = "large-600m", device: str | None = None
    ) -> UzbekAsr:
        if variant not in CHECKPOINTS:
            raise ValueError(f"unknown variant {variant!r}: {sorted(CHECKPOINTS)}")
        path = hf_hub_download(REPO_ID, CHECKPOINTS[variant])
        return cls(Path(path), device=device)

    def transcribe(self, audio_path: str | Path) -> str:
        with tempfile.TemporaryDirectory() as scratch:
            prepared = Path(scratch) / "input.wav"
            to_mono_16k(Path(audio_path), prepared)
            text = self._transcribe_wav(prepared)
        return capitalize_sentences(text) if self.punctuated else text

    def _transcribe_wav(self, wav_path: Path) -> str:
        audio, rate = sf.read(str(wav_path), dtype="float32")
        if len(audio) <= MAX_CHUNK_SECONDS * rate:
            try:
                with torch.inference_mode():
                    result = self.model.transcribe(str(wav_path))
            except ValueError:
                return self._transcribe_halves(audio, rate)
            return str(getattr(result, "text", result)).strip()
        return self._transcribe_halves(audio, rate)

    def _transcribe_halves(self, audio: np.ndarray, rate: int) -> str:
        parts: list[str] = []
        with tempfile.TemporaryDirectory() as scratch:
            for index, chunk in enumerate(split_on_silence(audio, rate)):
                target = Path(scratch) / f"chunk{index}.wav"
                sf.write(str(target), chunk, rate)
                parts.append(self._transcribe_wav(target))
        return " ".join(part for part in parts if part).strip()


def split_on_silence(audio: np.ndarray, rate: int) -> tuple[np.ndarray, np.ndarray]:
    window = max(1, rate // 5)
    low, high = int(len(audio) * 0.3), int(len(audio) * 0.7)
    smoothed = np.convolve(
        np.abs(audio[low:high]), np.ones(window) / window, mode="same"
    )
    cut = low + int(np.argmin(smoothed))
    return audio[:cut], audio[cut:]


def to_mono_16k(source: Path, target: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(target),
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe Uzbek speech with a fine-tuned GigaAM CTC model."
    )
    parser.add_argument(
        "audio", nargs="+", help="audio or video files (any ffmpeg format)"
    )
    parser.add_argument(
        "--variant",
        default="large-600m",
        choices=sorted(CHECKPOINTS),
        help="checkpoint to download from the Hub (default: large-600m)",
    )
    parser.add_argument(
        "--checkpoint", type=Path, help="local .pt checkpoint instead of --variant"
    )
    parser.add_argument("--device", help="torch device (default: cuda if available)")
    args = parser.parse_args()

    asr = (
        UzbekAsr(args.checkpoint, device=args.device)
        if args.checkpoint
        else UzbekAsr.from_hub(args.variant, device=args.device)
    )
    for path in args.audio:
        print(f"{path}: {asr.transcribe(path)}")


if __name__ == "__main__":
    main()
