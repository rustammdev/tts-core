from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    import torch

    from uztts_asr.dataset import Sample
    from uztts_asr.vocab import TextEncoder


@dataclass(frozen=True, slots=True)
class Batch:
    audio: torch.Tensor
    audio_lens: torch.Tensor
    targets: torch.Tensor
    target_lens: torch.Tensor
    texts: tuple[str, ...]

    def to(self, device: torch.device) -> Batch:
        return Batch(
            audio=self.audio.to(device, non_blocking=True),
            audio_lens=self.audio_lens.to(device, non_blocking=True),
            targets=self.targets.to(device, non_blocking=True),
            target_lens=self.target_lens.to(device, non_blocking=True),
            texts=self.texts,
        )


def collate(samples: list[Sample], targets: list[list[int]]) -> Batch:
    import torch

    max_audio = max(sample.audio.shape[0] for sample in samples)
    max_target = max(len(target) for target in targets)
    audio = torch.zeros(len(samples), max_audio, dtype=torch.float32)
    target_tensor = torch.zeros(len(samples), max_target, dtype=torch.int64)
    for row, (sample, target) in enumerate(zip(samples, targets, strict=True)):
        audio[row, : sample.audio.shape[0]] = torch.from_numpy(sample.audio)
        target_tensor[row, : len(target)] = torch.tensor(target, dtype=torch.int64)
    return Batch(
        audio=audio,
        audio_lens=torch.tensor(
            [sample.audio.shape[0] for sample in samples], dtype=torch.int64
        ),
        targets=target_tensor,
        target_lens=torch.tensor(
            [len(target) for target in targets], dtype=torch.int64
        ),
        texts=tuple(sample.text for sample in samples),
    )


def batches_of(
    samples: Iterable[Sample], encoder: TextEncoder, batch_seconds: float
) -> Iterator[Batch]:
    pending: list[Sample] = []
    pending_targets: list[list[int]] = []
    longest = 0.0
    for sample in samples:
        target = encoder.encode(sample.text)
        if not target:
            continue
        padded = max(longest, sample.duration) * (len(pending) + 1)
        if pending and padded > batch_seconds:
            yield collate(pending, pending_targets)
            pending, pending_targets, longest = [], [], 0.0
        pending.append(sample)
        pending_targets.append(target)
        longest = max(longest, sample.duration)
    if pending:
        yield collate(pending, pending_targets)
