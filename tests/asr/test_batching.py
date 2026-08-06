from __future__ import annotations

import numpy

from uztts_asr.batching import batches_of, collate
from uztts_asr.dataset import Sample
from uztts_asr.vocab import TextEncoder

VOCAB = [" ", "'", "b", "c", "h", "i", "k", "r", "u"]


def sample_of(text: str, seconds: float) -> Sample:
    return Sample(
        audio=numpy.ones(int(seconds * 16000), dtype=numpy.float32),
        text=text,
        duration=seconds,
        source="usc",
    )


def test_collate_pads_audio_and_targets() -> None:
    encoder = TextEncoder(VOCAB)
    samples = [sample_of("bir", 0.5), sample_of("ikki", 1.0)]
    targets = [encoder.encode(sample.text) for sample in samples]
    assert targets[0] is not None and targets[1] is not None
    batch = collate(samples, [targets[0], targets[1]])
    assert batch.audio.shape == (2, 16000)
    assert batch.audio_lens.tolist() == [8000, 16000]
    assert batch.audio[0, 8000:].abs().sum() == 0
    assert batch.target_lens.tolist() == [3, 4]
    assert batch.targets[0, 3] == 0
    assert batch.texts == ("bir", "ikki")


def test_batches_respect_padded_budget() -> None:
    encoder = TextEncoder(VOCAB)
    samples = [sample_of("bir", 1.0)] * 3
    batches = list(batches_of(iter(samples), encoder, batch_seconds=2.5))
    assert [batch.audio.shape[0] for batch in batches] == [2, 1]


def test_batches_skip_unencodable_text() -> None:
    encoder = TextEncoder(VOCAB)
    samples = [sample_of("bir", 1.0), sample_of("zzz", 1.0)]
    batches = list(batches_of(iter(samples), encoder, batch_seconds=10.0))
    assert len(batches) == 1
    assert batches[0].texts == ("bir",)
