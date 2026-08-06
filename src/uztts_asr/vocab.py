from __future__ import annotations

from torch import nn

PUNCT_TOKENS = (".", ",", "?", "!")
NEW_ROW_BIAS = -8.0

_APOSTROPHE_MAP = str.maketrans(dict.fromkeys("ʻʼ’‘`´", "'"))
_DASH_MAP = str.maketrans(dict.fromkeys("-—–", " "))


def extended_vocab(base: list[str]) -> list[str]:
    return [*base, *PUNCT_TOKENS]


class TextEncoder:
    def __init__(self, vocab: list[str]) -> None:
        self.vocab = vocab
        self.blank_id = len(vocab)
        self._index = {token: i for i, token in enumerate(vocab)}

    def clean(self, text: str) -> str:
        mapped = text.translate(_APOSTROPHE_MAP).translate(_DASH_MAP)
        return " ".join(mapped.split())

    def encode(self, text: str) -> list[int] | None:
        try:
            return [self._index[char] for char in self.clean(text)]
        except KeyError:
            return None


def extend_ctc_conv(conv: nn.Conv1d, extra: int) -> nn.Conv1d:
    import torch

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
