from __future__ import annotations

import torch

from uztts_asr.vocab import (
    NEW_ROW_BIAS,
    PUNCT_TOKENS,
    TextEncoder,
    extend_ctc_conv,
    extended_vocab,
)


def test_extended_vocab_appends_punct() -> None:
    assert extended_vocab([" ", "a"]) == [" ", "a", ".", ",", "?", "!"]


def test_clean_maps_apostrophes_and_dashes() -> None:
    encoder = TextEncoder([" ", "'", "g", "o", "t", "r"])
    assert encoder.clean("toʻgʻri") == "to'g'ri"
    assert encoder.clean("toʼxta — bor") == "to'xta bor"
    assert encoder.clean("g'oʻr") == "g'o'r"


def test_encode_roundtrip_and_blank() -> None:
    vocab = [" ", "'", "b", "i", "r", "o"]
    encoder = TextEncoder(vocab)
    assert encoder.blank_id == len(vocab)
    tokens = encoder.encode("bir boʻri")
    assert tokens is not None
    assert "".join(vocab[token] for token in tokens) == "bir bo'ri"


def test_encode_unknown_char_returns_none() -> None:
    encoder = TextEncoder([" ", "a"])
    assert encoder.encode("azb") is None


def test_extend_ctc_conv_moves_blank_last() -> None:
    conv = torch.nn.Conv1d(3, 5, kernel_size=1)
    extra = len(PUNCT_TOKENS)
    extended = extend_ctc_conv(conv, extra)
    assert extended.out_channels == 5 + extra
    assert extended.in_channels == 3
    assert conv.bias is not None and extended.bias is not None
    assert torch.equal(extended.weight[:4], conv.weight[:4])
    assert torch.equal(extended.bias[:4], conv.bias[:4])
    assert torch.equal(extended.weight[-1], conv.weight[4])
    assert torch.equal(extended.bias[-1], conv.bias[4])
    assert torch.all(extended.weight[4 : 4 + extra] == 0)
    assert torch.all(extended.bias[4 : 4 + extra] == NEW_ROW_BIAS)
