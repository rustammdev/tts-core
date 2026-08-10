from __future__ import annotations

from uztts_asr.postprocess import capitalize_sentences


def test_capitalize_sentences_after_punctuation() -> None:
    assert (
        capitalize_sentences("salom. qalaysiz? yaxshi! rahmat")
        == "Salom. Qalaysiz? Yaxshi! Rahmat"
    )


def test_capitalize_sentences_apostrophe_and_commas() -> None:
    assert (
        capitalize_sentences("o'zbek tili, davlat tili. a'lo darajada")
        == "O'zbek tili, davlat tili. A'lo darajada"
    )


def test_capitalize_sentences_plain_text_unchanged_midway() -> None:
    assert (
        capitalize_sentences("belgisiz matn shu holida") == "Belgisiz matn shu holida"
    )
    assert capitalize_sentences("") == ""
