import pytest

from uztts_text import (
    cyrillic_to_latin,
    expand_numbers,
    int_to_words,
    normalize_apostrophes,
)


def test_apostrophe_variants_after_o_g_become_okina() -> None:
    assert normalize_apostrophes("o'g`il oʻzi g’ayrat") == "oʻgʻil oʻzi gʻayrat"


def test_apostrophe_elsewhere_becomes_tutuq() -> None:
    assert normalize_apostrophes("a'lo s’ana") == "aʼlo sʼana"


def test_cyrillic_uppercase_digraph_before_uppercase() -> None:
    assert cyrillic_to_latin("ШУМ") == "SHUM"
    assert cyrillic_to_latin("Шум") == "Shum"


def test_cyrillic_e_word_initial_and_after_vowel() -> None:
    assert cyrillic_to_latin("Ел елга ишонди") == "Yel yelga ishondi"
    assert cyrillic_to_latin("дуел") == "duyel"


def test_cyrillic_passthrough_latin() -> None:
    assert cyrillic_to_latin("salom дунё") == "salom dunyo"


def test_int_to_words_scales() -> None:
    assert int_to_words(214000513) == "ikki yuz oʻn toʻrt million besh yuz oʻn uch"
    assert int_to_words(-7) == "minus yetti"


def test_int_to_words_too_large() -> None:
    with pytest.raises(ValueError, match="too large"):
        int_to_words(10**19)


def test_expand_numbers_separators_and_percent() -> None:
    assert expand_numbers("12 va 15%") == "oʻn ikki va oʻn besh foiz"
    assert expand_numbers("5 2026") == "besh ikki ming yigirma olti"
