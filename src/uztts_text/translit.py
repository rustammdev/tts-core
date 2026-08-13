from __future__ import annotations

from uztts_text.apostrophes import OKINA, TUTUQ

_SIMPLE = {
    "а": "a",
    "б": "b",
    "в": "v",
    "г": "g",
    "д": "d",
    "ё": "yo",
    "ж": "j",
    "з": "z",
    "и": "i",
    "й": "y",
    "к": "k",
    "л": "l",
    "м": "m",
    "н": "n",
    "о": "o",
    "п": "p",
    "р": "r",
    "с": "s",
    "т": "t",
    "у": "u",
    "ф": "f",
    "х": "x",
    "ц": "ts",
    "ч": "ch",
    "ш": "sh",
    "щ": "sh",
    "ъ": TUTUQ,
    "ь": "",
    "э": "e",
    "ю": "yu",
    "я": "ya",
    "ў": f"o{OKINA}",
    "қ": "q",
    "ғ": f"g{OKINA}",
    "ҳ": "h",
}

_VOWELS = set("аеёиоуэюяўАЕЁИОУЭЮЯЎ")
_CYRILLIC = set(_SIMPLE) | {"е"} | {ch.upper() for ch in _SIMPLE} | {"Е"}


def _convert_e(previous: str | None) -> str:
    if previous is None or not previous.isalpha() or previous in _VOWELS:
        return "ye"
    if previous in "ъьЪЬ":
        return "ye"
    return "e"


def _match_case(source: str, converted: str, next_char: str | None) -> str:
    if not source.isupper() or not converted:
        return converted
    if len(converted) > 1 and next_char is not None and next_char.isupper():
        return converted.upper()
    return converted[0].upper() + converted[1:]


def cyrillic_to_latin(text: str) -> str:
    pieces: list[str] = []
    for index, char in enumerate(text):
        if char not in _CYRILLIC:
            pieces.append(char)
            continue
        lower = char.lower()
        converted = (
            _convert_e(text[index - 1] if index else None)
            if lower == "е"
            else _SIMPLE[lower]
        )
        next_char = text[index + 1] if index + 1 < len(text) else None
        pieces.append(_match_case(char, converted, next_char))
    return "".join(pieces)
