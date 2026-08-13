from __future__ import annotations

from uztts_events.schema import EventLabel

AUDIOSET_CLASS_NAMES: dict[EventLabel, tuple[str, ...]] = {
    EventLabel.LAUGHTER: (
        "Laughter",
        "Baby laughter",
        "Giggle",
        "Snicker",
        "Belly laugh",
        "Chuckle, chortle",
    ),
    EventLabel.MUSIC: ("Music",),
    EventLabel.APPLAUSE: ("Applause", "Clapping"),
    EventLabel.COUGH: ("Cough",),
}


def resolve_class_ids(class_names: list[str]) -> dict[EventLabel, list[int]]:
    name_to_id = {name: index for index, name in enumerate(class_names)}
    missing = [
        name
        for names in AUDIOSET_CLASS_NAMES.values()
        for name in names
        if name not in name_to_id
    ]
    if missing:
        raise ValueError(f"class names not found in label set: {missing}")
    return {
        label: [name_to_id[name] for name in names]
        for label, names in AUDIOSET_CLASS_NAMES.items()
    }
