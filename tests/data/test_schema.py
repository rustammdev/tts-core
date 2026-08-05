from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from uztts_data import License, QualityTag, Segment

FULL: dict[str, object] = {
    "id": "ch_rizanova_000123",
    "audio_path": "data/processed/ch_rizanova/000123.wav",
    "text": "Assalomu alaykum, xush kelibsiz.",
    "text_normalized": "assalomu alaykum xush kelibsiz",
    "speaker_id": "ch_rizanova_c0",
    "channel_id": "ch_rizanova",
    "duration": 3.42,
    "sample_rate": 24000,
    "quality_tag": "clean",
    "snr_db": 34.1,
    "separated": False,
    "source": "youtube",
    "license": "web_scraped",
    "style_caption": None,
    "asr_cer": 0.01,
    "asr_avg_logprob": -0.31,
    "asr_compression_ratio": 1.42,
    "lang_prob": 0.97,
}

MINIMAL: dict[str, object] = {
    "id": "spk001_000001",
    "audio_path": "data/raw/spk001/000001.wav",
    "speaker_id": "spk001",
    "duration": 1.5,
    "sample_rate": 24000,
    "source": "own_recording",
    "license": "owned",
}


def test_full_segment_parses() -> None:
    segment = Segment.model_validate(FULL)
    assert segment.quality_tag is QualityTag.CLEAN
    assert segment.license is License.WEB_SCRAPED
    assert segment.channel_id == "ch_rizanova"
    assert segment.audio_path == Path("data/processed/ch_rizanova/000123.wav")


def test_derived_fields_stay_unset_before_pipeline_fills_them() -> None:
    segment = Segment.model_validate(MINIMAL)
    assert segment.text is None
    assert segment.text_normalized is None
    assert segment.channel_id is None
    assert segment.quality_tag is None
    assert segment.snr_db is None
    assert segment.separated is False
    assert segment.style_caption is None
    assert segment.asr_cer is None
    assert segment.asr_avg_logprob is None
    assert segment.asr_compression_ratio is None
    assert segment.lang_prob is None


def test_field_order_matches_contract() -> None:
    assert list(Segment.model_fields) == list(FULL)


def test_unknown_field_rejected() -> None:
    with pytest.raises(ValidationError):
        Segment.model_validate(FULL | {"emotion": "happy"})


def test_segment_is_immutable() -> None:
    segment = Segment.model_validate(FULL)
    with pytest.raises(ValidationError):
        segment.duration = 1.0  # type: ignore[misc]


def test_windows_separators_normalised_on_read() -> None:
    segment = Segment.model_validate(
        MINIMAL | {"audio_path": r"data\processed\spk001\000001.wav"}
    )
    assert segment.audio_path.parts == ("data", "processed", "spk001", "000001.wav")


def test_audio_path_always_serialises_posix_style() -> None:
    segment = Segment.model_validate(
        MINIMAL | {"audio_path": r"data\processed\spk001\000001.wav"}
    )
    assert '"data/processed/spk001/000001.wav"' in segment.model_dump_json()


def test_text_is_stripped() -> None:
    segment = Segment.model_validate(MINIMAL | {"text": "  salom  "})
    assert segment.text == "salom"


def test_normalized_text_without_raw_text_rejected() -> None:
    with pytest.raises(ValidationError):
        Segment.model_validate(MINIMAL | {"text_normalized": "salom"})


def test_cer_above_one_allowed() -> None:
    assert Segment.model_validate(MINIMAL | {"asr_cer": 1.4}).asr_cer == 1.4


@pytest.mark.parametrize(
    "override",
    [
        {"id": "SPK001_000123"},
        {"id": "spk 001"},
        {"id": "_spk001"},
        {"speaker_id": ""},
        {"duration": 0.0},
        {"duration": -1.0},
        {"sample_rate": 0},
        {"asr_cer": -0.1},
        {"license": "cc-by-nc"},
        {"quality_tag": "perfect"},
        {"source": "   "},
        {"channel_id": "Ch_Rizanova"},
        {"lang_prob": 1.5},
        {"lang_prob": -0.1},
        {"asr_compression_ratio": 0.0},
    ],
)
def test_invalid_values_rejected(override: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        Segment.model_validate(FULL | override)
