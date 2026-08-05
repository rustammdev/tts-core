from __future__ import annotations

import json
import shutil
import subprocess
import wave
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from tests.data.test_channels import make_channel
from uztts_data.ingest import (
    DONE_MARKER,
    FAILURE_LOG,
    FILTERED_MARKER,
    DurationBounds,
    FetchedMedia,
    IngestError,
    Status,
    SubtitleKind,
    VideoMeta,
    ingest,
    ingest_channels,
    parse_metadata,
    read_urls,
    to_pcm_wav,
    video_id_from_url,
    watch_urls,
)

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not installed"
)

INFO: dict[str, Any] = {
    "id": "dQw4w9WgXcQ",
    "webpage_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "title": "O'zbek tili darsi",
    "channel": "Til va adabiyot",
    "duration": 742.5,
    "language": "uz",
    "upload_date": "20240115",
    "subtitles": {"uz": [{"ext": "vtt"}]},
    "automatic_captions": {"uz": [{"ext": "vtt"}], "en": [{"ext": "vtt"}]},
}


@dataclass
class FakeSource:
    info: dict[str, Any] = field(default_factory=lambda: dict(INFO))
    subtitles: bool = True
    failures: int = 0
    channel_videos: dict[str, list[str]] = field(default_factory=dict)
    metadata_calls: list[str] = field(default_factory=list)
    fetch_calls: list[str] = field(default_factory=list)
    list_calls: list[str] = field(default_factory=list)

    def metadata(self, url: str) -> Mapping[str, Any]:
        self.metadata_calls.append(url)
        if self.failures > 0:
            self.failures -= 1
            raise RuntimeError("network unreachable")
        info = dict(self.info)
        video_id = video_id_from_url(url)
        if video_id is not None:
            info["id"] = video_id
        return info

    def video_urls(self, url: str) -> list[str]:
        self.list_calls.append(url)
        if url not in self.channel_videos:
            raise RuntimeError("channel unreachable")
        return self.channel_videos[url]

    def fetch(self, url: str, destination: Path) -> FetchedMedia:
        self.fetch_calls.append(url)
        destination.mkdir(parents=True, exist_ok=True)
        audio = destination / "audio.wav"
        audio.write_bytes(b"RIFF")
        subs = None
        if self.subtitles:
            subs = destination / "subs.vtt"
            subs.write_text("WEBVTT\n", encoding="utf-8")
        return FetchedMedia(audio=audio, subtitles=subs)


def test_parse_metadata_maps_every_field() -> None:
    meta = parse_metadata(INFO, "https://youtu.be/dQw4w9WgXcQ")
    assert meta.video_id == "dQw4w9WgXcQ"
    assert meta.url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    assert meta.title == "O'zbek tili darsi"
    assert meta.channel == "Til va adabiyot"
    assert meta.duration == 742.5
    assert meta.language == "uz"
    assert meta.upload_date == "2024-01-15"


def test_parse_metadata_prefers_manual_subtitles() -> None:
    assert parse_metadata(INFO, "u").uz_subtitles is SubtitleKind.MANUAL


def test_parse_metadata_falls_back_to_automatic_subtitles() -> None:
    info = INFO | {"subtitles": {"en": []}}
    assert parse_metadata(info, "u").uz_subtitles is SubtitleKind.AUTOMATIC


def test_parse_metadata_matches_regional_uzbek_tag() -> None:
    info = INFO | {"subtitles": {"uz-UZ": []}, "automatic_captions": {}}
    assert parse_metadata(info, "u").uz_subtitles is SubtitleKind.MANUAL


def test_parse_metadata_without_uzbek_subtitles() -> None:
    info = INFO | {"subtitles": {"en": []}, "automatic_captions": {"ru": []}}
    assert parse_metadata(info, "u").uz_subtitles is None


def test_parse_metadata_falls_back_to_uploader() -> None:
    info = {key: value for key, value in INFO.items() if key != "channel"}
    assert parse_metadata(info | {"uploader": "Kanal"}, "u").channel == "Kanal"


def test_parse_metadata_tolerates_missing_optional_fields() -> None:
    meta = parse_metadata({"id": "abc", "duration": 12}, "https://example.test")
    assert meta.url == "https://example.test"
    assert meta.title == ""
    assert meta.channel is None
    assert meta.language is None
    assert meta.upload_date is None
    assert meta.uz_subtitles is None


def test_parse_metadata_rejects_malformed_upload_date() -> None:
    assert parse_metadata(INFO | {"upload_date": "2024"}, "u").upload_date is None


@pytest.mark.parametrize("missing", ["id", "duration"])
def test_parse_metadata_requires_id_and_duration(missing: str) -> None:
    info = {key: value for key, value in INFO.items() if key != missing}
    with pytest.raises(IngestError):
        parse_metadata(info, "https://example.test")


@pytest.mark.parametrize(
    "url",
    [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://m.youtube.com/watch?v=dQw4w9WgXcQ&list=PL123",
        "https://youtu.be/dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ?t=30",
        "https://www.youtube.com/shorts/dQw4w9WgXcQ",
        "https://www.youtube.com/embed/dQw4w9WgXcQ",
        "https://www.youtube.com/live/dQw4w9WgXcQ",
    ],
)
def test_video_id_from_url(url: str) -> None:
    assert video_id_from_url(url) == "dQw4w9WgXcQ"


@pytest.mark.parametrize(
    "url",
    [
        "https://www.youtube.com/watch?v=short",
        "https://www.youtube.com/@kanal",
        "https://www.youtube.com/playlist?list=PL123",
        "not a url",
    ],
)
def test_video_id_from_url_returns_none_when_unknown(url: str) -> None:
    assert video_id_from_url(url) is None


def test_read_urls_drops_blanks_comments_and_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "urls.txt"
    path.write_text(
        "\n# izoh\nhttps://youtu.be/dQw4w9WgXcQ\n  \nhttps://youtu.be/dQw4w9WgXcQ\n"
        "https://youtu.be/aQw4w9WgXcQ\n",
        encoding="utf-8",
    )
    assert read_urls(path) == [
        "https://youtu.be/dQw4w9WgXcQ",
        "https://youtu.be/aQw4w9WgXcQ",
    ]


def test_ingest_writes_expected_layout(tmp_path: Path) -> None:
    source = FakeSource()
    outcomes = ingest(["https://youtu.be/dQw4w9WgXcQ"], tmp_path, source)
    assert [outcome.status for outcome in outcomes] == [Status.INGESTED]

    video_dir = tmp_path / "dQw4w9WgXcQ"
    assert sorted(path.name for path in video_dir.iterdir()) == [
        DONE_MARKER,
        "audio.wav",
        "meta.json",
        "subs.vtt",
    ]
    meta = VideoMeta.model_validate_json(
        (video_dir / "meta.json").read_text(encoding="utf-8")
    )
    assert meta.video_id == "dQw4w9WgXcQ"
    assert meta.title == "O'zbek tili darsi"


def test_ingest_without_subtitles_writes_no_vtt(tmp_path: Path) -> None:
    ingest(["https://youtu.be/dQw4w9WgXcQ"], tmp_path, FakeSource(subtitles=False))
    assert not (tmp_path / "dQw4w9WgXcQ" / "subs.vtt").exists()


def test_ingest_skips_completed_video_without_network(tmp_path: Path) -> None:
    url = "https://youtu.be/dQw4w9WgXcQ"
    source = FakeSource()
    ingest([url], tmp_path, source)
    outcomes = ingest([url], tmp_path, source)
    assert [outcome.status for outcome in outcomes] == [Status.SKIPPED]
    assert len(source.metadata_calls) == 1
    assert len(source.fetch_calls) == 1


def test_ingest_redoes_video_without_done_marker(tmp_path: Path) -> None:
    url = "https://youtu.be/dQw4w9WgXcQ"
    source = FakeSource()
    ingest([url], tmp_path, source)
    (tmp_path / "dQw4w9WgXcQ" / DONE_MARKER).unlink()
    assert ingest([url], tmp_path, source)[0].status is Status.INGESTED
    assert len(source.fetch_calls) == 2


def test_ingest_retries_transient_failures(tmp_path: Path) -> None:
    source = FakeSource(failures=2)
    outcomes = ingest(["https://youtu.be/dQw4w9WgXcQ"], tmp_path, source)
    assert outcomes[0].status is Status.INGESTED
    assert len(source.metadata_calls) == 3


def test_ingest_gives_up_after_three_attempts(tmp_path: Path) -> None:
    source = FakeSource(failures=99)
    outcomes = ingest(["https://youtu.be/dQw4w9WgXcQ"], tmp_path, source)
    assert outcomes[0].status is Status.FAILED
    assert len(source.metadata_calls) == 3
    assert not (tmp_path / "dQw4w9WgXcQ" / DONE_MARKER).exists()


def test_ingest_records_failure_and_continues(tmp_path: Path) -> None:
    class HalfBrokenSource(FakeSource):
        def metadata(self, url: str) -> Mapping[str, Any]:
            if "broken" in url:
                raise RuntimeError("private video")
            return super().metadata(url)

    urls = ["https://youtu.be/brokenWgXcQ", "https://youtu.be/dQw4w9WgXcQ"]
    outcomes = ingest(urls, tmp_path, HalfBrokenSource())
    assert [outcome.status for outcome in outcomes] == [
        Status.FAILED,
        Status.INGESTED,
    ]

    failures = [
        json.loads(line)
        for line in (tmp_path / FAILURE_LOG).read_text(encoding="utf-8").splitlines()
    ]
    assert len(failures) == 1
    assert failures[0]["video_id"] == "brokenWgXcQ"
    assert "private video" in failures[0]["error"]


def test_watch_urls_flattens_nested_entries_and_dedupes() -> None:
    info: dict[str, Any] = {
        "entries": [
            {"id": "dQw4w9WgXcQ"},
            {"entries": [{"id": "aQw4w9WgXcQ"}, {"id": "dQw4w9WgXcQ"}]},
            {"id": "PLnotavideo123456"},
            "junk",
        ]
    }
    assert watch_urls(info) == [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://www.youtube.com/watch?v=aQw4w9WgXcQ",
    ]


def test_ingest_channels_nests_videos_under_channel_id(tmp_path: Path) -> None:
    channel = make_channel(1)
    candidate = make_channel(2, status="candidate")
    source = FakeSource(
        channel_videos={
            channel.url: [
                "https://youtu.be/dQw4w9WgXcQ",
                "https://youtu.be/aQw4w9WgXcQ",
            ]
        }
    )
    outcomes = ingest_channels([channel, candidate], tmp_path, source)
    assert [outcome.status for outcome in outcomes] == [Status.INGESTED] * 2
    assert source.list_calls == [channel.url]

    meta = VideoMeta.model_validate_json(
        (tmp_path / "ch_001" / "dQw4w9WgXcQ" / "meta.json").read_text(encoding="utf-8")
    )
    assert meta.channel_id == "ch_001"
    assert (tmp_path / "ch_001" / "aQw4w9WgXcQ" / DONE_MARKER).is_file()


def test_ingest_channels_logs_listing_failure_and_continues(tmp_path: Path) -> None:
    broken = make_channel(1)
    working = make_channel(2)
    source = FakeSource(channel_videos={working.url: ["https://youtu.be/dQw4w9WgXcQ"]})
    outcomes = ingest_channels([broken, working], tmp_path, source)
    assert [outcome.status for outcome in outcomes] == [
        Status.FAILED,
        Status.INGESTED,
    ]
    assert (tmp_path / "ch_001" / FAILURE_LOG).is_file()
    assert source.list_calls.count(broken.url) == 3


def test_too_long_video_is_filtered_without_download(tmp_path: Path) -> None:
    source = FakeSource()
    bounds = DurationBounds(min_seconds=60.0, max_seconds=600.0)
    url = "https://youtu.be/dQw4w9WgXcQ"
    outcomes = ingest([url], tmp_path, source, "ch_001", bounds)
    assert outcomes[0].status is Status.FILTERED
    assert outcomes[0].error is not None and "max" in outcomes[0].error

    video_dir = tmp_path / "dQw4w9WgXcQ"
    assert (video_dir / FILTERED_MARKER).is_file()
    assert not source.fetch_calls

    rerun = ingest([url], tmp_path, source, "ch_001", bounds)
    assert rerun[0].status is Status.SKIPPED
    assert len(source.metadata_calls) == 1


def test_too_short_video_is_filtered(tmp_path: Path) -> None:
    source = FakeSource(info=dict(INFO) | {"duration": 12.0})
    bounds = DurationBounds(min_seconds=60.0)
    outcomes = ingest(["https://youtu.be/dQw4w9WgXcQ"], tmp_path, source, None, bounds)
    assert outcomes[0].status is Status.FILTERED
    assert outcomes[0].error is not None and "min" in outcomes[0].error


def test_to_pcm_wav_reports_missing_ffmpeg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(shutil, "which", lambda _: None)
    with pytest.raises(IngestError, match="ffmpeg not found"):
        to_pcm_wav(tmp_path / "in.m4a", tmp_path / "out.wav", 24000)


@needs_ffmpeg
def test_to_pcm_wav_produces_mono_16bit_at_target_rate(tmp_path: Path) -> None:
    stereo = tmp_path / "source.flac"
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1",
            "-ar",
            "48000",
            "-ac",
            "2",
            str(stereo),
        ],
        check=True,
    )
    target = tmp_path / "audio.wav"
    to_pcm_wav(stereo, target, 24000)

    with wave.open(str(target)) as handle:
        assert handle.getframerate() == 24000
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert handle.getnframes() == pytest.approx(24000, rel=0.05)


@needs_ffmpeg
def test_to_pcm_wav_reports_broken_input(tmp_path: Path) -> None:
    broken = tmp_path / "source.m4a"
    broken.write_bytes(b"not audio")
    with pytest.raises(IngestError, match="ffmpeg failed"):
        to_pcm_wav(broken, tmp_path / "audio.wav", 24000)
