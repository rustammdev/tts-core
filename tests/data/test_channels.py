from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from uztts_data.channels import (
    Channel,
    ChannelStat,
    ChannelStatus,
    Genre,
    merge_stats,
    read_stats,
    render_report,
    validate_registry,
)
from uztts_data.cli import app
from uztts_data.manifest import write_jsonl

runner = CliRunner()

FULL: dict[str, object] = {
    "channel_id": "ch_rizanova",
    "url": "https://www.youtube.com/@rizanova/videos",
    "name": "RizaNova",
    "genre": "conversation",
    "script": "latin",
    "est_quality": "clean",
    "status": "approved",
    "reject_reason": None,
    "notes": "suhbat, studiya sifati",
}


def make_channel(index: int, **overrides: object) -> Channel:
    payload: dict[str, object] = {
        "channel_id": f"ch_{index:03d}",
        "url": f"https://www.youtube.com/@kanal{index}/videos",
        "name": f"Kanal {index}",
        "genre": "conversation",
        "script": "latin",
        "est_quality": "clean",
        "status": "approved",
    }
    return Channel.model_validate(payload | overrides)


@dataclass
class FakeLister:
    durations: dict[str, list[float]]
    calls: list[str] = field(default_factory=list)

    def video_durations(self, url: str) -> list[float]:
        self.calls.append(url)
        if url not in self.durations:
            raise RuntimeError("network unreachable")
        return self.durations[url]


def test_full_channel_parses() -> None:
    channel = Channel.model_validate(FULL)
    assert channel.genre is Genre.CONVERSATION
    assert channel.status is ChannelStatus.APPROVED


def test_field_order_matches_contract() -> None:
    assert list(Channel.model_fields) == list(FULL)


def test_rejected_requires_reason() -> None:
    with pytest.raises(ValidationError):
        make_channel(1, status="rejected")


def test_reason_requires_rejected_status() -> None:
    with pytest.raises(ValidationError):
        make_channel(1, reject_reason="dublyaj")


def test_rejected_with_reason_parses() -> None:
    channel = make_channel(1, status="rejected", reject_reason="doimiy fon musiqa")
    assert channel.status is ChannelStatus.REJECTED


@pytest.mark.parametrize(
    "override",
    [
        {"url": "https://example.com/@kanal"},
        {"url": "not a url"},
        {"channel_id": "Ch_001"},
        {"genre": "podcast"},
        {"script": "arabic"},
        {"est_quality": "great"},
        {"status": "pending"},
    ],
)
def test_invalid_values_rejected(override: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        Channel.model_validate(FULL | override)


def test_validate_registry_flags_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "channels.jsonl"
    write_jsonl(path, [make_channel(1), make_channel(1)])
    report = validate_registry(path)
    assert not report.ok
    assert "duplicate channel_id" in report.issues[0].message


def test_merge_stats_fetches_active_and_skips_rejected() -> None:
    channels = [
        make_channel(1),
        make_channel(2, status="candidate"),
        make_channel(3, status="rejected", reject_reason="telefon yozuvlari"),
    ]
    lister = FakeLister({channels[0].url: [3600.0, 1800.0], channels[1].url: [7200.0]})
    stats, errors = merge_stats(channels, {}, lister)
    assert not errors
    assert [s.channel_id for s in stats] == ["ch_001", "ch_002"]
    assert stats[0].hours == 1.5
    assert stats[0].video_count == 2
    assert channels[2].url not in lister.calls


def test_merge_stats_reuses_existing_without_network() -> None:
    channel = make_channel(1)
    existing = {"ch_001": ChannelStat(channel_id="ch_001", video_count=5, hours=10.0)}
    lister = FakeLister({})
    stats, errors = merge_stats([channel], existing, lister)
    assert not errors
    assert stats == [existing["ch_001"]]
    assert not lister.calls


def test_merge_stats_refresh_refetches() -> None:
    channel = make_channel(1)
    existing = {"ch_001": ChannelStat(channel_id="ch_001", video_count=5, hours=10.0)}
    lister = FakeLister({channel.url: [3600.0]})
    stats, _ = merge_stats([channel], existing, lister, refresh=True)
    assert lister.calls == [channel.url]
    assert stats[0].hours == 1.0


def test_merge_stats_collects_errors_and_continues() -> None:
    channels = [make_channel(1), make_channel(2)]
    lister = FakeLister({channels[1].url: [3600.0]})
    stats, errors = merge_stats(channels, {}, lister)
    assert [s.channel_id for s in stats] == ["ch_002"]
    assert errors and "ch_001" in errors[0]


def test_render_report_shares_and_gate() -> None:
    channels = [make_channel(1), make_channel(2, genre="news")]
    stats = {
        "ch_001": ChannelStat(channel_id="ch_001", video_count=100, hours=750.0),
        "ch_002": ChannelStat(channel_id="ch_002", video_count=40, hours=250.0),
    }
    report = render_report(channels, stats)
    assert "conversation" in report
    assert "75.0%" in report
    assert "Gate-2: raw hours 1000.0 / 1000 — met" in report


def test_render_report_flags_shortfall_and_missing() -> None:
    channels = [make_channel(1), make_channel(2)]
    stats = {"ch_001": ChannelStat(channel_id="ch_001", video_count=10, hours=42.0)}
    report = render_report(channels, stats)
    assert "missing stats: 1 channel(s)" in report
    assert "not met" in report


def test_cli_channels_validate_ok(tmp_path: Path) -> None:
    path = tmp_path / "channels.jsonl"
    write_jsonl(path, [make_channel(1), make_channel(2)])
    result = runner.invoke(app, ["channels", "validate", str(path)])
    assert result.exit_code == 0
    assert "2 channel(s) ok" in result.stdout


def test_cli_channels_validate_fails_on_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "channels.jsonl"
    write_jsonl(path, [make_channel(1), make_channel(1)])
    assert runner.invoke(app, ["channels", "validate", str(path)]).exit_code == 1


def test_cli_channels_stats_offline_with_cached_stats(tmp_path: Path) -> None:
    registry = tmp_path / "channels.jsonl"
    write_jsonl(registry, [make_channel(1)])
    stats_path = tmp_path / "channel_stats.jsonl"
    write_jsonl(
        stats_path, [ChannelStat(channel_id="ch_001", video_count=12, hours=20.5)]
    )
    result = runner.invoke(
        app, ["channels", "stats", str(registry), "--out", str(stats_path)]
    )
    assert result.exit_code == 0
    assert "Gate-2" in result.stdout
    assert read_stats(stats_path)["ch_001"].hours == 20.5
