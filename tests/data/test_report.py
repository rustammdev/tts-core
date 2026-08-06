from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from uztts_data.channels import Channel, ChannelStat, ChannelStatus, Genre
from uztts_data.cli import app
from uztts_data.manifest import write_jsonl
from uztts_data.paths import DATA_ROOT_ENV
from uztts_data.report import ReportData, build_report

from .test_channels import make_channel

runner = CliRunner()


def make_data(
    channels: tuple[Channel, ...],
    stats: dict[str, ChannelStat] | None = None,
    ingested: dict[str, float] | None = None,
) -> ReportData:
    return ReportData(
        channels=channels,
        stats=stats or {},
        ingested_hours=ingested or {},
        generated_at="2026-08-06 12:00",
    )


def test_report_lists_channels_with_links_and_stats() -> None:
    channel = make_channel(
        1, name="Kanal Bir", genre=Genre.NEWS, status=ChannelStatus.CANDIDATE
    )
    stat = ChannelStat(channel_id=channel.channel_id, video_count=12, hours=34.5)
    page = build_report(make_data((channel,), {channel.channel_id: stat}))

    assert f'href="{channel.url}"' in page
    assert "Kanal Bir" in page
    assert "34.5" in page
    assert "yangiliklar" in page
    assert "nomzod" in page


def test_report_marks_missing_stats_and_rejected() -> None:
    kept = make_channel(1)
    rejected = make_channel(
        2, status=ChannelStatus.REJECTED, reject_reason="doimiy fon musiqa"
    )
    page = build_report(make_data((kept, rejected)))

    assert "&mdash;" in page
    assert "1 kanal statistikasi yo'q" in page
    assert "rad etilgan" in page
    assert "doimiy fon musiqa" in page


def test_report_escapes_untrusted_text() -> None:
    channel = make_channel(1, name="<script>alert(1)</script>")
    page = build_report(make_data((channel,)))
    assert "<script>alert(1)" not in page
    assert "&lt;script&gt;" in page


def test_report_gate_progress_capped_at_full_bar() -> None:
    channel = make_channel(1)
    stat = ChannelStat(channel_id=channel.channel_id, video_count=9, hours=4000.0)
    page = build_report(make_data((channel,), {channel.channel_id: stat}))
    assert "width:100.0%" in page
    assert "(400%)" in page


def test_report_genre_share_verdicts() -> None:
    conversation = make_channel(1, genre=Genre.CONVERSATION)
    news = make_channel(2, genre=Genre.NEWS)
    stats = {
        conversation.channel_id: ChannelStat(
            channel_id=conversation.channel_id, video_count=1, hours=10.0
        ),
        news.channel_id: ChannelStat(
            channel_id=news.channel_id, video_count=1, hours=90.0
        ),
    }
    page = build_report(make_data((conversation, news), stats))
    assert "kam" in page
    assert "ko'p" in page


def test_report_shows_speech_hours_when_present() -> None:
    channel = make_channel(1)
    stat = ChannelStat(channel_id=channel.channel_id, video_count=4, hours=10.0)
    data = ReportData(
        channels=(channel,),
        stats={channel.channel_id: stat},
        ingested_hours={channel.channel_id: 6.0},
        generated_at="2026-08-06 12:00",
        speech_hours={channel.channel_id: 4.2},
    )
    page = build_report(data)
    assert "Nutq (segment kesgani)" in page
    assert "4.2" in page
    assert "yuklanganning 70%" in page


def test_cli_report_writes_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path))
    registry = tmp_path / "channels.jsonl"
    channel = make_channel(1)
    write_jsonl(registry, [channel])
    write_jsonl(
        tmp_path / "manifests" / "channel_stats.jsonl",
        [ChannelStat(channel_id=channel.channel_id, video_count=3, hours=5.0)],
    )
    out = tmp_path / "report" / "index.html"

    result = runner.invoke(app, ["report", str(registry), "--out", str(out)])

    assert result.exit_code == 0
    page = out.read_text(encoding="utf-8")
    assert channel.name in page
    assert "5.0" in page
