from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import uztts_data.cli as cli
from uztts_data.channels import ChannelStatus, Genre, Script, read_registry
from uztts_data.paths import DATA_ROOT_ENV
from uztts_data.tg import (
    TOKEN_ENV,
    ChannelPost,
    ResolvedChannel,
    ack_text,
    channel_url,
    extract_tags,
    genre_from_tags,
    intake,
    mint_channel_id,
    posts_from_updates,
    read_offset,
    write_offset,
)

runner = CliRunner()


class FakeResolver:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def resolve(self, url: str) -> ResolvedChannel:
        self.calls.append(url)
        if "broken" in url:
            raise ValueError("no metadata")
        return ResolvedChannel(
            url="https://www.youtube.com/@resolved/videos", name="Resolved"
        )


def make_update(
    update_id: int,
    text: str,
    entities: list[dict[str, Any]] | None = None,
    kind: str = "channel_post",
) -> dict[str, Any]:
    return {
        "update_id": update_id,
        kind: {
            "message_id": update_id * 10,
            "chat": {"id": -100123},
            "text": text,
            "entities": entities or [],
        },
    }


def make_post(text: str, links: tuple[str, ...]) -> ChannelPost:
    return ChannelPost(
        update_id=1, chat_id=-100123, message_id=10, text=text, links=links
    )


def test_posts_from_updates_extracts_links_and_skips_non_text() -> None:
    updates = [
        make_update(1, "https://www.youtube.com/@kunuz #yangiliklar"),
        make_update(
            2,
            "yashirin link",
            entities=[{"type": "text_link", "url": "https://youtu.be/dQw4w9WgXcQ"}],
        ),
        make_update(3, "https://example.com/not-youtube"),
        {"update_id": 4, "channel_post": {"chat": {"id": 1}, "photo": []}},
        {"update_id": 5, "edited_channel_post": {"text": "x"}},
    ]
    posts = posts_from_updates(updates)
    assert len(posts) == 4
    assert posts[0].links == ("https://www.youtube.com/@kunuz",)
    assert posts[0].chat_id == -100123
    assert posts[1].links == ("https://youtu.be/dQw4w9WgXcQ",)
    assert posts[2].links == ()


def test_extract_tags_normalizes() -> None:
    assert extract_tags("#Podkast #ta'lim #vlog #vlog matn") == [
        "podkast",
        "talim",
        "vlog",
    ]


def test_genre_from_tags() -> None:
    assert genre_from_tags(["podkast"]) is Genre.CONVERSATION
    assert genre_from_tags(["talim"]) is Genre.EDUCATION
    assert genre_from_tags(["sayohat"]) is Genre.VLOG
    assert genre_from_tags(["yangiliklar"]) is Genre.NEWS
    assert genre_from_tags(["hikoya"]) is Genre.AUDIOBOOK
    assert genre_from_tags(["nimadir"]) is Genre.OTHER
    assert genre_from_tags([]) is Genre.OTHER


@pytest.mark.parametrize(
    ("link", "expected"),
    [
        ("https://www.youtube.com/@KunUz", "https://www.youtube.com/@KunUz/videos"),
        (
            "https://youtube.com/@KunUz/videos",
            "https://www.youtube.com/@KunUz/videos",
        ),
        (
            "https://www.youtube.com/channel/UCxxxx",
            "https://www.youtube.com/channel/UCxxxx/videos",
        ),
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", None),
        ("https://youtu.be/dQw4w9WgXcQ", None),
        ("https://www.youtube.com/", None),
    ],
)
def test_channel_url(link: str, expected: str | None) -> None:
    assert channel_url(link) == expected


def test_mint_channel_id() -> None:
    assert mint_channel_id("https://www.youtube.com/@KunUz/videos") == "ch_kunuz"
    assert (
        mint_channel_id("https://www.youtube.com/channel/UCxxxx/videos") == "ch_ucxxxx"
    )
    minted = mint_channel_id("https://www.youtube.com/@Ózbek/videos")
    assert minted.startswith("ch_")


def test_intake_adds_candidates_and_dedupes(tmp_path: Path) -> None:
    registry = tmp_path / "channels.jsonl"
    registry.write_text("", encoding="utf-8")
    resolver = FakeResolver()
    posts = [
        make_post(
            "#podkast kocha tilida",
            ("https://www.youtube.com/@Suhbatlar",),
        ),
        make_post("#yangiliklar", ("https://www.youtube.com/@KunUz",)),
        make_post("takror", ("https://youtube.com/@kunuz/videos",)),
        make_post("video link", ("https://youtu.be/dQw4w9WgXcQ",)),
        make_post("sinadi", ("https://youtu.be/broken000",)),
    ]

    result = intake(posts, registry, resolver)

    assert [c.channel_id for c in result.added] == [
        "ch_suhbatlar",
        "ch_kunuz",
        "ch_resolved",
    ]
    assert result.skipped == ("ch_kunuz",)
    assert len(result.errors) == 1
    assert resolver.calls == [
        "https://youtu.be/dQw4w9WgXcQ",
        "https://youtu.be/broken000",
    ]

    saved = list(read_registry(registry))
    assert len(saved) == 3
    first = saved[0]
    assert first.status is ChannelStatus.CANDIDATE
    assert first.genre is Genre.CONVERSATION
    assert first.script is Script.LATIN
    assert first.notes == "#podkast"
    assert saved[2].name == "Resolved"


def test_intake_keeps_existing_rows(tmp_path: Path) -> None:
    registry = tmp_path / "channels.jsonl"
    resolver = FakeResolver()
    intake(
        [make_post("#vlog", ("https://www.youtube.com/@birinchi",))],
        registry,
        resolver,
    )
    intake(
        [make_post("#talim", ("https://www.youtube.com/@ikkinchi",))],
        registry,
        resolver,
    )
    saved = list(read_registry(registry))
    assert [c.channel_id for c in saved] == ["ch_birinchi", "ch_ikkinchi"]


def test_intake_marks_cyrillic_names(tmp_path: Path) -> None:
    registry = tmp_path / "channels.jsonl"

    class CyrillicResolver:
        def resolve(self, url: str) -> ResolvedChannel:
            return ResolvedChannel(
                url="https://www.youtube.com/@tarix/videos", name="Тарих дарслари"
            )

    result = intake(
        [make_post("#maruza", ("https://youtu.be/dQw4w9WgXcQ",))],
        registry,
        CyrillicResolver(),
    )
    assert result.added[0].script is Script.CYRILLIC


def test_ack_text_lists_channels_with_tags(tmp_path: Path) -> None:
    registry = tmp_path / "channels.jsonl"
    result = intake(
        [make_post("#yangiliklar", ("https://www.youtube.com/@KunUz",))],
        registry,
        FakeResolver(),
    )
    text = ack_text(result.added)
    assert "1 kanal" in text
    assert "ch_kunuz" in text
    assert "#yangiliklar" in text
    assert "#uztts" in text


def test_offset_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "tg_offset"
    assert read_offset(path) is None
    write_offset(path, 42)
    assert read_offset(path) == 42


class FakeBot:
    def __init__(self, updates: list[dict[str, Any]]) -> None:
        self.updates = updates
        self.sent: list[tuple[int, str]] = []
        self.offsets: list[int | None] = []

    def get_updates(
        self, offset: int | None = None, timeout: int = 0
    ) -> list[dict[str, Any]]:
        self.offsets.append(offset)
        return self.updates

    def send_message(self, chat_id: int, text: str) -> None:
        self.sent.append((chat_id, text))


def test_cli_tg_pull(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path))
    monkeypatch.setenv(TOKEN_ENV, "secret")
    registry = tmp_path / "channels.jsonl"
    registry.write_text("", encoding="utf-8")

    bot = FakeBot([make_update(7, "#podkast https://www.youtube.com/@Suhbatlar")])
    monkeypatch.setattr(cli, "TelegramBot", lambda token: bot)
    monkeypatch.setattr(cli, "YtDlpResolver", FakeResolver)

    result = runner.invoke(cli.app, ["tg", "pull", str(registry)])

    assert result.exit_code == 0
    assert "added: ch_suhbatlar" in result.stdout
    assert "posts=1 added=1 skipped=0 failed=0" in result.stdout
    assert read_offset(tmp_path / "tg_offset") == 8
    assert read_offset(tmp_path / "tg_chat_id") == -100123
    assert len(bot.sent) == 1
    assert bot.sent[0][0] == -100123
    assert list(read_registry(registry))


def test_cli_tg_pull_requires_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    registry = tmp_path / "channels.jsonl"
    registry.write_text("", encoding="utf-8")
    result = runner.invoke(cli.app, ["tg", "pull", str(registry)])
    assert result.exit_code == 2
