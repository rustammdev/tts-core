from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

from uztts_data.channels import Channel, ChannelStatus, Genre, Script, read_registry
from uztts_data.manifest import write_jsonl
from uztts_data.schema import QualityTag

TOKEN_ENV = "TELEGRAM_BOT_TOKEN"
OFFSET_FILENAME = "tg_offset"

_API_BASE = "https://api.telegram.org"
_LINK_RE = re.compile(r"https?://[^\s<>()\"']+")
_TAG_RE = re.compile(r"#([\w'ʼ’ʻ-]+)")  # noqa: RUF001

_TAG_GENRES = {
    "podkast": Genre.CONVERSATION,
    "podcast": Genre.CONVERSATION,
    "suhbat": Genre.CONVERSATION,
    "intervyu": Genre.CONVERSATION,
    "tokshou": Genre.CONVERSATION,
    "talim": Genre.EDUCATION,
    "dars": Genre.EDUCATION,
    "maruza": Genre.EDUCATION,
    "kurs": Genre.EDUCATION,
    "vlog": Genre.VLOG,
    "sayohat": Genre.VLOG,
    "kundalik": Genre.VLOG,
    "yangilik": Genre.NEWS,
    "yangiliklar": Genre.NEWS,
    "news": Genre.NEWS,
    "hikoya": Genre.AUDIOBOOK,
    "hikoyalar": Genre.AUDIOBOOK,
    "audiokitob": Genre.AUDIOBOOK,
    "ertak": Genre.AUDIOBOOK,
}

_GENRE_TAGS = {
    Genre.CONVERSATION: "#suhbat",
    Genre.NEWS: "#yangiliklar",
    Genre.EDUCATION: "#talim",
    Genre.VLOG: "#vlog",
    Genre.AUDIOBOOK: "#hikoya",
    Genre.OTHER: "#boshqa",
}


class TgError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class ChannelPost:
    update_id: int
    chat_id: int
    message_id: int
    text: str
    links: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ResolvedChannel:
    url: str
    name: str


class ChannelResolver(Protocol):
    def resolve(self, url: str) -> ResolvedChannel: ...


@dataclass(frozen=True, slots=True)
class IntakeResult:
    added: tuple[Channel, ...]
    skipped: tuple[str, ...]
    errors: tuple[str, ...]


class TelegramBot:
    def __init__(self, token: str, timeout: float = 60.0) -> None:
        self._token = token
        self._timeout = timeout

    def get_updates(
        self, offset: int | None = None, timeout: int = 0
    ) -> list[dict[str, Any]]:
        params: dict[str, object] = {
            "timeout": timeout,
            "allowed_updates": '["channel_post","message"]',
        }
        if offset is not None:
            params["offset"] = offset
        return cast(list[dict[str, Any]], self._call("getUpdates", params))

    def send_message(self, chat_id: int, text: str) -> None:
        self._call("sendMessage", {"chat_id": chat_id, "text": text})

    def _call(self, method: str, params: Mapping[str, object]) -> object:
        request = Request(
            f"{_API_BASE}/bot{self._token}/{method}",
            data=urlencode(params).encode(),
        )
        try:
            with urlopen(request, timeout=self._timeout) as response:
                payload: dict[str, Any] = json.load(response)
        except OSError as exc:
            raise TgError(f"telegram {method} failed: {exc}") from exc
        if not payload.get("ok"):
            raise TgError(f"telegram {method} failed: {payload.get('description')}")
        return payload["result"]


def posts_from_updates(updates: Iterable[Mapping[str, Any]]) -> list[ChannelPost]:
    posts: list[ChannelPost] = []
    for update in updates:
        message = update.get("channel_post") or update.get("message")
        if not isinstance(message, Mapping):
            continue
        text = str(message.get("text") or message.get("caption") or "")
        entities = message.get("entities") or message.get("caption_entities") or []
        chat = message.get("chat") or {}
        posts.append(
            ChannelPost(
                update_id=int(update.get("update_id", 0)),
                chat_id=int(chat.get("id", 0)),
                message_id=int(message.get("message_id", 0)),
                text=text,
                links=tuple(_youtube_links(text, entities)),
            )
        )
    return posts


def _youtube_links(text: str, entities: Iterable[Mapping[str, Any]]) -> list[str]:
    candidates = _LINK_RE.findall(text)
    for entity in entities:
        if entity.get("type") == "text_link" and entity.get("url"):
            candidates.append(str(entity["url"]))
    links: list[str] = []
    for candidate in candidates:
        cleaned = candidate.rstrip(".,;:!?")
        if _is_youtube(cleaned) and cleaned not in links:
            links.append(cleaned)
    return links


def _is_youtube(url: str) -> bool:
    host = urlparse(url).hostname or ""
    return host == "youtu.be" or host.endswith("youtube.com")


def extract_tags(text: str) -> list[str]:
    tags: list[str] = []
    for raw in _TAG_RE.findall(text):
        tag = re.sub(r"[^a-z0-9_]", "", raw.lower())
        if tag and tag not in tags:
            tags.append(tag)
    return tags


def genre_from_tags(tags: Iterable[str]) -> Genre:
    for tag in tags:
        if tag in _TAG_GENRES:
            return _TAG_GENRES[tag]
    return Genre.OTHER


def channel_url(link: str) -> str | None:
    parsed = urlparse(link)
    if (parsed.hostname or "") == "youtu.be":
        return None
    parts = [part for part in parsed.path.split("/") if part]
    if not parts:
        return None
    head = parts[0]
    if head.startswith("@"):
        return f"https://www.youtube.com/{head}/videos"
    if head in {"channel", "c", "user"} and len(parts) > 1:
        return f"https://www.youtube.com/{head}/{parts[1]}/videos"
    return None


def mint_channel_id(url: str) -> str:
    parts = [part for part in urlparse(url).path.split("/") if part]
    stem = parts[-2] if len(parts) >= 2 and parts[-1] == "videos" else parts[-1]
    slug = re.sub(r"[^a-z0-9_-]", "", stem.lower().lstrip("@"))
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", slug):
        slug = hashlib.sha1(url.encode()).hexdigest()[:8]
    return f"ch_{slug}"


def _display_name(url: str) -> str:
    parts = [part for part in urlparse(url).path.split("/") if part]
    stem = parts[-2] if len(parts) >= 2 and parts[-1] == "videos" else parts[-1]
    return stem.lstrip("@")


def _has_cyrillic(text: str) -> bool:
    return any("Ѐ" <= char <= "ӿ" for char in text)


def intake(
    posts: Sequence[ChannelPost],
    registry: Path,
    resolver: ChannelResolver,
) -> IntakeResult:
    existing = list(read_registry(registry)) if registry.is_file() else []
    known_ids = {channel.channel_id for channel in existing}
    known_urls = {channel.url for channel in existing}

    added: list[Channel] = []
    skipped: list[str] = []
    errors: list[str] = []
    for post in posts:
        tags = extract_tags(post.text)
        genre = genre_from_tags(tags)
        for link in post.links:
            url = channel_url(link)
            name: str | None = None
            if url is None:
                try:
                    resolved = resolver.resolve(link)
                except Exception as exc:
                    errors.append(f"{link}: {type(exc).__name__}: {exc}")
                    continue
                url, name = resolved.url, resolved.name
            channel_id = mint_channel_id(url)
            if channel_id in known_ids or url in known_urls:
                skipped.append(channel_id)
                continue
            if name is None:
                name = _display_name(url)
            channel = Channel(
                channel_id=channel_id,
                url=url,
                name=name,
                genre=genre,
                script=Script.CYRILLIC if _has_cyrillic(name) else Script.LATIN,
                est_quality=QualityTag.MEDIUM,
                status=ChannelStatus.CANDIDATE,
                notes=" ".join(f"#{tag}" for tag in tags) or None,
            )
            known_ids.add(channel_id)
            known_urls.add(url)
            added.append(channel)
    if added:
        write_jsonl(registry, existing + added)
    return IntakeResult(
        added=tuple(added), skipped=tuple(skipped), errors=tuple(errors)
    )


def ack_text(added: Sequence[Channel]) -> str:
    lines = [f"📥 Registrga qo'shildi: {len(added)} kanal"]
    lines.extend(
        f"• {channel.channel_id} — {channel.name} {_GENRE_TAGS[channel.genre]}"
        for channel in added
    )
    lines.append("#uztts #registr")
    return "\n".join(lines)


def read_offset(path: Path) -> int | None:
    if not path.is_file():
        return None
    content = path.read_text(encoding="utf-8").strip()
    return int(content) if content else None


def write_offset(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{value}\n", encoding="utf-8")


class YtDlpResolver:
    def resolve(self, url: str) -> ResolvedChannel:
        from yt_dlp import YoutubeDL

        options = {
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
            "skip_download": True,
            "extract_flat": True,
            "playlist_items": "1",
        }
        with YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=False)
        if info is None:
            raise TgError(f"yt-dlp returned no metadata for {url}")
        handle = info.get("uploader_id")
        channel_id = info.get("channel_id")
        if isinstance(handle, str) and handle.startswith("@"):
            resolved_url = f"https://www.youtube.com/{handle}/videos"
        elif isinstance(channel_id, str):
            resolved_url = f"https://www.youtube.com/channel/{channel_id}/videos"
        else:
            raise TgError(f"cannot resolve channel for {url}")
        name = info.get("channel") or info.get("uploader") or handle or channel_id
        return ResolvedChannel(url=resolved_url, name=str(name))
