from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from html import escape

from uztts_data.channels import (
    RAW_HOURS_TARGET,
    Channel,
    ChannelStat,
    ChannelStatus,
    Genre,
)
from uztts_data.schema import QualityTag

PILOT_HOURS_TARGET = 50.0

GENRE_LABELS = {
    Genre.CONVERSATION: "suhbat / podkast",
    Genre.NEWS: "yangiliklar / hujjatli",
    Genre.EDUCATION: "ta'lim",
    Genre.VLOG: "vlog",
    Genre.AUDIOBOOK: "audiokitob / hikoya",
    Genre.OTHER: "boshqa",
}

GENRE_TARGET_SHARES = {
    Genre.CONVERSATION: (40.0, 50.0),
    Genre.EDUCATION: (15.0, 25.0),
    Genre.VLOG: (10.0, 15.0),
    Genre.NEWS: (5.0, 15.0),
    Genre.AUDIOBOOK: (5.0, 10.0),
    Genre.OTHER: (0.0, 5.0),
}

STATUS_LABELS = {
    ChannelStatus.CANDIDATE: "nomzod",
    ChannelStatus.APPROVED: "tasdiqlangan",
    ChannelStatus.REJECTED: "rad etilgan",
}

QUALITY_LABELS = {
    QualityTag.CLEAN: "toza",
    QualityTag.MEDIUM: "o'rtacha",
    QualityTag.NOISY: "shovqinli",
}


@dataclass(frozen=True, slots=True)
class ReportData:
    channels: tuple[Channel, ...]
    stats: Mapping[str, ChannelStat]
    ingested_hours: Mapping[str, float]
    generated_at: str
    speech_hours: Mapping[str, float] = field(default_factory=dict)


def build_report(data: ReportData, target_hours: float = RAW_HOURS_TARGET) -> str:
    body = "\n".join(
        [
            _summary_section(data, target_hours),
            _genre_section(data),
            _channels_section(data),
            _howto_section(),
        ]
    )
    return _page(data.generated_at, body)


def _active(data: ReportData) -> list[Channel]:
    return [c for c in data.channels if c.status is not ChannelStatus.REJECTED]


def _available_hours(data: ReportData) -> float:
    return sum(
        data.stats[c.channel_id].hours
        for c in _active(data)
        if c.channel_id in data.stats
    )


def _summary_section(data: ReportData, target_hours: float) -> str:
    counts = dict.fromkeys(ChannelStatus, 0)
    for channel in data.channels:
        counts[channel.status] += 1
    available = _available_hours(data)
    ingested = sum(data.ingested_hours.values())
    missing = sum(1 for c in _active(data) if c.channel_id not in data.stats)

    rows = [
        _row(
            "Kanallar",
            f"{len(data.channels)} ta &mdash; "
            f"{counts[ChannelStatus.CANDIDATE]} nomzod, "
            f"{counts[ChannelStatus.APPROVED]} tasdiqlangan, "
            f"{counts[ChannelStatus.REJECTED]} rad etilgan",
        ),
        _row(
            "Mavjud xom audio",
            f"{available:,.1f} soat"
            + (f" ({missing} kanal statistikasi yo'q)" if missing else ""),
        ),
        _row(
            f"Gate-2 ({target_hours:,.0f} soat)",
            _bar(available, target_hours),
        ),
        _row(
            f"Yuklab olingan (pilot {PILOT_HOURS_TARGET:,.0f} soat)",
            _bar(ingested, PILOT_HOURS_TARGET),
        ),
    ]
    speech = sum(data.speech_hours.values())
    if speech:
        share = speech / ingested * 100 if ingested else 0.0
        rows.append(
            _row(
                "Nutq (segment kesgani)",
                f"{speech:,.1f} soat &mdash; yuklanganning {share:.0f}%",
            )
        )
    return _section("Umumiy holat", f'<table class="kv">{"".join(rows)}</table>')


def _genre_section(data: ReportData) -> str:
    active = _active(data)
    total = _available_hours(data)
    body_rows = []
    for genre in Genre:
        members = [c for c in active if c.genre is genre]
        if not members:
            continue
        hours = sum(
            data.stats[c.channel_id].hours
            for c in members
            if c.channel_id in data.stats
        )
        share = hours / total * 100 if total else 0.0
        low, high = GENRE_TARGET_SHARES[genre]
        if share < low:
            verdict = "kam"
        elif share > high:
            verdict = "ko'p"
        else:
            verdict = "maqsadda"
        body_rows.append(
            "<tr>"
            f"<td>{escape(GENRE_LABELS[genre])}</td>"
            f'<td class="num">{len(members)}</td>'
            f'<td class="num">{hours:,.1f}</td>'
            f'<td class="num">{share:.1f}%</td>'
            f'<td class="num">{low:.0f}&ndash;{high:.0f}%</td>'
            f'<td class="verdict-{_verdict_class(verdict)}">{verdict}</td>'
            "</tr>"
        )
    table = (
        '<table class="wikitable sortable">'
        "<thead><tr><th>janr</th><th>kanallar</th><th>soat</th>"
        "<th>ulush</th><th>maqsad</th><th>holat</th></tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody></table>"
        '<p class="note">Ulushlar mavjud xom soatga nisbatan; maqsad '
        "oralig'i <em>docs/kanal-tanlash.md</em> dagi aralashma.</p>"
    )
    return _section("Janr aralashmasi", table)


def _channels_section(data: ReportData) -> str:
    body_rows = []
    for channel in data.channels:
        stat = data.stats.get(channel.channel_id)
        ingested = data.ingested_hours.get(channel.channel_id, 0.0)
        speech = data.speech_hours.get(channel.channel_id, 0.0)
        note = channel.reject_reason or channel.notes or ""
        body_rows.append(
            "<tr>"
            f"<td><code>{escape(channel.channel_id)}</code></td>"
            f'<td><a href="{escape(channel.url, quote=True)}">'
            f"{escape(channel.name)}</a></td>"
            f"<td>{escape(GENRE_LABELS[channel.genre])}</td>"
            f'<td class="status-{channel.status.value}">'
            f"{STATUS_LABELS[channel.status]}</td>"
            f"<td>{QUALITY_LABELS[channel.est_quality]}</td>"
            f'<td class="num">{stat.video_count if stat else "&mdash;"}</td>'
            f'<td class="num">{f"{stat.hours:,.1f}" if stat else "&mdash;"}</td>'
            f'<td class="num">{ingested:,.1f}</td>'
            f'<td class="num">{speech:,.1f}</td>'
            f"<td>{escape(note)}</td>"
            "</tr>"
        )
    table = (
        '<table class="wikitable sortable">'
        "<thead><tr><th>id</th><th>kanal</th><th>janr</th><th>holat</th>"
        "<th>sifat (taxmin)</th><th>video</th><th>soat</th>"
        "<th>yuklangan</th><th>nutq</th><th>izoh</th></tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody></table>"
        '<p class="note">Ustun sarlavhasini bosib saralang. '
        "Kanal nomi YouTube'dagi videolar sahifasiga olib boradi &mdash; "
        "nomzodni baholash uchun 2&ndash;3 videoni 30 soniyadan tinglang.</p>"
    )
    return _section("Kanallar", table)


def _howto_section() -> str:
    items = (
        "<li>Yangi kanal qo'shish: linkni Telegram kanaliga tashlang "
        "(teg bilan: <code>#podkast</code> <code>#talim</code> "
        "<code>#vlog</code> <code>#yangiliklar</code> <code>#hikoya</code>), "
        "so'ng <code>make tg-pull</code>.</li>"
        "<li>Soatlarni yangilash: <code>uztts-data channels stats</code> "
        "(yangi kanallar uchun; <code>--refresh</code> keshni yangilaydi).</li>"
        "<li>Shu sahifani yangilash: <code>make report</code> "
        "(sahifa har 60 soniyada o'zi qayta yuklanadi).</li>"
        "<li>Yuklab olish: <code>uztts-ingest --channels "
        "configs/channels.jsonl</code> &mdash; faqat "
        "<em>tasdiqlangan</em> kanallarni oladi.</li>"
        "<li>Qaror manbalari: <code>configs/channels.jsonl</code> (registr), "
        "<code>docs/kanal-tanlash.md</code> (baholash mezoni).</li>"
    )
    return _section("Boshqaruv", f"<ul>{items}</ul>")


def _verdict_class(verdict: str) -> str:
    return {"maqsadda": "ok", "kam": "low", "ko'p": "high"}[verdict]


def _row(label: str, value: str) -> str:
    return f"<tr><th>{label}</th><td>{value}</td></tr>"


def _bar(value: float, target: float) -> str:
    percent = min(value / target * 100, 100.0) if target else 0.0
    return (
        f'<span class="bar"><i style="width:{percent:.1f}%"></i></span> '
        f"{value:,.1f} / {target:,.0f} soat ({value / target * 100:.0f}%)"
    )


def _section(title: str, body: str) -> str:
    return f"<h2>{escape(title)}</h2>\n{body}"


_CSS = """
body { margin: 0; background: #fff; color: #202122;
  font: 14px/1.5 -apple-system, "Segoe UI", "Helvetica Neue", Arial, sans-serif; }
.page { max-width: 1080px; margin: 0 auto; padding: 12px 24px 48px; }
h1 { font-family: Georgia, "Times New Roman", serif; font-weight: normal;
  font-size: 28px; margin: 8px 0 2px; border-bottom: 1px solid #a2a9b1;
  padding-bottom: 4px; }
h2 { font-family: Georgia, "Times New Roman", serif; font-weight: normal;
  font-size: 21px; margin: 28px 0 8px; border-bottom: 1px solid #eaecf0;
  padding-bottom: 3px; }
.subtitle { color: #54595d; font-size: 13px; margin: 0 0 16px; }
a { color: #0645ad; text-decoration: none; }
a:hover { text-decoration: underline; }
code { font-family: ui-monospace, Consolas, monospace; font-size: 13px;
  background: #f8f9fa; padding: 0 3px; }
table.wikitable { border-collapse: collapse; width: 100%;
  background: #f8f9fa; border: 1px solid #a2a9b1; }
.wikitable th, .wikitable td { border: 1px solid #a2a9b1;
  padding: 4px 8px; text-align: left; vertical-align: top; }
.wikitable th { background: #eaecf0; cursor: pointer; user-select: none;
  white-space: nowrap; }
.wikitable th[data-dir="asc"]::after { content: " \\2191"; }
.wikitable th[data-dir="desc"]::after { content: " \\2193"; }
.wikitable tbody tr:hover td { background: #eaf3ff; }
td.num { text-align: right; font-variant-numeric: tabular-nums;
  white-space: nowrap; }
table.kv { border-collapse: collapse; }
table.kv th { text-align: left; padding: 3px 24px 3px 0; font-weight: 600;
  white-space: nowrap; vertical-align: top; }
table.kv td { padding: 3px 0; }
.bar { display: inline-block; width: 240px; height: 12px;
  background: #eaecf0; vertical-align: -1px; }
.bar i { display: block; height: 100%; background: #36c; }
td.status-candidate { background: #eaecf0; }
td.status-approved { background: #d5fdd5; }
td.status-rejected { background: #fee7e6; }
td.verdict-ok { background: #d5fdd5; }
td.verdict-low { background: #fef6e7; }
td.verdict-high { background: #fee7e6; }
.note { color: #54595d; font-size: 13px; }
ul { margin: 4px 0; padding-left: 20px; }
li { margin: 3px 0; }
"""

_JS = """
for (const table of document.querySelectorAll("table.sortable")) {
  const headers = table.querySelectorAll("th");
  headers.forEach((header, index) => header.addEventListener("click", () => {
    const body = table.tBodies[0];
    const rows = Array.from(body.rows);
    const direction = header.dataset.dir === "desc" ? 1 : -1;
    headers.forEach(h => delete h.dataset.dir);
    header.dataset.dir = direction === 1 ? "asc" : "desc";
    const text = row => row.cells[index].innerText.trim();
    const num = value => parseFloat(value.replace(/[^0-9.,-]/g, "")
      .replace(",", ""));
    rows.sort((a, b) => {
      const x = text(a), y = text(b);
      const nx = num(x), ny = num(y);
      if (!isNaN(nx) && !isNaN(ny)) return direction * (nx - ny);
      return direction * x.localeCompare(y);
    });
    rows.forEach(row => body.appendChild(row));
  }));
}
"""


def _page(generated_at: str, body: str) -> str:
    return (
        "<!doctype html>\n"
        '<html lang="uz">\n<head>\n<meta charset="utf-8">\n'
        '<meta http-equiv="refresh" content="60">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        "<title>uz-tts — data hisoboti</title>\n"
        f"<style>{_CSS}</style>\n</head>\n<body>\n"
        '<div class="page">\n'
        "<h1>uz-tts — data hisoboti</h1>\n"
        f'<p class="subtitle">Yangilangan: {escape(generated_at)} &middot; '
        "manba: <code>configs/channels.jsonl</code> + "
        "<code>manifests/channel_stats.jsonl</code> + "
        "<code>manifests/raw.jsonl</code></p>\n"
        f"{body}\n"
        f"</div>\n<script>{_JS}</script>\n</body>\n</html>\n"
    )
