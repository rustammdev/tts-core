INDEX_HTML = """<!doctype html>
<html lang="uz">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>UzSTT demo</title>
<style>
:root { color-scheme: light dark;
  --bg: #f6f7f9; --card: #ffffff; --text: #1a1d21; --muted: #667085;
  --accent: #2957d0; --tag: #b7791f; --border: #e0e3e8; }
@media (prefers-color-scheme: dark) { :root {
  --bg: #14161a; --card: #1e2126; --text: #e6e8eb; --muted: #98a2b3;
  --accent: #7ba0f4; --tag: #e5b567; --border: #32363d; } }
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text);
  font: 15px/1.55 system-ui, sans-serif; }
main { max-width: 860px; margin: 0 auto; padding: 24px 16px 64px; }
h1 { font-size: 22px; margin: 8px 0 2px; }
.sub { color: var(--muted); margin: 0 0 20px; }
.card { background: var(--card); border: 1px solid var(--border);
  border-radius: 10px; padding: 18px; margin-bottom: 16px; }
label { display: block; font-weight: 600; margin: 12px 0 4px; }
input[type=text], select { width: 100%; padding: 8px 10px; border-radius: 7px;
  border: 1px solid var(--border); background: var(--bg); color: var(--text); }
input[type=file] { margin-top: 4px; }
.row { display: flex; gap: 16px; flex-wrap: wrap; align-items: end; }
.row > div { flex: 1; min-width: 220px; }
.check { display: flex; gap: 8px; align-items: center; margin-top: 14px;
  font-weight: 600; }
button { margin-top: 16px; background: var(--accent); color: #fff; border: 0;
  padding: 10px 22px; border-radius: 8px; font-size: 15px; cursor: pointer; }
button:disabled { opacity: .5; cursor: wait; }
#status { margin-top: 12px; color: var(--muted); }
.stats { display: flex; gap: 18px; flex-wrap: wrap; color: var(--muted);
  font-size: 13px; margin-bottom: 10px; }
#fulltext { white-space: pre-wrap; }
#fulltext mark { background: none; color: var(--tag); font-weight: 700; }
table { width: 100%; border-collapse: collapse; font-size: 14px; }
td { padding: 6px 8px; border-top: 1px solid var(--border);
  vertical-align: top; }
td.t { color: var(--muted); white-space: nowrap; width: 1%; }
td.ev { color: var(--tag); font-weight: 600; }
.hidden { display: none; }
</style>
</head>
<body>
<main>
<h1>UzSTT demo</h1>
<p class="sub">Video/audio yuklang yoki YouTube havolasini bering — matn,
vaqt belgilari va audio hodisalar ([kulgu], [musiqa]).</p>

<div class="card">
  <div class="row">
    <div>
      <label>Fayl (video yoki audio)</label>
      <input type="file" id="file" accept="audio/*,video/*">
    </div>
    <div>
      <label>yoki YouTube havolasi</label>
      <input type="text" id="url" placeholder="https://youtube.com/watch?v=...">
    </div>
  </div>
  <div class="row">
    <div>
      <label>Model</label>
      <select id="model">
        <option value="uz-stt">UzSTT — bizniki (gemini_full_220m)</option>
        <option value="gigaam">GigaAM 220M — baza</option>
        <option value="gigaam-large">GigaAM-large 600M — baza</option>
      </select>
    </div>
    <div>
      <div class="check">
        <input type="checkbox" id="events" checked>
        <label for="events" style="margin:0">Hodisa teglari</label>
      </div>
    </div>
  </div>
  <button id="go">Transkript qilish</button>
  <div id="status"></div>
</div>

<div class="card hidden" id="result">
  <div class="stats" id="stats"></div>
  <h3 style="margin:6px 0">Matn</h3>
  <div id="fulltext"></div>
  <h3 style="margin:18px 0 6px">Segmentlar</h3>
  <table id="segments"></table>
</div>

<script>
const el = id => document.getElementById(id);
const pad = n => String(Math.floor(n)).padStart(2, "0");
const fmt = s => `${pad(s/60)}:${pad(s%60)}`;

el("go").onclick = async () => {
  const form = new FormData();
  const file = el("file").files[0];
  const url = el("url").value.trim();
  if (!file && !url) { el("status").textContent = "Fayl yoki havola kerak."; return; }
  if (file) form.append("file", file);
  if (url) form.append("url", url);
  form.append("model", el("model").value);
  form.append("events", el("events").checked);

  el("go").disabled = true;
  el("result").classList.add("hidden");
  el("segments").innerHTML = "";
  el("fulltext").textContent = "";
  el("stats").innerHTML = "";
  el("status").textContent = "Tayyorlanyapti (yuklash, VAD)…";
  const t0 = Date.now();
  const state = { meta: null, count: 0 };
  try {
    const resp = await fetch("/api/transcribe", { method: "POST", body: form });
    if (!resp.ok) {
      const data = await resp.json();
      throw new Error(data.detail || resp.statusText);
    }
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\\n");
      buffer = lines.pop();
      for (const line of lines) {
        if (line.trim()) handle(JSON.parse(line), state, t0);
      }
    }
    el("status").textContent = "";
  } catch (err) {
    el("status").textContent = "Xato: " + err.message;
  } finally {
    el("go").disabled = false;
  }
};

function handle(msg, state, t0) {
  if (msg.type === "error") throw new Error(msg.detail);
  if (msg.type === "meta") {
    state.meta = msg;
    el("stats").innerHTML =
      `<span>${msg.model_label}</span><span>audio ${fmt(msg.duration)}</span>`;
    el("result").classList.remove("hidden");
    el("status").textContent = `Transkript: 0/${msg.chunks}`;
  } else if (msg.type === "segment") {
    state.count = msg.done;
    el("status").textContent =
      `Transkript: ${msg.done}/${state.meta ? state.meta.chunks : "?"}`;
    addRow(`${fmt(msg.start)}–${fmt(msg.end)}`,
           `<td>${escapeHtml(msg.text)}</td>`, msg.start);
    el("fulltext").textContent += (state.count > 1 ? " " : "") + msg.text;
  } else if (msg.type === "events") {
    for (const ev of msg.events) {
      addRow(`${fmt(ev.start)}–${fmt(ev.end)}`,
             `<td class="ev">[${ev.label}] (${ev.score})</td>`, ev.start);
    }
  } else if (msg.type === "done") {
    const elapsed = (Date.now() - t0) / 1000;
    const dur = state.meta ? state.meta.duration : 0;
    const rt = dur ? (dur / elapsed).toFixed(1) : "?";
    const timings = Object.entries(msg.timings)
      .map(([k, v]) => `${k} ${v}s`).join(" · ");
    el("stats").innerHTML +=
      `<span>ishlov ${elapsed.toFixed(1)}s (${rt}x realtime)</span>` +
      `<span>${timings}</span>`;
    el("fulltext").innerHTML = escapeHtml(msg.text)
      .replace(/\\[(kulgu|musiqa|qarsak|yoʻtal)\\]/g,
               "<mark>[$1]</mark>");
  }
}

function addRow(time, body, start) {
  const table = el("segments");
  const row = document.createElement("tr");
  row.dataset.start = start;
  row.innerHTML = `<td class="t">${time}</td>${body}`;
  const rows = [...table.children];
  const next = rows.find(r => parseFloat(r.dataset.start) > start);
  table.insertBefore(row, next || null);
}

function escapeHtml(s) {
  const map = {"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"};
  return s.replace(/[&<>"]/g, c => map[c]);
}
</script>
</main>
</body>
</html>
"""
