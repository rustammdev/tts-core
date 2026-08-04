# Vazifa prompti — YouTube ingest va transkripsiya

Pastdagi `---` orasidagi qism — Claude Code'ga to'g'ridan-to'g'ri yopishtiriladigan prompt.

---

Kontekstni CLAUDE.md dan o'qi. Ayniqsa 4-bo'lim (data kontrakti) va 6-bo'lim
(pipeline bosqichlari) — chiqish formati aynan o'shanga mos bo'lishi shart.

**Vazifa:** pipeline'ning `ingest` va `transcribe` bosqichlarini yoz. Kirish —
YouTube URL ro'yxati. Chiqish — 4-bo'limdagi sxemaga mos JSONL manifest.

## Modullar

`src/uztts_data/ingest.py` va `src/uztts_data/transcribe.py`. Har biri typer
asosidagi mustaqil CLI. Har biri idempotent — qayta ishga tushirilganda
tugallangan ishni takrorlamaydi (holat `data/interim/<video_id>/.done` marker
fayllari orqali kuzatiladi).

## 1-bosqich: ingest

```
uztts-ingest --urls urls.txt --out data/raw/
```

Kutubxona: **yt-dlp** (Python API sifatida, subprocess emas).

Har bir video uchun:

- **Metadata** — `extract_info(download=False)` bilan: video_id, sarlavha,
  kanal, davomiylik, til belgisi, yuklangan sana. `data/raw/<video_id>/meta.json`
  ga yoz.
- **Audio** — faqat audio oqim (`bestaudio`), keyin ffmpeg orqali 24000 Hz,
  mono, 16-bit PCM WAV ga o'gir. `data/raw/<video_id>/audio.wav`.
- **Subtitrlar** — mavjud bo'lsa yuklab ol (`writesubtitles`,
  `writeautomaticsub`, `subtitleslangs=["uz"]`) va `subs.vtt` ga saqla.

**Subtitrlar haqida muhim qoida:** YouTube subtitrlari hech qachon train matni
sifatida ishlatilmaydi. Sabab: avtomatik subtitrlarda tinish belgilari yo'q,
o'zbekchada xatolik darajasi yuqori, timestamp'lar taxminiy; qo'lda yozilgan
subtitrlar esa ko'pincha qisqartirilgan yoki qayta ifodalangan bo'ladi. Ular
faqat arzon pre-filtr sifatida ishlatiladi: video haqiqatan o'zbek tilidami,
ichida nutq bormi. Matn manbai — har doim ASR.

Xatoliklarni yutib yuborma, lekin bitta video tushmasa butun jarayon
to'xtamasin: xatoni `data/raw/_failed.jsonl` ga yoz va davom et.

## 2-bosqich: transcribe

```
uztts-transcribe --in data/raw/ --out data/manifests/raw.jsonl --model large-v3
```

Kutubxona: **faster-whisper** (CTranslate2 backend — oddiy openai-whisper dan
sezilarli tez va kam xotira ishlatadi).

Sozlamalar:

- `language="uz"`, `word_timestamps=True`
- `vad_filter=True` (ichki Silero VAD) — pauzalar bo'yicha segmentlash
- `condition_on_previous_text=False` — uzun audioda halyutsinatsiya siklini
  oldini oladi

Segmentlash qoidalari:

- Maqsadli uzunlik 2–20 soniya
- Kesish faqat so'z chegarasida (word timestamp'lardan foydalan)
- Segment boshi va oxiriga 50 ms padding
- 20 soniyadan uzun segmentni eng uzun ichki pauza bo'yicha bo'l

Har bir segment alohida WAV fayl sifatida `data/interim/<video_id>/NNNNNN.wav`
ga yoziladi va manifestga bitta satr qo'shiladi.

## Manifest maydonlari

CLAUDE.md 4-bo'limidagi sxema to'liq to'ldiriladi. Shu vazifaga xos qiymatlar:

- `source` → `"youtube:<video_id>"`
- `license` → `"unknown"`
- `speaker_id` → `"unknown:<video_id>"` (diarizatsiya keyingi bosqich)
- `style_caption` → `null`
- `quality_tag` → SNR bahosidan hisoblanadi: `>25dB = clean`,
  `15–25dB = medium`, `<15dB = noisy`

Qo'shimcha diagnostik maydonlar (sxemaga `asr_meta` obyekti sifatida):
`avg_logprob`, `no_speech_prob`, `compression_ratio`, `language_probability`.

**Qattiq qoida:** `license: "unknown"` bo'lgan segmentlar train manifestiga
avtomatik o'tmaydi. `uztts-data promote` buyrug'i orqali, aniq bayroq bilan
tasdiqlanishi kerak. Buni schema validatsiyasida majburiy qil.

## 3-bosqich: filter

`src/uztts_data/filter.py` — sifatsiz segmentlarni chiqarib tashlaydi. Bu yerda
ground-truth matn yo'q, shuning uchun ASR ishonch signallaridan foydalaniladi:

| Mezon | Chegara |
|---|---|
| `avg_logprob` | > -1.0 |
| `no_speech_prob` | < 0.3 |
| `compression_ratio` | < 2.4 (yuqorisi — takroriy halyutsinatsiya) |
| `language_probability` (uz) | > 0.8 |
| davomiylik | 2–20 s |
| SNR | > 15 dB |
| matn uzunligi / davomiylik | 5–25 belgi/soniya oralig'ida |

Chegaralar `configs/data/filter.yaml` da, kodda hardcode qilinmasin.

Filtr **o'chirmaydi** — `passed: true/false` maydonini qo'yadi va sababni
`filter_reason` ga yozadi. Statistika chiqarilsin: nechta segment, qaysi mezon
bo'yicha nechtasi tushdi.

## Testlar

- `ingest`: yt-dlp mock qilingan holda metadata parsing va fayl yo'llari
- `transcribe`: 3 ta qisqa namuna WAV ustida segmentlash chegaralari
- `filter`: har bir mezon uchun chegara ustida/ostida bitta holat
- Manifest satrlari Pydantic sxemasidan o'tishi

## Nima qilinmasin

- Diarizatsiya, manba ajratish, uslub izohlari — bu bosqichda emas
- Whisper'ni fine-tune qilish — alohida vazifa
- Parallel/distributed ishlov — avval bitta jarayon to'g'ri ishlasin
- Retry/backoff murakkabliklari — oddiy 3 marta urinish yetadi

Ishni shu tartibda bajar: **ingest → test → to'xta va menga xabar ber.**
Tasdiqlagandan keyin `transcribe` ga o't.
