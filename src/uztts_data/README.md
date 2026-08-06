# uztts_data

Data kontrakti va uning ustidagi asboblar. Pipeline bosqichlari ham shu yerga
qo'shiladi — har biri manifestni o'qib, yangi manifest yozadi.

## Modullar

| Modul | Nima |
|---|---|
| `schema.py` | `Segment` — yagona data kontrakti (CLAUDE.md §4) |
| `manifest.py` | JSONL o'qish/yozish, validatsiya, sha256 |
| `channels.py` | Kanal registri kontrakti va Gate-2 statistikasi |
| `paths.py` | `UZTTS_DATA_ROOT` asosidagi data yo'llari |
| `cli.py` | `uztts-data` buyrug'i |
| `ingest.py` | YouTube'dan audio + metadata, `uztts-ingest` buyrug'i |
| `tg.py` | Telegram kanalidan link qabul qilish (`uztts-data tg pull`) |
| `report.py` | Registr + statistikadan HTML holat sahifasi (`uztts-data report`) |

## Kanal registri

Manba tanlash **kanal darajasida** bo'ladi (yo'l xaritasi, 02-bosqich).
Qo'lda boshqariladigan ro'yxat: `configs/channels.jsonl`, har satr bitta kanal:

```json
{"channel_id":"ch_rizanova","url":"https://www.youtube.com/@rizanova/videos","name":"RizaNova","genre":"conversation","script":"latin","est_quality":"clean","status":"approved","reject_reason":null,"notes":"suhbat, studiya sifati"}
```

- `genre` — `conversation` / `news` / `education` / `vlog` / `audiobook` / `other`
- `script` — `latin` / `cyrillic` / `mixed`
- `est_quality` — `clean` / `medium` / `noisy` (segment sxemasidagi teg bilan bir xil)
- `status` — `candidate` / `approved` / `rejected`; `rejected` bo'lsa
  `reject_reason` majburiy (dublyaj, doimiy fon musiqa, telefon yozuvi, 50/50 ruscha)
- `url` — kanalning `/videos` sahifasi tavsiya etiladi

```bash
uztts-data channels validate                 # configs/channels.jsonl tekshiradi
uztts-data channels stats                    # yt-dlp bilan soatlarni hisoblaydi
uztts-data channels stats --refresh          # keshni yangilaydi
```

`stats` natijasi `$UZTTS_DATA_ROOT/manifests/channel_stats.jsonl` ga yoziladi
(keshlangan kanal qayta so'ralmaydi) va Gate-2 hisobotini chiqaradi: janr
bo'yicha soat ulushlari va 1000 soatlik maqsadga nisbatan holat.

## tg — link qabul qilish

Nomzod kanallar Telegram kanali orqali keladi: linklar tegli post qilinadi,
bot (kanalda admin) ularni o'qiydi. Token — `.env` dagi `TELEGRAM_BOT_TOKEN`.

```bash
make tg-pull        # yoki: uv run uztts-data tg pull
```

Har yangi postdan YouTube linklari olinadi, teglardan janr aniqlanadi
(`#podkast`/`#suhbat` → conversation, `#talim`/`#dars`/`#maruza` → education,
`#vlog`/`#sayohat` → vlog, `#yangiliklar` → news, `#hikoya`/`#audiokitob` →
audiobook, tegsiz → other) va registrga `status: candidate` bilan yoziladi.
Video linki bo'lsa kanal yt-dlp bilan aniqlanadi. Takrorlar o'tkazib
yuboriladi; bot kanalga tegli tasdiq javobini yuboradi (`--no-ack` o'chiradi).

Cheklov: Bot API kanal **tarixini** ko'rmaydi — bot admin bo'lgandan keyingi
postlargina keladi (eski postni kanalga forward qilish kifoya). O'qilgan
joy `$UZTTS_DATA_ROOT/tg_offset` da saqlanadi.

## report — holat sahifasi

```bash
make report     # -> reports/index.html (git'ga kirmaydi)
```

Registr, kanal statistikasi va xom manifestdan bitta statik HTML quradi:
umumiy holat (Gate-2 va pilot progressi), janr aralashmasi maqsadga
nisbatan, saralanadigan kanallar jadvali (YouTube linklari bilan) va
boshqaruv buyruqlari. Sahifa brauzerda har 60 soniyada o'zi yangilanadi —
ma'lumot o'zgarganda `make report` ni qayta ishga tushirish kifoya.

## ingest

**Talab:** PATH'da `ffmpeg` (`sudo apt install ffmpeg`).

Ikki rejim, aynan bittasi tanlanadi:

```bash
uztts-ingest --channels configs/channels.jsonl          # asosiy yo'l
uztts-ingest --channels configs/channels.jsonl --only ch_rizanova
uztts-ingest --urls urls.txt                            # bir martalik videolar
```

`--channels` registrdagi **faqat `approved`** kanallarni oladi: har kanalning
video ro'yxati flat-extract bilan tuziladi va har video odatdagi idempotent
oqimdan o'tadi. `--urls` rejimida videolar `adhoc` pseudo-kanaliga tushadi.

Davomiylik chegaralari: `--min-duration 60` va `--max-duration 14400`
(soniya, 0 — o'chirilgan). Chegaradan tashqari video **yuklab olinmaydi** —
`.filtered` marker va sabab yoziladi, keyingi ishga tushirishda so'ralmaydi.

`--max-channel-hours 3` — kanal boshiga soat chegarasi (0 — o'chirilgan):
yuklangan soat chegaraga yetganda kanalning qolgan videolari bu ishga
tushirishda olinmaydi (`capped` satri nechta qolganini aytadi). Chegara
diskda tugallangan videolarni ham hisoblaydi, shuning uchun qayta ishga
tushirish xavfsiz; kattaroq chegara bilan chaqirilsa davom etadi. Katta
kanallardan (masalan KunUZ) faqat kerakli ulushni olish usuli shu.

Har bir video uchun `$UZTTS_DATA_ROOT/raw/<channel_id>/<video_id>/`:

| Fayl | Nima |
|---|---|
| `meta.json` | video_id, url, sarlavha, kanal, `channel_id`, davomiylik, til, sana, `uz_subtitles` |
| `audio.wav` | 24000 Hz, mono, 16-bit PCM |
| `subs.vtt` | mavjud bo'lsa (train uchun emas — pastga qarang) |
| `.done` / `.filtered` | bosqich tugagani; bori qayta ishlanmaydi |

Tushgan videolar kanal papkasidagi `_failed.jsonl` ga yoziladi, qolganlari
davom etadi; buyruq oxirida chiqish kodi 1 bo'ladi.

## scan-raw va stats

```bash
uztts-data scan-raw     # raw/ -> manifests/raw.jsonl (video darajasidagi segmentlar)
uztts-data stats "$UZTTS_DATA_ROOT/manifests/raw.jsonl"
```

`scan-raw` diskdan qayta tiklanadigan manifest quradi: har tugallangan video —
bitta satr, davomiylik va sample rate to'g'ridan-to'g'ri WAV'dan o'qiladi.
`audio_path` manifestda **data root'ga nisbatan** yoziladi. `stats` kanal
bo'yicha soatlarni chiqaradi — Gate-3 yo'qotish hisobining boshlang'ich nuqtasi.

**Subtitrlar train matni sifatida ishlatilmaydi** — faqat videoda o'zbekcha nutq
borligini bildiruvchi arzon signal. Matn manbai har doim ASR
(`docs/decisions/003-youtube-ingest.md`).

`MediaSource` protokoli tarmoq bilan yagona chegara — testlarda yt-dlp o'rniga
soxta manba qo'yiladi.

## Manifest

```python
from pathlib import Path

from uztts_data import read_manifest, write_manifest

segments = [
    s for s in read_manifest(Path("data/manifests/raw.jsonl")) if s.duration > 3
]
write_manifest(Path("data/manifests/long.jsonl"), segments)
```

```bash
uztts-data validate data/manifests/train.jsonl
uztts-data hash     data/manifests/train.jsonl
```

## Kontrakt qoidalari

- `Segment` **frozen** — bosqich segmentni o'zgartirmaydi, `model_copy(update=...)`
  bilan yangisini yasaydi.
- `extra="forbid"` — manifestda notanish maydon bo'lsa xato. Sxemaga maydon
  qo'shish buzuvchi o'zgarish, avval so'raladi.
- `read_manifest` generator: xato satrda `ManifestError` fayl nomi va satr
  raqami bilan tashlanadi.
- `write_manifest` avval `.tmp` ga yozib, keyin `replace` qiladi — yarim
  yozilgan manifest qolmaydi.
- Maydon tartibi CLAUDE.md §4 dagi tartibga qat'iy mos, test bilan qulflangan.
