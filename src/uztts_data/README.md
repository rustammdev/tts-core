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

## ingest

**Talab:** PATH'da `ffmpeg` (`sudo apt install ffmpeg`).

```bash
uztts-ingest --urls urls.txt --out data/raw/
```

`urls.txt` — har satrda bitta URL; bo'sh satrlar, `#` bilan boshlanganlari va
takrorlanganlari tashlanadi.

Har bir video uchun `data/raw/<video_id>/`:

| Fayl | Nima |
|---|---|
| `meta.json` | video_id, url, sarlavha, kanal, davomiylik, til, sana, `uz_subtitles` |
| `audio.wav` | 24000 Hz, mono, 16-bit PCM |
| `subs.vtt` | mavjud bo'lsa (train uchun emas — pastga qarang) |
| `.done` | bosqich tugagani; bori qayta ishlanmaydi |

Tushgan videolar `data/raw/_failed.jsonl` ga yoziladi, qolganlari davom etadi;
buyruq oxirida chiqish kodi 1 bo'ladi.

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
