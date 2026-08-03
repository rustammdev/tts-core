# CLAUDE.md — O'zbekcha TTS loyihasi

> Loyihaning yagona haqiqat manbai. Qaror o'zgarsa — avval shu fayl yangilanadi.

**Loyiha:** `uz-tts` · **Holat:** V0 skelet · **Til:** kod inglizcha, hujjatlar
o'zbekcha

---

## 1. Maqsad

O'zbek tili uchun **hissiy jihatdan ifodali** TTS. Farqlanish nuqtasi — tabiiy
intonatsiya, kitob o'qigandek tekis o'qish emas.

**V0 doirasi:** bitta ovoz (muallif yozuvi), 30 daqiqa → 3 soat data, uchidan-
uchiga ishlaydigan zanjir: yozuv → tozalash → train → sintez → baholash.
Maqsad **sifat emas, ishlaydigan quvur**.

**V0 da qilinmaydi:** ko'p spikerli train, sun'iy ovoz generatsiyasi, real-time
streaming, public API, Kubernetes. Bularning hech biri kelajakda to'siqqa
aylanmasligi kerak.

---

## 2. Asosiy qarorlar

| Qaror | Tanlov | Sabab |
|---|---|---|
| Train usuli | Noldan emas, **fine-tune** | Noldan train yuzlab soat data talab qiladi |
| Baza model | **Orpheus 3B** (Apache-2.0) | Ifodali, LLM asosida, tijoratga ochiq |
| Zaxira nomzod | Chatterbox-Turbo (MIT) | Bake-off uchun ikkinchi variant |
| Litsenziya siyosati | **Faqat Apache-2.0 / MIT** | XTTS-v2 (CPML) va F5-TTS (CC-BY-NC) tijoratni bloklaydi |
| Train strategiyasi | Ikki bosqichli (V1 dan) | 1) ko'p spiker → til va prosodiya, 2) yakka ovoz → tembr |
| Text frontend | Modeldan **mustaqil paket** | Model almashsa ham qayta yozilmaydi |

Batafsil va yangi qarorlar — `docs/decisions/`.

---

## 3. Repo tuzilishi

```
tts-core/
├── CLAUDE.md
├── Makefile              # barcha buyruqlar shu yerdan
├── pyproject.toml        # deps + ruff + mypy + pytest
├── src/
│   └── uztts_data/       # data kontrakti va pipeline bosqichlari
├── data/                 # git'ga kirmaydi
│   ├── raw/  interim/  processed/
│   └── manifests/        # JSONL — data kontrakti
├── tests/
└── docs/decisions/
```

Paket **faqat ishlaydigan kod paydo bo'lganda** yaratiladi — bo'sh stub paket
yozilmaydi (sabab: `docs/decisions/002-repo-skeleton.md`). Rejadagi paketlar:

| Paket | Vazifa | Holat |
|---|---|---|
| `uztts_data` | schema, manifest, pipeline bosqichlari | qisman |
| `uztts_text` | o'zbekcha text frontend (mustaqil) | keyingi qadam |
| `uztts_train` | baza model adapteri + train loop | ⬜ |
| `uztts_eval` | avtomatik va inson baholash | ⬜ |
| `uztts_serve` | FastAPI | ⬜ V1 |

---

## 4. Data kontrakti — eng muhim qism

Butun kengayish qobiliyati shu formatga bog'liq. **V0 da bitta spiker bo'lsa ham
to'liq sxema ishlatiladi.** Keyin podkast data qo'shilganda hech narsa qayta
yozilmaydi.

`data/manifests/*.jsonl`, har satr bitta segment:

```json
{"id":"spk001_000123","audio_path":"data/processed/spk001/000123.wav","text":"Assalomu alaykum, xush kelibsiz.","text_normalized":"assalomu alaykum xush kelibsiz","speaker_id":"spk001","duration":3.42,"sample_rate":24000,"quality_tag":"clean","snr_db":34.1,"source":"own_recording","license":"owned","style_caption":null,"asr_cer":0.01}
```

**Majburiy** (`ingest` dayoq ma'lum): `id`, `audio_path`, `speaker_id`,
`duration`, `sample_rate`, `source`, `license`.

**Ixtiyoriy** — pipeline bosqichlari to'ldiradi: `text` (transcribe),
`text_normalized` (uztts_text), `quality_tag` va `snr_db` (filter),
`asr_cer` (align), `style_caption` (V1 caption).

Maydonlar haqida:

- `quality_tag` — `clean` / `medium` / `noisy`. Train paytida shart
  (conditioning) sifatida beriladi. Shovqinli data **tashlanmaydi — belgilanadi**.
  Inference'da `clean` tanlanadi.
- `license` — `owned` / `licensed` / `public_domain`. Har bir segment o'z kelib
  chiqishini bilib tursin; keyinchalik audit uchun kerak.
- `speaker_id` — V0 da bitta qiymat, lekin kod hech qachon "bitta spiker bor"
  deb faraz qilmasin.
- `style_caption` — V0 da `null`, V1 da tabiiy tildagi uslub izohi.

Sxema: `src/uztts_data/schema.py` (Pydantic, `extra="forbid"`, `frozen=True`).
Serializatsiya maydon tartibi shu jadvalga qat'iy mos — testda tekshiriladi.

**Sxema o'zgarishi — buzuvchi o'zgarish. Avval so'rang.**

---

## 5. Text frontend (`uztts_text`)

Modeldan butunlay mustaqil. Kiruvchi: xom matn. Chiquvchi: normallashgan matn.

Qamrovi: sonlar (`1 245 000` → "bir million ikki yuz qirq besh ming"), valyuta,
foiz, sana, vaqt, telefon raqamlari, qisqartmalar (INN, PINFL, MFO, JSHSHIR),
lotin ↔ kirill transliteratsiya, apostroflar (`o'`, `g'`, tutuq belgisi) —
barcha variantlarni bitta shaklga, ruscha kod-almashinuv.

**Har bir qoida uchun test majburiy.** `tests/text/golden.jsonl` — kirish/chiqish
juftliklari; bu fayl faqat o'sadi.

Model qanchalik yaxshi bo'lmasin, foydalanuvchi eshitadigan xatolarning katta
qismi shu bosqichda tug'iladi.

---

## 6. Pipeline bosqichlari

Har bir bosqich — **alohida idempotent CLI**. Qayta ishga tushirilsa tugallangan
ishni qayta bajarmaydi. Har biri manifestni o'qiydi va yangi manifest yozadi.

| # | Bosqich | Vazifa | V0 |
|---|---|---|---|
| 1 | `ingest` | Xom audio + metadata (YouTube, `uztts-ingest`) | ✅ |
| 2 | `separate` | Musiqa/fon ajratish (Demucs) | ⬜ V1 |
| 3 | `diarize` | Spikerlarni ajratish (pyannote) | ⬜ V1 |
| 4 | `segment` | VAD, 3–20s bo'laklar | ⬜ |
| 5 | `transcribe` | ASR (Whisper, o'zbekcha fine-tune) | ⬜ |
| 6 | `align` | Matn ↔ audio moslash | ⬜ |
| 7 | `filter` | CER, SNR, davomiylik bo'yicha tozalash | ⬜ |
| 8 | `caption` | Uslub izohlari (audio LLM) | ⬜ V1 |

V1 bosqichlari **hozir yozilmaydi**, lekin CLI interfeysi va manifest oqimi
ularni keyin qo'shishga tayyor.

---

## 7. Train

- Konfiguratsiya YAML orqali, hardcode qilingan giperparametr yo'q
- Har bir run: konfig nusxasi + git commit hash + manifest hash
  (`uztts-data hash`) saqlansin
- Checkpoint har N qadamda, oxirgi 3 tasi saqlanadi
- Experiment tracking: boshida oddiy JSONL log yetadi

Bir xil konfig + bir xil manifest = bir xil natija. Seed qat'iy belgilansin.

---

## 8. Baholash

**Avtomatik** (har train'dan keyin): sintez → ASR → WER; spiker o'xshashligi
(ECAPA-TDNN cosine); sifat proksisi (UTMOS yoki DNSMOS). Doimiy test jumlalari
to'plami — modellar orasida solishtirish uchun.

**Inson** (har muhim bosqichda): 10 nafar ona tili sohibi, A/B ko'r-ko'rona test,
5 ballik MOS — tabiiylik, tushunarlilik, hissiy ifodalilik alohida.

Natijalar `docs/eval/` da. "Yaxshi bo'ldi" degan his emas, raqam.

---

## 9. Infra

- Python 3.11, `uv` bilan boshqariladi
- Tizim talabi: `ffmpeg` (audio konvertatsiya)
- Barcha buyruqlar `Makefile` orqali
- Data versiyalash: V0 da papka + manifest hash; V1 da DVC yoki S3
- GPU: bitta consumer GPU (RTX 4090 sinfi) yetarli bo'lsin
- Sirlar `.env` da, hech qachon commit qilinmaydi

---

## 10. Kod konvensiyalari

- Formatlash va lint: `ruff`; type hints majburiy, `mypy --strict`
- Testlar: `pytest`; `make check` — lint + typecheck + test
- **Izoh yozilmaydi.** Nom va tuzilish o'zini tushuntirsin. Docstring ham
  shunga tegishli. Istisno: sababi koddan ko'rinmaydigan qaror.
- Bitta funksiya bitta ish qiladi; pipeline bosqichlari sof funksiya bo'lsin
- Notebook'lar `notebooks/` da, `src/` ga import qilinmaydi

---

## 11. Claude uchun ish qoidalari

1. **Bosqichma-bosqich ishla.** Bitta bosqichni tugatib, tasdiq so'ra.
2. **Chatga uzun kod bloklarini chiqarma.** Kodni faylga yoz, chatda qisqa xulosa.
3. **Data kontraktini (4-bo'lim) o'zgartirishdan oldin so'ra.**
4. **V1 belgisi qo'yilgan narsalarni hozir qurma**, lekin ularni bloklaydigan
   qaror ham qabul qilma.
5. **Yangi kutubxona qo'shishdan oldin litsenziyasini tekshir.** Copyleft yoki
   non-commercial bo'lsa — avval so'ra.
6. **Har bir yangi paket uchun `README.md`** — nima qiladi, qanday ishlatiladi.
7. Qaror qabul qilinsa — `docs/decisions/NNN-sarlavha.md` (kontekst / qaror /
   muqobillar / oqibatlar).
8. Noaniqlik bo'lsa — taxmin qilma, so'ra.

---

## 12. Yo'l xaritasi

1. ✅ **Skelet** — `pyproject.toml`, `Makefile`, ruff/mypy/pytest
2. ✅ **Data kontrakti** — `schema.py`, manifest o'qish/yozish, `validate` CLI
3. ⬜ **`uztts_text` MVP** — sonlar, apostrof, transliteratsiya + `golden.jsonl`
4. ⬜ **Pipeline V0** — `ingest → segment → transcribe → align → filter`,
   30 daqiqalik namunada uchidan-uchiga
5. ⬜ **Train adapteri** — Orpheus fine-tune, YAML konfig, `synthesize` CLI

Shundan keyin: birinchi sintez namunasini tinglab, keyingi yo'nalish belgilanadi.
