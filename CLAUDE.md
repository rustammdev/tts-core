# CLAUDE.md — O'zbekcha TTS loyihasi

> Loyihaning yagona haqiqat manbai. Qaror o'zgarsa — avval shu fayl yangilanadi.

**Loyiha:** `uz-tts` · **Holat:** V0 pilot quvuri qurilmoqda · **Til:** kod
inglizcha, hujjatlar o'zbekcha

---

## 1. Maqsad

O'zbek tili uchun **hissiy jihatdan ifodali** TTS. Farqlanish nuqtasi — tabiiy
intonatsiya, kitob o'qigandek tekis o'qish emas.

**Asosiy reja** — `docs/refs/uzbek-tts-train-yol-xaritasi.pdf`: YouTube'dan
1000 soat xom audio → filtrlashdan keyin 350–450 soat → ko'p spikerli
fine-tune → sun'iy ovoz. Har bosqichda o'lchanadigan o'tish sharti (gate).
Moslashuv qarori: `docs/decisions/004-1000h-infra-moslashuv.md`. Texnik
asoslar: `docs/refs/neyron-nutq-sintezi.pdf`.

**V0 doirasi (pilot):** 50 soat xom data, butun quvur uchidan-uchiga:
kanal registri → ingest → segment → transcribe → filter → hisobot.
Maqsad **sifat emas, ishlaydigan quvur** — xaritaning Gate-3 sharti.

**V0 da qilinmaydi:** train, sun'iy ovoz generatsiyasi, diarizatsiya
(pyannote), manba ajratish (Demucs), real-time streaming, public API.
Bularning hech biri kelajakda to'siqqa aylanmasligi kerak.

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
| Data manbasi | YouTube, **kanal darajasida** tanlov | 100–200 sifatli kanal ≈ 1000 soat; dublyaj, doimiy fon musiqa, telefon yozuvi chiqariladi |
| Ichki yozuv tizimi | **Lotin** | kirill deterministik o'giriladi; aralash yozuv modelni chalg'itadi |
| Data joyi | `UZTTS_DATA_ROOT` env, WSL ext4 | `/mnt/c` sekin va tor; rejaga 500 GB kerak |
| Saqlash / ko'chmalik | Hugging Face **private** repolar | dataset + checkpoint nusxasi; xom YouTube audio hech qachon public qilinmaydi |

Batafsil va yangi qarorlar — `docs/decisions/`.

---

## 3. Repo tuzilishi

```
tts-core/
├── CLAUDE.md
├── Makefile              # barcha buyruqlar shu yerdan
├── pyproject.toml        # deps + ruff + mypy + pytest
├── configs/              # qo'lda boshqariladigan konfiglar (kanal registri, filtrlar)
├── src/
│   └── uztts_data/       # data kontrakti va pipeline bosqichlari
├── data/                 # git'ga kirmaydi; joyi UZTTS_DATA_ROOT bilan
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
| `uztts_asr` | GigaAM fine-tune: data prep, train, WER (007) | boshlanmoqda |
| `uztts_text` | o'zbekcha text frontend (mustaqil) | MVP tayyor |
| `uztts_events` | audio hodisa teglari: [kulgu], [musiqa] (009-qaror) | MVP tayyor, kalibrlash kutilmoqda |
| `uztts_train` | TTS baza model adapteri + train loop | ⬜ |
| `uztts_eval` | avtomatik va inson baholash | ⬜ |
| `uztts_serve` | FastAPI | ⬜ V1 |

---

## 4. Data kontrakti — eng muhim qism

Butun kengayish qobiliyati shu formatga bog'liq. **V0 da bitta spiker bo'lsa ham
to'liq sxema ishlatiladi.** Keyin podkast data qo'shilganda hech narsa qayta
yozilmaydi.

`data/manifests/*.jsonl`, har satr bitta segment:

```json
{"id":"ch_rizanova_000123","audio_path":"data/processed/ch_rizanova/000123.wav","text":"Assalomu alaykum, xush kelibsiz.","text_normalized":"assalomu alaykum xush kelibsiz","speaker_id":"ch_rizanova_c0","channel_id":"ch_rizanova","duration":3.42,"sample_rate":24000,"quality_tag":"clean","snr_db":34.1,"separated":false,"source":"youtube","license":"web_scraped","style_caption":null,"asr_cer":0.01,"asr_avg_logprob":-0.31,"asr_compression_ratio":1.42,"lang_prob":0.97}
```

**Majburiy** (`ingest` dayoq ma'lum): `id`, `audio_path`, `speaker_id`,
`duration`, `sample_rate`, `source`, `license`. YouTube manbada `channel_id`
ham ingest'da to'ldiriladi.

**Ixtiyoriy** — pipeline bosqichlari to'ldiradi: `text`, `asr_avg_logprob`,
`asr_compression_ratio`, `lang_prob` (transcribe), `text_normalized`
(uztts_text), `quality_tag` va `snr_db` (filter), `asr_cer` (align),
`separated` (V1 separate, default `false`), `style_caption` (V1 caption).

Maydonlar haqida:

- `quality_tag` — `clean` / `medium` / `noisy`. Train paytida shart
  (conditioning) sifatida beriladi. Shovqinli data **tashlanmaydi — belgilanadi**.
  Inference'da `clean` tanlanadi.
- `license` — `owned` / `licensed` / `public_domain` / `web_scraped`. Har bir
  segment o'z kelib chiqishini bilib tursin; keyinchalik audit uchun kerak.
  YouTube data — `web_scraped`.
- `channel_id` — YouTube kanali. Uch vazifasi bor: spiker taxmini (kanal +
  klaster), aralashma balansi, validation'ni **kanal bo'yicha** bo'lish.
  O'z yozuvlarida `null`.
- `speaker_id` — V0 da kanal proxy'si (`ch_xxx_c0`), format klasterga tayyor.
  Kod hech qachon "bitta spiker bor" deb faraz qilmasin.
- `separated` — Demucs qo'llanganini belgilaydi; bunday segmentlar train
  setining 20% idan oshmasligi kerak (xarita 03).
- `asr_avg_logprob`, `asr_compression_ratio`, `lang_prob` — filtr mezonlari
  uchun ASR diagnostikalari (xarita 05).
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
- Data root: `UZTTS_DATA_ROOT` env (default `data/`); ishchi muhitda WSL ext4
  (`~/uztts-data`) — `/mnt/c` orqali audio I/O qilinmaydi
- Data versiyalash: V0 da papka + manifest hash + HF private dataset snapshot;
  V1 da DVC qayta ko'riladi
- Hugging Face: private dataset/model repolar; `HF_TOKEN` `.env` da
- GPU: lokalda RTX 5060 Ti 16 GB — inference/ASR yetadi; katta train uchun
  checkpoint va data HF orqali ko'chma bo'lsin (ijara GPU'ga o'tish oson)
- Sirlar `.env` da, hech qachon commit qilinmaydi (namuna: `.env.example`)

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
2. ✅ **Data kontrakti** — `schema.py` (004 kengaytmasi bilan), manifest
   o'qish/yozish, `validate` CLI, `UZTTS_DATA_ROOT`
3. ⬜ **Kanal registri** — `channels.jsonl` kontrakti, `channels validate` va
   `channels stats` (janr/soat hisobi) → **Gate-2**
4. ✅ **Kanal darajasida ingest** — kanal URL → barcha videolar, `channel_id`,
   davomiylik chegaralari; `scan-raw` → `manifests/raw.jsonl`; `stats` hisobot
5. ⬜ **Segment** — silero-vad, 2–20 s, yo'qotish statistikasi → xom manifest
6. ⬜ **Transcribe** — faster-whisper + diagnostikalar; 2 soatlik etalon
   tayyor (v2, 008-qaror) + WER vositasi → **Gate-4: WER ≤ 10%**
   (hozircha eng yaxshisi gemini_full_220m, kanonik WER 13.4% — `docs/eval/`,
   o'lchov konventsiyasi: 010-qaror)
7. ✅ **`uztts_text` MVP** — kirill→lotin, apostroflar, sonlar + `golden.jsonl`
   (kasr/valyuta/sana va klitika kanonizatsiyasi keyingi bosqich)
8. ⬜ **Filter** — `configs/filter.yaml`, qatlam hisoboti → **Gate-5**
9. ⬜ **O'z ASR modelimiz** — GigaAM fine-tune (USC + Common Voice +
   FLEURS, punktuatsiya bilan; 007-qaror); konsensus GigaAM-large bilan
   (turbo-uzbek etalon v2 da 39.6% WER — yaroqsiz)
10. ⬜ **O'lchov skaffoldi** — `split --by-channel`, 200 jumlalik doimiy
    test to'plami

Shundan keyin: tokenizer tekshiruvi (xarita 06), baza model bake-off
(Orpheus vs Chatterbox-Turbo) va train fazalari.
