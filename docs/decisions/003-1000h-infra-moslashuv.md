# 003 — Data-infra'ni 1000 soatlik yo'l xaritasiga moslash

**Holat:** qabul qilindi (2026-08-05, to'rttala taklif tasdiqlandi)
**Sana:** 2026-08-05
**Manbalar:** `docs/refs/uzbek-tts-train-yol-xaritasi.pdf`, `docs/refs/neyron-nutq-sintezi.pdf`

---

## Kontekst

Ikkala hujjat loyihaga ko'chirildi va o'qildi. Yo'l xaritasi 8 bosqichli, har
bosqichda o'lchanadigan o'tish sharti (gate) bor. Model tanlash — eng oxirgi
qaror; hozirgi vazifa data yig'ish va tayyorlash infratuzilmasini shu
xaritaga moslash.

Hozirgi repo holati:

- `uztts_data`: schema (Pydantic), manifest validate/hash, YouTube ingest
  (yt-dlp → 24 kHz mono WAV, idempotent, `.done` markerlar) — ishlaydi, testlangan.
- Yo'q: kanal registri, segment (VAD), transcribe, filter, eval, `uztts_text`.

### Aniqlangan nomuvofiqliklar

1. **Doira:** `CLAUDE.md` V0 = bitta ovoz, 30 daqiqa → 3 soat. Yo'l xaritasi =
   1000 soat xom, 300+ spiker, sun'iy ovoz. Ikkalasini kelishtirish kerak.
2. **Litsenziya maydoni:** schema'da `license ∈ {owned, licensed, public_domain}`.
   YouTube'dan yig'ilgan data bularning hech biriga kirmaydi — yangi qiymat kerak
   (masalan `web_scraped`), aks holda kontrakt yolg'on gapiradi.
3. **`channel_id` yo'q:** yo'l xaritasida kanal ID uch joyda kritik — spiker
   taxmini (kanal + klaster = global spiker), aralashma balansi, validation'ni
   kanal bo'yicha bo'lish. Hozirgi schema'da bu maydon yo'q.
4. **ASR diagnostikalari yo'q:** filtr mezonlari (`avg_logprob > −1.0`,
   siqilish nisbati `< 2.4`, til ehtimoli `> 0.8`) manifestda saqlanadigan joy yo'q.
5. **`CLAUDE.md` `docs/decisions/002-repo-skeleton.md` ga havola qiladi, fayl
   mavjud emas** — 001/002 keyin to'ldiriladi, shu hujjat 003 raqamini oladi.

### Tizim inventarizatsiyasi (halol hisob, 2026-08-05)

| Resurs | Holat | Xulosa |
|---|---|---|
| WSL ext4 (`/`) | 933 GB bo'sh | data shu yerda yashaydi — 500 GB rejaga yetadi |
| `/mnt/c` | 232 GB bo'sh, sekin I/O | faqat kod; audio uchun yaroqsiz |
| GPU | RTX 5060 Ti, 16 GB VRAM | xarita 4090/24GB (yoki A100) nazarda tutgan |
| RAM (WSL) | 15 GB | segment/transcribe uchun yetadi, batch kichik |

GPU oqibatlari: transkripsiya (faster-whisper, int8/fp16) — muammosiz.
Whisper fine-tune — LoRA/8-bit bilan sig'adi. TTS train (Faza 1, 1.7–3B model)
— 16 GB'da QLoRA bilan yoki ijaraga olingan GPU'da (bu qaror train bosqichida,
hozir emas — lekin infra uni bloklamasligi kerak: checkpoint va data HF Hub
orqali ko'chma bo'lsin).

---

## Qaror (taklif)

1. **Yo'l xaritasi — loyihaning asosiy rejasi.** `CLAUDE.md` yangilanadi:
   V0 = **50 soatlik pilot** — butun quvur (kanal registri → ingest → segment
   → transcribe → filter → hisobot) kichik hajmda uchidan-uchiga o'tadi.
   Bu xaritaning Gate-3 sharti bilan aynan mos ("quvur 50 soatlik namunada
   o'tdi") va `CLAUDE.md`ning "sifat emas, ishlaydigan quvur" tamoyilini saqlaydi.
2. **Schema kengaytiriladi** (4-bo'lim buzuvchi o'zgarish — alohida tasdiq bilan).
3. **Hugging Face — saqlash va ASR qatlami:** private dataset/model repolar,
   tayyor o'zbek korpuslari (USC, Common Voice uz, FLEURS uz) ASR fine-tune
   uchun. Data'ning haqiqat manbai lokal manifest bo'lib qoladi; HF — nusxa,
   versiya va ko'chmalik qatlami.
4. **Data root WSL ext4'ga ko'chadi**, `UZTTS_DATA_ROOT` env orqali sozlanadi.

---

## Reja: katta qadamlar, kichik actionlar

Har qadam xaritadagi gate'ga xizmat qiladi. Tartib — xarita tartibi.

### A. Kanal registri (xarita 02 → Gate-2)

Eng qimmat xato manba tanlashda qilinadi — shuning uchun birinchi qurilma
shu. Kanal ro'yxati **kanal darajasida**, video darajasida emas.

- A1. `Channel` sxemasi: `channel_id`, `url`, `name`, `genre`
  (suhbat / yangiliklar / ta'lim / vlog / audiokitob), `est_quality`,
  `script` (latin / cyrillic / mixed), `status`
  (candidate / approved / rejected), `reject_reason`, `notes`.
  Fayl: `data/manifests/channels.jsonl`.
- A2. `uztts-data channels validate` — sxema tekshiruvi; chiqarib tashlash
  qoidalari eslatmasi (dublyaj, doimiy fon musiqa, telefon yozuvi, 50/50 ruscha).
- A3. `uztts-data channels stats` — janr ulushlari va taxminiy soatlar
  (yt-dlp flat-extract bilan kanal videolari davomiyligi yig'indisi).
  **Gate-2 hisoboti shu buyruqdan chiqadi.**

### B. Ingest'ni kanal darajasiga ko'tarish (xarita 03)

- B1. `UZTTS_DATA_ROOT` — data root env orqali; default `~/uztts-data`
  (ext4). Kod `/mnt/c`da qolaveradi.
- B2. Kanal/playlist URL qabul qilish: yt-dlp flat-extract → video ro'yxati →
  mavjud per-video idempotent oqim. `meta.json`ga `channel_id`.
- B3. Xom manifest yozish: majburiy maydonlar (`id`, `audio_path`,
  `speaker_id`, `duration`, `sample_rate`, `source`, `license`) + `channel_id`.
- B4. `uztts-data stats` — kanal/janr bo'yicha soatlar, disk sarfi.
  **Gate-2/3 raqamlari shu yerdan.**

### C. Segment (xarita 03)

- C1. VAD: `silero-vad` (MIT) → 2–20 s bo'laklar; ustma-ust nutq va musiqa
  aralash joylar V0'da tashlab yuboriladi (xarita: "muzokarasiz").
- C2. Spiker taxmini V0: `speaker_id = channel_id` proxy (bir kanal — bitta
  asosiy ovoz taxmini). Diarizatsiya (pyannote) — V1, lekin `speaker_id`
  formati `ch{NNN}_c{K}` klasterga tayyor.
- C3. Yo'qotish statistikasi: har video uchun necha % VAD'dan o'tdi —
  **Gate-3 "yo'qotish rejali" hisoboti**.
- C4. Demucs (manba ajratish) — V1; lekin schema'da `separated` bayrog'i
  hozir qo'shiladi (keyin 20% byudjetni nazorat qilish uchun).

### D. Transcribe — "loyihaning shipi" (xarita 04 → Gate-4)

- D1. `faster-whisper` (MIT) large-v3 bilan birinchi o'tish; har segmentga
  `text` + diagnostikalar: `asr_avg_logprob`, `asr_compression_ratio`,
  `lang_prob`.
- D2. Etalon to'plam vositasi: 2 soatlik tasodifiy namunani ajratish, qo'lda
  tekshirish formati, `jiwer` (Apache-2.0) bilan WER. **Gate-4: WER ≤ 10%
  — bunga yetmaguncha train yo'q.**
- D3. Whisper'ni o'zbekchada fine-tune: USC (~105 soat) + Common Voice uz +
  FLEURS uz, HF `transformers` + LoRA, 5060 Ti'da bir necha kun. Natija —
  private HF model repo.
- D4. Ikki modelli konsensus: base large-v3 vs uz-fine-tune; mos kelganlar —
  ishonchli qatlam, farq — shubha belgisi (`asr_cer` maydoniga ikkala
  transkript orasidagi CER yoziladi).
- D5. Yozuv tizimi qarori: **ichki format — lotin**. Kirill→lotin
  deterministik transliteratsiya va apostrof normalizatsiya `uztts_text`da —
  bu `uztts_text` MVP'ning birinchi real vazifasi (CLAUDE.md yo'l
  xaritasidagi 3-qadam bilan mos).

### E. Filter va sifat qatlamlari (xarita 05 → Gate-5)

- E1. `configs/filter.yaml` — barcha chegaralar konfigda, kodda emas:
  uzunlik 2–20 s, `avg_logprob > −1.0`, siqilish `< 2.4`, belgi/soniya 5–25,
  `lang_prob > 0.8` (kod-almashinuvli gaplarni o'ldirmasligini etalonda tekshirish).
- E2. `filter` CLI: hech narsa o'chirilmaydi — `quality_tag` (clean/medium/noisy
  = A/B/C qatlamlar) va rad sababi yoziladi.
- E3. Gate-5 hisobot: qatlam bo'yicha soatlar, spikerlar soni
  (maqsad: ≥300 soat, A ≥ 50 soat, spikerlar ≥ 300).

### F. O'lchov intizomi skaffoldi (xarita 09)

- F1. `uztts-data split --by-channel` — validation **butun kanallar bo'yicha**,
  hech qachon segment bo'yicha emas.
- F2. `docs/eval/testset.jsonl` — doimiy 200 jumlalik test to'plami (oddiy
  gaplar, raqam/sana, savol-undov, uzun gaplar, ruscha aralash). Faqat o'sadi.

Tokenizer tekshiruvi (xarita 06) va train fazalari (07–08) — model tanlash
bosqichiga tegishli, hozir qurilmaydi; lekin F2 test to'plami va D5 lotin
formati ularni bloklamaydigan qilib hozir belgilanadi.

---

## Hugging Face roli

| Vazifa | Vosita | Izoh |
|---|---|---|
| Processed data nusxa + versiya | private dataset repo, parquet + `hf_transfer` | xom YouTube audio **hech qachon public qilinmaydi** |
| ASR fine-tune data | USC, Common Voice uz (CC-0), FLEURS uz (CC-BY) | faqat ASR uchun — TTS modelga litsenziya yuqmaydi |
| Whisper-uz checkpoint | private model repo | D3 natijasi |
| TTS checkpoint ko'chmaligi | private model repo | 16 GB lokal GPU ↔ ijara GPU orasida |
| Inson baholash (V1) | HF Spaces (Gradio) | MOS/A-B test UI |

`HF_TOKEN` — `.env`da, hech qachon commit qilinmaydi. `datasets`,
`huggingface_hub` — Apache-2.0, litsenziya toza.

## Yangi kutubxonalar (litsenziya tekshiruvi)

| Kutubxona | Litsenziya | Qadam |
|---|---|---|
| silero-vad | MIT | C |
| faster-whisper (ctranslate2) | MIT | D |
| jiwer | Apache-2.0 | D |
| transformers, peft, datasets, huggingface_hub | Apache-2.0 | D3, HF |
| soundfile | BSD-3 | C |
| pyannote.audio (V1) | MIT (og'irliklar HF gated) | V1 |
| demucs (V1) | MIT | V1 |

## Muqobillar

- **Emilia-Pipe'ni to'g'ridan-to'g'ri olish** (xarita adabiyotida bor):
  tayyor podkast-korpus quvuri, lekin o'z kontraktimizga moslash qimmatga
  tushadi; dizayn g'oyalari olinadi (bosqich tartibi, filtr mezonlari), kod emas.
- **Avval 3 soatlik V0'ni tugatish:** kichik doira afzalligi bor, lekin kanal
  registri va `channel_id`siz yig'ilgan har bir soat keyin qayta ishlanadi —
  qimmat xato. Rad etildi.
- **DVC bilan data versiyalash:** V1'da qayta ko'riladi; hozircha manifest
  hash + HF snapshot yetadi.

## Oqibatlar

- Schema kengayishi — bir martalik buzuvchi o'zgarish; hozirgi manifestlar
  bo'sh, migratsiya arzon. Kechiktirilsa — har soat data bilan qimmatlashadi.
- 1000 soat miqyosi divan hisobida: xom WAV ~165 GB, oraliq bilan ~500 GB —
  ext4'da joy bor.
- Transkripsiya hisobi: faster-whisper large-v3, 5060 Ti, ~10–15× real-time →
  1000 soat ≈ 3–5 kun uzluksiz GPU. Pilot 50 soat ≈ yarim kun.
