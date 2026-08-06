# 007 — O'z ASR modelimiz: GigaAM'ni o'zbekchaga fine-tune qilish

Sana: 2026-08-06 · Holat: qabul qilingan

## Kontekst

Eshitish benchmarkida (006 davomi) GigaAM-Multilingual (MIT) o'zbekchada
barcha Whisper variantlaridan ustun chiqdi — foydalanuvchi tasdiqladi.
Kamchiliklari: ayrim so'zlarda xato, punktuatsiya va bosh harflar yo'q.
Repo fine-tuning'ni rasman qo'llaydi (PyTorch Lightning, VRAM'ga
moslashgan misollar, charwise CTC).

### GigaAM liniyalari (2026-08 holatiga)

| Liniya | Chiqqan | Til | Checkpointlar |
|---|---|---|---|
| v1 | 2024-04 | rus | `v1_ssl`, `v1_ctc`, `v1_rnnt`, `emo` |
| v2 | 2024-12 | rus | `v2_ssl`, `v2_ctc`, `v2_rnnt` |
| v3 | 2025-11 | rus | `v3_ssl`, `v3_ctc`, `v3_rnnt`, `v3_e2e_ctc`, `v3_e2e_rnnt` |
| **multilingual** | **2026-06** | 70+ til, o'zbekcha bor | `multilingual_ssl` (220M), `multilingual_large_ssl` (600M), `multilingual_ctc`, `multilingual_large_ctc` |

Multilingual liniya v3'dan **yangiroq**; rus/qozoq/qirg'iz/o'zbek uchun
repo "best-in-class WER" deydi. Biz shu liniyada ishlaymiz.

## Qaror

GigaAM-Multilingual'ni o'zbekchaga fine-tune qilib o'z ASR modelimizni
yasaymiz. Transkripsiya quvuri unga o'tadi; Whisper-uz fine-tune rejasi
(xarita 9-qadam) GigaAM fine-tune bilan almashadi — maqsad o'sha
Gate-4: **etalonda WER ≤ 10%** va baseline GigaAM'dan yaxshi.

### Data (tekshirilgan, litsenziyasi ochiq)

Asosiy korpus — `uzinfocom-edu-ai/uzbek-asr-curated-701h` (Apache-2.0):
658/21/21 soat train/val/test, 6 manba (CV 158s, UzbekVoice 125s, USC 50s,
YouTube yangiliklar/IT/podkast 368s — bizning domen), tozalangan va
bo'lingan. Diqqat: punktuatsiya olib tashlangan, hammasi kichik harf;
YouTube qismining belgilash usuli hujjatlashtirilmagan — hissasi etalonda
ablation bilan tekshiriladi.

Qo'shimcha:

| Manba | Vazifa | Litsenziya |
|---|---|---|
| Common Voice uz v25 (asl matn) | punktuatsiya fazasi | CC0 |
| FLEURS uz | benchmark + punktuatsiya | CC-BY |
| USC to'liq (105s) | zaxira / taqqoslash | ochiq |
| FeruzaSpeech | rad: faqat akademik | — |

Keyingi bosqich (v2): o'z YouTube korpusimizdan yuqori-moslik (GigaAM ↔
turbo konsensus) segmentlarda self-training — YODAS uslubi.

### Punktuatsiya strategiyasi

1. **Asosiy:** CTC lug'atiga `. , ? !` belgilarini qo'shib, punktuatsiyali
   subsetda (CV + FLEURS) o'rgatish — model belgini pauza/ohangdan chiqaradi.
2. **Zaxira (parallel o'lchanadi):** turbo-uzbek punktuatsiyasini alignment
   orqali GigaAM matniga ko'chirish; yoki matn-only punktuatsiya-tiklash
   modeli (kun.uz kabi korpusda o'rgatiladi).
   Qaysi biri etalonda yaxshi — o'sha qoladi.

### Trening rejasi (RTX 5060 Ti 16 GB)

1. Datasetlarni yuklash va yagona manifest formatiga keltirish
2. Etalon: FLEURS test + 2 soatlik o'z YouTube etalonimiz (qo'lda
   tekshiriladi) + WER vositasi (jiwer, MIT)
3. Baseline o'lchov: GigaAM `multilingual_large_ctc`, turbo-uzbek
4. Fine-tune: avval 220M (`multilingual_ctc`) to'liq — 16 GB'ga bemalol
   sig'adi; yaxshi natija bersa 600M (`multilingual_large_ctc`)
   muzlatilgan quyi qatlamlar /
   grad checkpointing bilan; sig'masa ijara GPU (checkpoint HF orqali)
5. Har run: konfig + seed + manifest hash; checkpoint HF private repo

## Muqobillar

- **Whisper-uz fine-tune (asl reja)** — kechiktirildi: baseline'i (turbo)
  GigaAM'dan pastroq boshlanadi; ikki modelli konsensusda ishtirokchi
  bo'lib qoladi.
- **Faqat tayyor GigaAM + tashqi punktuatsiya** — fine-tune o'zini
  oqlamasa shu minimal yo'l qoladi (zaxira).
- **GigaAM v3 (`v3_e2e_ctc`)** — rad: faqat ruscha. E2E variantlari
  punktuatsiyali matn beradi, lekin o'zbekcha bazasi yo'q, biz uni
  tanlasak tilni deyarli noldan o'rgatgan bo'lardik. Ya'ni punktuatsiyani
  v3'dan tekinga olib bo'lmaydi — quyidagi CTC lug'ati yo'li qoladi.
- **KenLM bilan dekodlash** — rad etilmadi, fine-tune'dan keyin ham
  qo'shsa bo'ladi; alohida o'lchanadi.

## Qo'shimcha (2026-08-06): STT alohida mahsulot, ikki modelli tartib

- STT modeli TTS uchun ichki vosita bo'libgina qolmay, **alohida mahsulot**
  sifatida ham chiqariladi — o'zbekcha ochiq STT bozori bo'sh.
- Tartib tasdiqlandi: fine-tune retsepti (punktuatsiya lug'ati, LR, data
  aralashmasi) avval **220M**da arzon sinovlar bilan topiladi; sifat
  yetmasa shu yerda iteratsiya qilinadi. Yakuniy mahsulot modeli tayyor
  retsept bilan **600M**da o'qitiladi. Natijada ikkala o'lcham ham
  mahsulot bo'ladi: kichigi tez/arzon, kattasi eng aniq.
- Baseline (FLEURS test, 650 namuna, 2.13 soat, normallashgan matn):
  turbo-uzbek WER 19.6% / CER 4.4%; GigaAM-large WER 6.7% / CER 1.2%.
  Fine-tune shu 6.7% dan yaxshi bo'lishi va punktuatsiya qo'shishi shart.

## Oqibatlar

- `uztts_train` paketi ochiladi (ASR fine-tune bilan boshlanadi).
- transformers/datasets/jiwer/lightning deplari qo'shiladi (litsenziyalari
  MIT/Apache — mos).
- Etalon to'plami transcribe/filter uchun ham xizmat qiladi (Gate-4).
