# 007 — O'z ASR modelimiz: GigaAM'ni o'zbekchaga fine-tune qilish

Sana: 2026-08-06 · Holat: qabul qilingan

## Kontekst

Eshitish benchmarkida (006 davomi) GigaAM-Multilingual (MIT) o'zbekchada
barcha Whisper variantlaridan ustun chiqdi — foydalanuvchi tasdiqladi.
Kamchiliklari: ayrim so'zlarda xato, punktuatsiya va bosh harflar yo'q.
Repo fine-tuning'ni rasman qo'llaydi (PyTorch Lightning, VRAM'ga
moslashgan misollar, charwise CTC).

## Qaror

GigaAM-Multilingual'ni o'zbekchaga fine-tune qilib o'z ASR modelimizni
yasaymiz. Transkripsiya quvuri unga o'tadi; Whisper-uz fine-tune rejasi
(xarita 9-qadam) GigaAM fine-tune bilan almashadi — maqsad o'sha
Gate-4: **etalonda WER ≤ 10%** va baseline GigaAM'dan yaxshi.

### Data (tekshirilgan, litsenziyasi ochiq)

| Manba | Soat | Punktuatsiya | Litsenziya |
|---|---|---|---|
| USC | 105 | tekshiriladi | ochiq (tadqiqot maqolasi bilan) |
| Common Voice uz v25 | 101 validated | bor | CC0 |
| FLEURS uz | ~12 (train qismi) | bor | CC-BY |
| FeruzaSpeech | — | — | rad: faqat akademik |

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
3. Baseline o'lchov: GigaAM large_ctc, turbo-uzbek
4. Fine-tune: avval 220M (`ctc`) to'liq — 16 GB'ga bemalol sig'adi;
   yaxshi natija bersa 600M (`large_ctc`) muzlatilgan quyi qatlamlar /
   grad checkpointing bilan; sig'masa ijara GPU (checkpoint HF orqali)
5. Har run: konfig + seed + manifest hash; checkpoint HF private repo

## Muqobillar

- **Whisper-uz fine-tune (asl reja)** — kechiktirildi: baseline'i (turbo)
  GigaAM'dan pastroq boshlanadi; ikki modelli konsensusda ishtirokchi
  bo'lib qoladi.
- **Faqat tayyor GigaAM + tashqi punktuatsiya** — fine-tune o'zini
  oqlamasa shu minimal yo'l qoladi (zaxira).
- **KenLM bilan dekodlash** — rad etilmadi, fine-tune'dan keyin ham
  qo'shsa bo'ladi; alohida o'lchanadi.

## Oqibatlar

- `uztts_train` paketi ochiladi (ASR fine-tune bilan boshlanadi).
- transformers/datasets/jiwer/lightning deplari qo'shiladi (litsenziyalari
  MIT/Apache — mos).
- Etalon to'plami transcribe/filter uchun ham xizmat qiladi (Gate-4).
