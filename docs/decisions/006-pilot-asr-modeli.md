# 006 — Pilot ASR modeli: whisper-large-v3-turbo-uzbek (CT2)

Sana: 2026-08-06 · Holat: qabul qilingan

## Kontekst

Baza `large-v3` pilot bo'laklarida yaroqsiz matn berdi: qardosh til
artefaktlari, tashlab ketilgan nutq, g'o'ldirash. 10 bo'lakli yonma-yon
benchmark (turli kanallar, bir xil sharoit) buni tasdiqladi va ikkita
o'zbekcha fine-tune'ni solishtirdi.

## Qaror

Pilot transkripsiyasi uchun `hostmepanda/whisper-large-v3-turbo-uzbek-ct2`
(MIT) — `uztts-transcribe` ning default modeli:

- sifat baza modeldan keskin yaxshi, navai-medium bilan taqqoslama
- **tinish belgilari va bosh harflarni saqlaydi** — ifodali TTS uchun
  prosodiya signali; navai-medium hammasini kichik harfda, belgisiz beradi
- tayyor CT2 format — faster-whisper'ga o'zgarishsiz tushadi, turbo
  arxitektura tezroq
- MIT litsenziya

`navai-uz/whisper-medium-uzbek` (Apache-2.0, e'lon qilingan WER 7.4–14.1)
rad etilmadi — xaritadagi **ikki modelli konsensus**ning ikkinchi modeli
sifatida saqlanadi: boshqa arxitektura va boshqa train dataseti, aynan
konsensusga kerakli xilma-xillik.

## Muqobillar

- **large-v3 (baza)** — rad: benchmarkda eng yomon, foydalanuvchi ham
  mustaqil sezdi.
- **navai-medium'ni asosiy qilish** — rad (hozircha): punktuatsiya yo'q,
  transformers orqali sekinroq, CT2 konversiya talab qiladi; sifati
  taqqoslama bo'lgani uchun konsensus roliga qoldirildi.
- **YouTube subtitrlari** — rad (003 kuchda): vaqt mosligi bo'sh,
  diagnostika yo'q, sifati kanalga qarab o'ynaydi.

## Oqibatlar

- Yakuniy model tanlovi Gate-4'da o'lchov bilan: 2 soatlik qo'lda
  tekshirilgan etalon, WER ≤ 10%; o'z fine-tune'imiz (USC + CV + FLEURS)
  shu etalon ustida turbo-uzbek bilan solishtiriladi.
- `lang_prob` diagnostikasi fine-tune'langan modelda ham yumshoq signal
  bo'lib qoladi; filtr asosan logprob/compression'ga tayanadi.
- Benchmark skripti scratchpad'da; etalon to'plami (10-qadam) chiqqanda
  rasmiy WER vositasiga aylantiriladi.
