# 009 — Audio hodisalar ([kulgu], [musiqa]): alohida tagger

**Sana:** 2026-08-11 · **Holat:** qabul qilingan (qurilish keyinroq)

## Kontekst

STT chiqishida va TTS data tayyorlashda nutqdan tashqari hodisalar kerak:
kulgu, musiqa, qarsak. Gemini transkriptlarida bunday teglar yo'q (400
namunada faqat bitta `[BLEEP]`), etalonda ham yo'q — ya'ni ASR modelini
teglar bilan train qilish uchun ham data, ham o'lchov yo'q.

## Qaror

Hodisa aniqlash ASR'dan butunlay ajratiladi — alohida `uztts_events` paketi
(`uztts_text` kabi modeldan mustaqil printsip). Kiruvchi: audio segment,
chiquvchi: hodisalar ro'yxati vaqt oraliqlari bilan.

Ikki bosqichli sxema:

1. **Skrining — CED** (`mispeech/ced-small`, Apache-2.0, AudioSet 527 sinf,
   16 kHz native): har segmentga klip darajasida teglar (kulgu / musiqa /
   qarsak / yo'tal bor-yo'qligi). CPU'da ham yuradi (ONNX yo'li bor).
   Ogohlik: faqat HF/ONNX artefaktlari ishlatiladi — muallifning GitHub
   train-repo'si GPL-3.0, undan kod olinmaydi.
2. **Lokalizatsiya — PretrainedSED `frame_mn10`** (MIT, ICASSP 2025,
   AudioSet-Strong, 3.8M parametr): faqat skrining belgilagan segmentlarda
   hodisaning aniq joyi (40 ms kadr) → transkriptga `[kulgu]` to'g'ri
   nuqtaga qo'yiladi.

Ishlatilish joylari (bitta paket, uch iste'molchi): pipeline `filter`
(musiqali segmentni belgilash), V1 `caption` (uslub izohi), STT
chiqishi/API post-processing.

To'liq run oldidan chegaralar (sigmoid threshold, sinf bo'yicha 0.15–0.5
oralig'i) ~100 qo'lda tekshirilgan segmentda kalibrlanadi. "Speech" sinfi
hamma joyda ~1.0 — hodisalar argmax bilan emas, mustaqil baholanadi.

## Muqobillar

- Teglarni ASR lug'atiga qo'shib qayta train — rad: tegli train data yo'q
  (baribir tagger bilan yasashga to'g'ri kelardi), etalonda teglar yo'q
  (o'lchab bo'lmaydi), WER'ga zarar xavfi.
- Whisper-AT / audio-LLM (Qwen2-Audio) — rad: og'ir, batch teglash uchun
  samarasiz.
- Audio-MAE (Meta) — rad: CC-BY-NC, tijoratni bloklaydi.
- jrgillick/laughter-detection — rad: faqat kulgu, 8 kHz, 2021 dan beri
  qaramliklari eskirgan; PretrainedSED qamrab oladi.

## Oqibatlar

- ASR sifati (WER) hodisa logikasidan mustaqil qoladi; ikkalasi alohida
  yaxshilanadi.
- Yangi qaramliklar litsenziyasi toza: Apache-2.0 (CED) + MIT (PretrainedSED).
- Tijoriy API'da teglar shu paket orqali beriladi; servis qatlami logikani
  takrorlamaydi.
