# 008 — Etalon v2: Gemini-transkriptli datasetdan namuna

**Sana:** 2026-08-11 · **Holat:** qabul qilingan

## Kontekst

Birinchi etalon (555 segment, `rustam1221/uz-stt-etalon`) o'z pipeline'imiz
chiqargan transkriptlar ustida qo'lda tahrir talab qilardi; 40+ segmentdan
keyin xatolar zichligi oshib, tahrir juda sekinlashdi.

`Abduqayum/Uzbek-STT-Dataset-780h` (YouTube: audiokitob, podkast, IT; Gemini
transkript) tekshiruvi: 1100 matn namunasida kirill va raqam deyarli yo'q,
punktuatsiya izchil; 54 klipda GigaAM-large bilan kelishuv — toza manbalarda
WER 0–10%, kod-almashinuvli podkastlarda 37–48% (aybning katta qismi
GigaAM'da). Gemini matni tahrir uchun ancha yaxshi boshlang'ich nuqta.

## Qaror

Etalon v2 shu datasetdan olinadi: 315 segment, 2.0 soat (Gate-4 talabi),
seed'li tasodifiy tanlov, musiqa/kirill/raqamli segmentlar chiqarilgan.
Har segmentga `asr_wer` (GigaAM-large bilan kelishmaslik) yozilgan.

Qo'lda tekshiruv faqat `asr_wer ≥ 0.18` bo'lgan 121 segmentda (42 daqiqa),
eng katta farqdan boshlab tartiblangan. Asos: dastlabki 7 tasodifiy segment
(shu jumladan `asr_wer` 0.26 bo'lgani ham) 100% to'g'ri chiqdi — past
kelishmaslikda Gemini xatosi ehtimoli juda kichik. Qolgan 194 segment Gemini
matni bilan avtomatik qabul qilinadi. Yakuniy etalon: tahrir bo'lsa —
tahrir, bo'lmasa Gemini matni; `discard` chiqariladi.

- Tahrir UI: `rustam1221/uz-stt-etalon-v2-ui` (private static Space)
- Tahrirlar: `rustam1221/uz-stt-etalon-v2` dataset, `corrections.jsonl`
- Lokal nusxa: `~/uztts-data/asr/etalon/v2_gemini/`
- Etalon v1 saqlanadi, vaqt bo'lganda tekshiruv davom etadi

## Muqobillar

- V1'ni tahrirlashda davom etish — rad: xato zichligi tufayli sekin.
- Gemini yorliqlarini tekshiruvsiz etalon qilish — rad: avtomatik yorliq
  etalonga yaramaydi, dataset kartasi ham xatolarni tan oladi.
- Konsensus bo'yicha kelishmagan segmentlarni tashlash — rad: test faqat
  oson segmentlarda qolib, WER sun'iy pasayadi.

## Oqibatlar

- Kanal metadata yo'q — train to'plamlari (jumladan islomov to'plamlari va
  shu datasetning o'zi train'ga olinsa) bilan kanal darajasida ustma-tushishni
  tekshirib bo'lmaydi. Bu dataset train'ga qo'shilsa, etalonga olingan 315
  satr (`source_row` saqlangan) train'dan chiqariladi; kanal darajasidagi
  leakage xavfi ochiq qoladi.
- Litsenziya `other` — faqat ichki baholash uchun, tarqatilmaydi.
- Avtomatik qabul qilingan 194 segmentda Gemini va GigaAM bir xil xato
  qilgan (korrelyatsiyalangan xato) holatlar tekshirilmay qoladi — ongli
  qabul qilingan xavf, tahrir vaqtini ~3 barobar tejaydi.
