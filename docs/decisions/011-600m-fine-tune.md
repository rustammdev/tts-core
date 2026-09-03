# 011 — 600M fine-tune (GigaAM-large)

Sana: 2026-08-13 · Holat: qabul qilingan · Yozilgan: 2026-09-03 (kechikkan
qayd — runlar o'z vaqtida hujjatlashtirilmagan, manba: HF checkpointlar
ichidagi konfig va `best_wer`)

## Kontekst

220M liniyasida data qaytimi so'ndi (`docs/eval/etalon-v2-2026-08-11.md`):
Gemini datasetning birinchi 120 soati −1.0 punkt bergan, qolgan 540 soat
etalonda ~0 qo'shgan. Xato dumi tahlili uch dastakni ko'rsatgan edi; imlo
kanonizatsiyasi (010) −0.6 punkt berdi, ya'ni qolgan masofa haqiqiy
model/data ishi.

007-qarordagi shart shu payt bajarildi: "600M train faqat kutilayotgan yutuq
2x sekinlikni oqlasa boshlanadi". 220M'da data tugagach, sig'im yagona
qolgan dastak bo'ldi.

## Qaror

`multilingual_large_ctc` (600M) ustida **aynan shu retsept** bilan ikki
bosqichli train:

1. `large_bench_600m` — bazadan to'g'ridan-to'g'ri, punktuatsiya lug'ati
   bilan, Gemini 120 soatga cheklangan (`source_hours`), 4000 qadam,
   LR 3e-5 → 3e-6, batch 60 s × 8 = 480 audio-soniya/qadam.
2. `large_full_600m` — `large_bench_600m` dan davom, to'liq data
   (795 s punktuatsiyali + ~660 s Gemini), LR 2e-5 → 2e-6,
   batch 36 s × 13 = 468 audio-soniya/qadam.

16 GB VRAM'ga sig'dirish uchun batch kichraytirilib accum oshirildi; qolgan
giperparametrlar 220M retseptidan o'zgarmadi — solishtirish toza bo'lsin.

## Natijalar

Val (spiker bo'yicha ajratilgan, punktuatsiyali):

| Run | Qadam | Val WER |
|---|---|---|
| `gemini_bench_220m` | 3500 / 4000 | 11.91% |
| `gemini_full_220m` | 5500 / 8000 | 11.21% |
| `large_bench_600m` | 4000 / 4000 | 11.08% |
| **`large_full_600m`** | **6000 / 8000** | **10.32%** |

Ikki kuzatuv:

- 600M bazadan **4000 qadamda** 11.08% ga chiqdi — 220M liniyasi 22000
  qadamda 11.91% ga yetgan edi. Sig'im data bilan almashmaydi.
- Sber'ning "600M atigi 5–10% nisbiy yutuq" tajribasi (007 qo'shimchasi)
  o'zbekchada tasdiqlanmadi: 11.21 → 10.32 bu 7.9% nisbiy, lekin run
  tugallanmagan holatda.

## Cheklov: run tugallanmagan

`large_full_600m` **8000 qadam rejadan 6000-qadamda to'xtatilgan**. HF'dagi
checkpoint — shu nuqtagacha bo'lgan eng yaxshi val natijasi. Ya'ni 600M
raqami **pol, shift emas**; qolgan 2000 qadam hech qachon ishlamagan.

## Oqibatlar

- Mahsulot modeli 600M bo'ladi, 220M tez/arzon variant sifatida qoladi.
- Gate-4 (etalonda WER ≤ 10%) holati `docs/eval/` da o'lchov bilan
  yangilanadi.
- Runlar o'z vaqtida hujjatlashtirilmagani xato bo'ldi: lokal
  `~/uztts-data/asr/runs/` yo'qolgach, journal fayllari ham yo'qoldi va
  raqamlar faqat checkpoint metadatasidan tiklandi. Bundan keyin har run
  tugagach `docs/eval/` ga bir qator yoziladi.
