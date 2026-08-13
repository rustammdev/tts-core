# 010 — WER o'lchovida klitika kanonizatsiyasi

**Sana:** 2026-08-13 · **Holat:** qabul qilingan

## Kontekst

Etalon v2 xato dumi tahlili (`docs/eval/etalon-v2-2026-08-11.md`) ko'rsatdi:
substitutsiyalarning ~10% i so'z emas, yozilish konventsiyasi farqi —
`qildikda ↔ qildik da`, `mingta ↔ ming ta`, `keldimi ↔ keldi mi`. Etalon
(Gemini matni) va model chiqishi bu shakllarni har xil yozadi; WER buni
"xato" deb sanaydi.

## Qaror

O'lchov (normalize qilingan) fazosida ikkala tomonga ham bir xil
deterministik qoida qo'llanadi — `uztts_text.join_clitics`:

- Mustaqil turgan `da, ku, chi, mi, yu, ya, ta` tokenlari oldingi so'zga
  qo'shiladi. Kanonik shakl — **qo'shib yozilgan** (`qildikda`, `beshta`,
  `keldimi`).
- `u` (olmosh bilan to'qnashadi: "men u kitobni") va `a`/`e` (mustaqil
  undovlar) ataylab ro'yxatga kirmagan — noto'g'ri qo'shilish xavfi
  foydasidan katta.

Etalon hisobotlarida ikkala raqam ham beriladi: oddiy WER (tarixiy
solishtirish uchun) va **kanonik WER (asosiy, Gate-4 shu bo'yicha)**.

O'lchangan ta'sir (gemini_full_220m, etalon v2): 13.93 → **13.35%**
(ref 65/311, hyp 70/311 segmentda o'zgarish). Avvalgi ~3–4 punktlik taxmin
oshirib yuborilgan — qolgan "yaqin-xato" turkumi (bilaydim↔biladim,
tagidan↔tegidan) talaffuz-imlo variantlari bo'lib, lug'atsiz deterministik
birxillashtirib bo'lmaydi; ular haqiqiy model/data masalasi.

## Muqobillar

- Ajratib yozishni kanon qilish — rad: `qildikda` dan `da` ni lug'atsiz
  ajratib bo'lmaydi (`uyda` dagi kelishik bilan to'qnashadi); qo'shish
  yo'nalishi deterministik.
- Train targetlarini ham kanonizatsiya qilish — hozircha yo'q: punktuatsiyali
  matnda defis konventsiyasi (`keldi-ku`) bilan aralashadi; alohida
  baholanadi.
- Talaffuz variantlari lug'ati — keyinga: qo'lda tasdiqlangan juftliklar
  to'plami sifatida, xatoni yashirib qo'ymasligi uchun ehtiyot talab.

## Oqibatlar

- Gate-4 o'lchovi endi kanonik WER: joriy eng yaxshi **13.4%** (≤10% maqsad).
- `uztts_text` ASR o'lchoviga birinchi marta ulandi — TTS frontend va ASR
  bir xil matn konventsiyasini ko'radi.
- Etalon gipotezalari `hyps_gemini_full.jsonl` sifatida saqlanadi — keyingi
  o'lchovlar transkripsiyasiz, sekundlarda.
