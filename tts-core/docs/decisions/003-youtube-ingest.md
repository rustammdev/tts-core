# 003 — YouTube ingest

## Kontekst

Birinchi data manbai — YouTube. Kirish: URL ro'yxati. Chiqish: har video uchun
24 kHz mono PCM WAV + metadata, keyingi bosqich (`transcribe`) uchun tayyor.

## Qaror

**Subtitrlar train matni sifatida ishlatilmaydi.** Avtomatik subtitrlarda tinish
belgilari yo'q, o'zbekchada xatolik darajasi yuqori, timestamp'lar taxminiy;
qo'lda yozilganlari esa ko'pincha qisqartirilgan yoki qayta ifodalangan. Matn
manbai — har doim ASR. Subtitr faqat arzon signal: `meta.json` dagi
`uz_subtitles` maydoni (`manual` / `automatic` / `null`) videoda o'zbekcha nutq
borligini ko'rsatadi.

**Ingest videoni o'zi tashlab yubormaydi.** Signal yoziladi, qaror keyingi
bosqichda qabul qilinadi — shovqinli datani belgilash siyosati bilan bir xil
(CLAUDE.md §4).

**`.done` marker o'zi qo'riqlaydigan papka ichida.** `ingest` uchun
`data/raw/<video_id>/.done`. Dastlabki reja markerni `data/interim/` da tutishni
taklif qilgan edi; unda `data/raw` o'chirilsa marker qolib ketadi va bosqich
"bajarilgan" deb o'tkazib yuboriladi. Marker chiqish bilan birga yo'qolsin.

**Idempotentlik URL dan boshlanadi.** `video_id_from_url` URL'dan id ni ajratadi
(`watch?v=`, `youtu.be/`, `shorts/`, `embed/`, `live/`), shuning uchun
tugallangan videolar uchun tarmoqqa umuman chiqilmaydi.

**Tashqi jarayon faqat ffmpeg.** yt-dlp Python API sifatida ishlatiladi.
ffmpeg'ning Python API'si yo'q — u subprocess, va `--urls` o'qilishidan oldin
PATH'da borligi tekshiriladi (aks holda har video bir xil xato bilan tushardi).

## Muqobillar

| Variant | Nega yo'q |
|---|---|
| yt-dlp'ni subprocess sifatida chaqirish | Xato holatlari matn parsing orqali; API to'g'ridan-to'g'ri obyekt qaytaradi |
| Konvertatsiyani yt-dlp postprocessor'iga berish | `postprocessor_args` kalitlari versiyaga bog'liq; alohida ffmpeg chaqiruvi oshkora va testlanadi |
| Subtitrni ASR o'rniga ishlatish | Yuqoridagi sabab — sifat train uchun yetarli emas |
| Xato bo'lganda to'xtash | 200 ta URL'dan bittasi yopiq bo'lsa butun partiya yo'qoladi |

## Oqibatlar

- **ffmpeg tizim talabi** bo'ldi (`apt install ffmpeg`). `uztts-ingest` uni
  topmasa 2 kodi bilan darhol chiqadi.
- Yuklangan xom oqim (`source.*`) konvertatsiyadan keyin o'chiriladi — 1 soatlik
  video uchun ~50 MB ortiqcha saqlanmaydi. Sample rate o'zgarsa qayta yuklanadi.
- Tushgan videolar `data/raw/_failed.jsonl` ga qo'shib boriladi; buyruq oxirida
  chiqish kodi 1 bo'ladi, lekin qolgan videolar to'liq ishlanadi.
- Playlist URL'lari kengaytirilmaydi (`noplaylist: True`) — bitta URL bitta
  video. Kanal bo'yicha yig'ish kerak bo'lsa alohida qadam.
