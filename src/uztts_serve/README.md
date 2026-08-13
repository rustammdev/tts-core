# uztts_serve

Lokal STT demo UI — birinchi mahsulot (STT) sifatini ko'z bilan baholash
uchun. **Bu public API emas** — faqat 127.0.0.1, autentifikatsiyasiz;
tijoriy servis qatlami keyin alohida quriladi (009-qarordagi printsip:
servis logikani takrorlamaydi, paketlarni chaqiradi).

```bash
UZTTS_DATA_ROOT=~/uztts-data uv run uztts-serve
# http://127.0.0.1:7860
```

Imkoniyatlar:

- Video/audio fayl yuklash yoki YouTube havolasi (yt-dlp)
- Model tanlash: UzSTT (bizniki, `gemini_full_220m/best.pt`), GigaAM 220M,
  GigaAM-large 600M
- Hodisa teglari: `uztts_events` (CED + SED) natijasi transkriptga
  `[kulgu]/[musiqa]` sifatida qo'shiladi, segmentlar jadvalida alohida ham
- Vaqt belgilari (silero-VAD bo'laklari), ishlov statistikasi (realtime ×)

Quvur: ffmpeg → 16 kHz mono → silero-VAD (0.6–25 s bo'laklar) → tanlangan
model bo'laklab transkript → (ixtiyoriy) events tagger butun audio ustida →
`merge_transcript` teglarni bo'lak chegaralariga joylaydi. Chegara: 30 daqiqa.
