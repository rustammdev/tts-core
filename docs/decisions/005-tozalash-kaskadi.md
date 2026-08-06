# 005 — Tozalash kaskadi: selektiv ajratish, teglash, tashlamaslik

Sana: 2026-08-06 · Holat: qabul qilingan

## Kontekst

Registrdagi 21 kanal (4 072 soat xom) ko'rsatdiki, fon musiqasiz kontent
topish deyarli imkonsiz — vlog va hikoya janrlarida musiqa normadir.
Xarita (03, 05) 100 soat xomdan ~35–45 soat train sifatida qolishini
va separated data train setining 20% idan oshmasligini belgilaydi.
Bizda xom zaxira ehtiyojdan ~4 baravar ko'p (4 072 soat vs 1 000 soat
maqsad), ya'ni eng iflos datani qutqarishga urinish shart emas.

## Qaror

Tozalash to'rt avtomatik bosqichda, har biri idempotent CLI:

1. **segment** — silero-vad (MIT, ONNX backend), 2–20 s bo'laklar,
   sukut va bo'shliqlar shu yerda yo'qoladi; yo'qotish statistikasi
2. **transcribe** — faster-whisper (MIT); `asr_avg_logprob`,
   `asr_compression_ratio`, `lang_prob` diagnostikalari iflos signal
3. **filter** — SNR + musiqa ehtimoli (PANNs/YAMNet, Apache-2.0) + ASR
   diagnostikalari → `quality_tag`: clean / medium / noisy.
   **Data tashlanmaydi — teglanadi**; train'da teg shart (conditioning)
   sifatida beriladi, inference'da `clean` ishlatiladi
4. **separate** — Demucs htdemucs (MIT), **faqat musiqa-belgilangan
   segmentlarga**; `separated=true`; train setida ≤20% (xarita 03)

Musiqa ehtimoli uchun sxemaga yangi maydon qo'shilmaydi: filter bosqichi
uni ichki hisoblab, natijani `quality_tag` ga yig'adi. Kelajakda alohida
maydon kerak bo'lsa — sxema o'zgarishi sifatida alohida so'raladi.

Vositalarni solishtirish mezoni SDR emas: ajratishdan keyin ASR-WER
yaxshilanishi va yakuniy TTS sifat o'lchovi (08-bo'lim).

## Muqobillar

- **Hamma audioni Demucsdan o'tkazish** — rad: har segmentda artefakt
  xavfi, katta compute, xaritaning 20% chegarasi buziladi.
- **Faqat musiqasiz kanallar bilan cheklanish** — rad: bunday kontent
  deyarli yo'q, janr balansini (ayniqsa vlog/hikoya) yo'qotamiz.
- **BS/Mel-RoFormer'ni darhol asosiy qilish** — kechiktirildi: sifati
  Demucsdan yuqori, lekin ochiq checkpointlar litsenziyasi tijoratga
  noaniq; Demucs etalonida o'lchab, litsenziyasi tasdiqlangach bake-off.
- **Umumiy denoise (DeepFilterNet) ni hozir qo'shish** — kechiktirildi:
  avval filter qatlamlari o'lchansin; denoise faqat `medium` qatlamni
  ko'tarish uchun, A/B bilan tekshirilib kiritiladi.

## Oqibatlar

- V0 tartibi o'zgarmaydi: segment → transcribe → filter birinchi
  quriladi, chunki eng katta tozalash yutug'i (sukut, ASR-filtr) shu
  yerda va arzon; separate ular o'lchov bergandan keyin selektiv kiradi.
- Yangi deplar bosqichma-bosqich: silero-vad + onnxruntime (segment),
  faster-whisper (transcribe), PANNs/YAMNet va Demucs (filter/separate).
  Har biri qo'shilishdan oldin litsenziya jadvalda tasdiqlangan.
- `separated` maydoni va 20% chegara data kontraktida allaqachon bor —
  sxema o'zgarishi talab qilinmaydi.
