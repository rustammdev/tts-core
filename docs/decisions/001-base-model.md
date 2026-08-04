# 001 — Baza model: Orpheus 3B fine-tune

## Kontekst

O'zbekcha ifodali TTS kerak. Qo'lda bor data — 30 daqiqadan bir necha soatgacha,
bitta ovoz. Natija kelajakda tijoratda ishlatilishi mumkin, shuning uchun
litsenziya cheklovi qat'iy.

## Qaror

**Orpheus 3B** (Apache-2.0) ni fine-tune qilamiz. Zaxira nomzod —
**Chatterbox-Turbo** (MIT), bake-off uchun.

Litsenziya siyosati: loyihaga faqat **Apache-2.0 yoki MIT** model va kutubxona
kiradi.

Train strategiyasi ikki bosqichli bo'ladi (V1 dan): 1) ko'p spikerli data bilan
til va prosodiya, 2) yakka ovoz bilan tembr.

## Muqobillar

| Variant | Nega yo'q |
|---|---|
| Noldan train | Yuzlab soat data talab qiladi, bizda yo'q |
| XTTS-v2 | CPML litsenziyasi tijoratni bloklaydi |
| F5-TTS | CC-BY-NC — non-commercial |

## Oqibatlar

- Orpheus LLM asosida ishlaydi — text frontend chiqishi tokenizer bilan mos
  bo'lishi kerak, shuning uchun `uztts_text` modeldan mustaqil paket
  (`docs/decisions/002`).
- Har bir yangi kutubxona qo'shilishida litsenziya tekshiriladi; copyleft yoki
  non-commercial bo'lsa avval so'raladi.
- 3B model bitta consumer GPU (RTX 4090 sinfi) da fine-tune qilinadi —
  train konfigi shu chegarani nazarda tutadi.
