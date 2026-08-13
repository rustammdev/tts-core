# uztts_events

Audio hodisalar ([kulgu], [musiqa], [qarsak], [yoʻtal]) — ASR'dan mustaqil
tagger (009-qaror). Kiruvchi: audio segment; chiquvchi: hodisalar ro'yxati
vaqt oraliqlari bilan; merger ularni transkriptga deterministik joylaydi.

## Arxitektura

| Bosqich | Modul | Model |
|---|---|---|
| 1. Skrining (klip darajasida) | `screen.py` | CED `mispeech/ced-small`, Apache-2.0 |
| 2. Lokalizatsiya (kadr darajasida) | `localize.py` | PretrainedSED `frame_mn10`, MIT |
| 3. Birlashtirish (sof funksiya) | `merge.py` | — |

Skrining hamma segmentda yuradi (yengil, CPU'da ham); lokalizatsiya faqat
skrining belgilagan segmentlarda. Hech qaysi bosqich ASR modeliga bog'lanmaydi.

## Kontrakt

```python
from uztts_events import AudioEvent, EventLabel, Word, merge_transcript

events = [AudioEvent(label=EventLabel.LAUGHTER, start=1.2, end=2.1, score=0.93)]
words = [Word(text="Salom", start=0.0, end=0.4), ...]
merge_transcript(words, events)
# "Salom do'stlar, [kulgu] bugun boshlaymiz."
```

- `consolidate_events` — bir xil tegdagi yaqin oraliqlarni qo'shadi
  (`merge_gap`, default 0.5 s), qisqalarini tashlaydi (`min_duration`, 0.2 s).
- Teg so'z chegarasiga qo'yiladi: hodisa boshlanishidan oldin tugagan
  so'zlardan keyin.
- `strip_event_tags` — WER o'lchovidan oldin teglarni olib tashlash uchun.

Chegaralar (threshold) sinf bo'yicha `configs/events.yaml` da bo'ladi va
~100 qo'lda tekshirilgan segmentda kalibrlanadi — kalibrsiz default'lar
bilan katta run qilinmaydi.

## Litsenziya ogohliki

CED og'irliklari HF'da Apache-2.0, lekin muallifning GitHub train-reposi
(RicherMans/CED) GPL-3.0 — undan kod ko'chirilmaydi, faqat HF/ONNX
artefaktlari ishlatiladi.
