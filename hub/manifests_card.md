---
language:
- uz
license: other
license_name: mixed-upstream
license_link: https://huggingface.co/datasets/uzinfocom-edu-ai/uzbek-asr-curated-701h
task_categories:
- automatic-speech-recognition
tags:
- uzbek
- asr
- training-manifests
pretty_name: Uzbek ASR Training Manifests
size_categories:
- 100K<n<1M
---

# Uzbek ASR Training Manifests

The exact training, validation and test splits behind
[`rustam1221/uzbek-asr-gigaam`](https://huggingface.co/rustam1221/uzbek-asr-gigaam):
974 hours of Uzbek speech drawn from seven public corpora, filtered, text-normalized,
and split by speaker.

**No audio is copied.** Each row is a pointer — a parquet file plus a row
index in the upstream dataset — and the training dataloader decodes the audio
when the batch is built. That keeps the whole corpus definition at 200 MB
instead of roughly a terabyte of duplicated audio, and it makes the manifests
a reviewable artifact: a `git diff` shows exactly which rows entered a run.

## Splits

| Split | Rows | Hours | Punctuated hours |
|---|---|---|---|
| train | 620,342 | 916.3 | 795.1 |
| val | 28,543 | 36.5 | 30.9 |
| test | 13,439 | 21.9 | 19.7 |

Punctuated hours matter because punctuation is trained as a second phase over
the subset that still carries `. , ? !` in its original text.

## Sources

| Source | Train hours | Rows | What it is | Upstream license |
|---|---|---|---|---|
| `uzbekvoice` | 490.4 | 416,141 | crowd-read sentences | open |
| `usc` | 97.0 | 97,730 | Uzbek Speech Corpus, read speech | open |
| `yt_it` | 91.7 | 13,103 | YouTube IT talks, Russian/English code-switching | Apache-2.0 |
| `common_voice` | 86.9 | 70,597 | Mozilla Common Voice uz, validated split only | CC0 |
| `yt_news` | 86.5 | 12,399 | YouTube news reads | Apache-2.0 |
| `yt_podcasts` | 56.2 | 8,105 | YouTube podcasts, spontaneous | Apache-2.0 |
| `fleurs` | 7.5 | 2,267 | FLEURS uz, kept as a comparable benchmark | CC-BY |

`usc` carries no punctuation in its source text and is therefore excluded from
the punctuation phase.

### Not in this snapshot

The models were later extended with `yt_gemini` — roughly 660 accepted hours
from `Abduqayum/Uzbek-STT-Dataset-780h`, spontaneous YouTube speech with
LLM-generated transcripts. That source is part of the training recipe and of
every `gemini_*` and `large_*` checkpoint, but its manifest snapshot is not
published here; the published snapshot is the 916-hour base mixture, which is
what every non-`gemini` checkpoint saw.

The 315 benchmark rows are removed from `yt_gemini` before training, keyed by
`exclude_rows.json` in
[`uzbek-asr-benchmark-spontaneous`](https://huggingface.co/datasets/rustam1221/uzbek-asr-benchmark-spontaneous).
Leakage was checked after the exclusion: zero benchmark rows reached the
training set.

## Row format

```json
{"parquet": "uzbekvoice_filtered/data/train-00003-of-00042.parquet", "row": 1517,
 "duration": 4.82, "text": "bugun havo juda issiq boʻldi",
 "text_raw": "Bugun havo juda issiq boʻldi.", "source": "uzbekvoice", "split": "train"}
```

| Field | Meaning |
|---|---|
| `parquet` + `row` | location of the audio in the upstream dataset |
| `audio_filepath` | used instead of `parquet`/`row` where audio is a local file (FLEURS, extracted from tar) |
| `duration` | seconds |
| `text` | normalized: Latin, lowercase, canonical apostrophes (ʻ/ʼ), no punctuation |
| `text_raw` | original text with punctuation, for the punctuation phase |
| `source` | one of the seven source names above |
| `split` | `train` / `val` / `test` |

## Filtering

Applied uniformly to every source, adapted from the `uzbek-asr-curated-701h`
recipe:

- rows containing digits are dropped — spoken form is a text-frontend problem,
  and leaving them in teaches the model to guess
- rows with characters outside the Uzbek Latin alphabet are dropped
- duration outside 0.5–30 s
- more than 18 characters per second, or outside 0.3–2.67 words per second —
  these catch truncated audio and runaway transcripts

## Splitting

Deterministic hash of the speaker id, 96/2/2. Speaker-disjoint splits matter
here: `uzbekvoice` is 45% of the corpus and heavily repeats speakers, so a
random row split would leak voices into validation and flatter the numbers.
Sources without a speaker column are hashed on the row identity. FLEURS keeps
its own dev/test split untouched so its numbers stay comparable with published
results.

## Integrity

```
test_manifest.jsonl   b89219fdb4f5a02f4e9b7cfab5e38972c750c215ddef9c745eb193f9145a46d2
train_manifest.jsonl  c1b32999ef1a2a842d7c7b2bb94899ab45ffb470ef0a1600245ba2e9444f388b
val_manifest.jsonl    403ebb22bf88ed9c75b87134b33ee627f20a4bca912ac893633893deb71fd8e2
```

Every training run records the config, the git commit and these manifest
hashes, so a checkpoint can be traced back to the exact corpus definition it
saw.

## Licensing

Each source keeps its own upstream license; the combination is not
redistributable under a single license, which is why this repository ships
pointers and not audio. Check the upstream terms before using any source
commercially.
