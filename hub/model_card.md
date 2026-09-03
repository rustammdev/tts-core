---
language:
- uz
license: mit
library_name: transformers
pipeline_tag: automatic-speech-recognition
tags:
- automatic-speech-recognition
- uzbek
- gigaam
- ctc
- punctuation
base_model: ai-sage/GigaAM-Multilingual
datasets:
- rustam1221/uzbek-asr-train-manifests
- rustam1221/uzbek-asr-benchmark-spontaneous
- google/fleurs
metrics:
- wer
- cer
---

# Uzbek ASR — GigaAM-Multilingual CTC fine-tunes

Open Uzbek speech recognition **with punctuation**, fine-tuned from
[GigaAM-Multilingual](https://huggingface.co/ai-sage/GigaAM-Multilingual) (MIT)
on 1.6k hours of Uzbek speech.

This repository holds the full ablation ladder — six checkpoints, each one a
question that was asked and answered — not just the final weights. The corpus
definition and the evaluation set are published alongside it, so every number
below can be checked rather than taken on trust:

- [`uzbek-asr-train-manifests`](https://huggingface.co/datasets/rustam1221/uzbek-asr-train-manifests) — the exact rows every run saw, with hashes
- [`uzbek-asr-benchmark-spontaneous`](https://huggingface.co/datasets/rustam1221/uzbek-asr-benchmark-spontaneous) — the human-verified test set, with per-clip predictions

## Results

Two test sets, because one of them flatters everybody.

### Spontaneous speech — the number that matters

311 human-verified clips of real Uzbek conversation
([benchmark](https://huggingface.co/datasets/rustam1221/uzbek-asr-benchmark-spontaneous)).
Corpus-level WER/CER, text lowercased, apostrophes canonicalized, punctuation
stripped on both sides.

| Model | Params | WER | CER |
|---|---|---|---|
| **`large_full_600m`** (this repo) | 600M | _measurement running_ | |
| **`gemini_full_220m`** (this repo) | 220M | **13.9%** | **3.8%** |
| `gemini_bench_220m` (this repo) | 220M | 14.0% | 3.8% |
| `punct_220m` (this repo) | 220M | 15.0% | 4.1% |
| `base_220m` (this repo) | 220M | 15.5% | 4.5% |
| GigaAM-large — untuned baseline | 600M | 16.1% | 5.8% |
| GigaAM — untuned baseline | 220M | 17.1% | 5.4% |
| whisper-large-v3-turbo-uzbek | 809M | 39.6% | 19.4% |

Under canonical scoring — Uzbek clitics joined on both sides, see below —
`gemini_full_220m` is **13.4%**.

### FLEURS (read speech)

650 samples, 2.13 h. Included for comparability with published results, not
because it measures the product.

| Model | Params | WER | CER |
|---|---|---|---|
| GigaAM-large — untuned baseline | 600M | 6.7% | 1.2% |
| `base_220m` (this repo) | 220M | 9.4% | 1.8% |
| GigaAM — untuned baseline | 220M | 9.5% | 1.7% |
| `punct_220m` (this repo) | 220M | 9.6% | 1.9% |
| whisper-large-v3-turbo-uzbek | 809M | 19.6% | 4.4% |

Read the two tables together and the point of the second one becomes clear:
on FLEURS the fine-tune looks like a rounding error, on real speech it is
1.6 points. The 600M checkpoints were never scored on FLEURS — by then the
question had stopped being interesting.

### Validation

Held-out, speaker-disjoint, punctuated. Same split for every run below, so
these are the numbers the ablation was actually steered by.

| Run | Params | Val WER |
|---|---|---|
| **`large_full_600m`** | 600M | **10.32%** |
| `large_bench_600m` | 600M | 11.08% |
| `gemini_full_220m` | 220M | 11.21% |
| `gemini_bench_220m` | 220M | 11.91% |
| `punct_220m` | 220M | 13.98% |

## Which checkpoint

| Variant | Path | Params | Use it when |
|---|---|---|---|
| **large-600m** | `checkpoints/large_full_600m/best.pt` | 600M | accuracy matters — the default |
| **220m** | `checkpoints/gemini_full_220m/best.pt` | 220M | throughput or a smaller GPU matters |
| 220m-no-youtube-llm | `checkpoints/punct_220m/best.pt` | 220M | you need weights trained only on human-transcribed audio |

The remaining checkpoints (`base_220m`, `gemini_bench_220m`,
`large_bench_600m`) are ablation stages, kept so the ladder below is auditable.

## Usage

```bash
pip install torch transformers huggingface_hub soundfile numpy
wget https://huggingface.co/rustam1221/uzbek-asr-gigaam/raw/main/inference.py

python inference.py interview.mp3
python inference.py interview.mp3 --variant 220m --device cpu
```

```python
from inference import UzbekAsr

asr = UzbekAsr.from_hub("large-600m")
print(asr.transcribe("interview.mp3"))
# Assalomu alaykum, bugungi suhbatimizda ta'lim tizimi haqida gaplashamiz.
```

`inference.py` is standalone: it downloads the checkpoint, rebuilds the
extended CTC head, restores the punctuation vocabulary, splits audio past the
model's 30 s limit at its quietest point, and applies sentence-start
capitalization.
`ffmpeg` must be on `PATH` — any audio or video container works.

## How it was built

### Why GigaAM and not Whisper

Whisper is the obvious starting point and it was the wrong one for Uzbek. On
spontaneous speech the best available Uzbek Whisper fine-tune
(`whisper-large-v3-turbo-uzbek`) scores 39.6% WER, against 16.1% for the
untuned GigaAM-large. Fine-tuning cannot close a gap that size; the base model
choice decided the outcome before any training ran.

GigaAM-Multilingual is character-level CTC, which also made punctuation cheap —
see below — and it is MIT-licensed, so the result stays commercially usable.

### Punctuation as four extra CTC rows

Most Uzbek ASR output is an unpunctuated lowercase stream, which is close to
unusable as a product. Instead of bolting on a separate punctuation-restoration
model, `. , ? !` were added directly to the CTC vocabulary: the head's output
convolution gains four rows, initialized to zero weight and **bias −8**, so the
new symbols start effectively silent and the warm-started model does not
regress while it learns them.

Every punctuation run therefore reports two validation numbers: WER with
symbols, and WER with symbols stripped. The second one is the regression
guard, and it held — 8.26% before the punctuation phase, 8.03% after. The
model learned to punctuate for free, from prosody, and got slightly better at
words while doing it.

Sentence-start capitalization is then a deterministic rule over the punctuation
(`inference.py`), not something the model has to spend capacity on.

### The ablation ladder

Each run warm-starts from the one above it. Validation is the held-out
speaker-disjoint split of the training corpus.

| Run | Params | Warm start | Training data | Steps | Val WER |
|---|---|---|---|---|---|
| `base_220m` | 220M | GigaAM 220M | 916 h, punctuation stripped | 13,000 / 14,000 | 8.26%¹ |
| `punct_220m` | 220M | `base_220m` | 795 h punctuated | 3,500 / 4,000 | 13.98% (8.03% stripped) |
| `gemini_bench_220m` | 220M | `punct_220m` | + 120 h LLM-transcribed YouTube | 3,500 / 4,000 | 11.91% |
| `gemini_full_220m` | 220M | `gemini_bench_220m` | + 660 h LLM-transcribed YouTube | 5,500 / 8,000 | 11.21% |
| `large_bench_600m` | 600M | GigaAM-large 600M | 795 h + 120 h | 4,000 / 4,000 | 11.08% |
| **`large_full_600m`** | **600M** | `large_bench_600m` | **795 h + 660 h** | **6,000 / 8,000** | **10.32%** |

¹ not comparable with the rows below — that run had no punctuation in its
targets.

All runs: character-level CTC, bf16, SpecAugment (2 frequency masks × 8 bins,
2 time masks × 5%), AdamW with weight decay 0.01, gradient clipping at 1.0,
cosine schedule after warmup, fixed seed, on a single RTX 5060 Ti 16 GB.
Batches are measured in audio-seconds rather than utterances so step size stays
constant across a corpus with 0.5–30 s clips: 480 audio-seconds per step for
the 220M runs, 468 for the 600M run.

`large_full_600m` was **stopped at step 6,000 of a planned 8,000**. The
published checkpoint is the best validation point reached; the last two
thousand steps were never run, so the 600M line is a floor rather than a
converged result.

### What the ladder showed

**FLEURS does not measure this task.** After the base fine-tune, FLEURS said
9.4% against the baseline's 9.5% — no gain worth reporting. On spontaneous
speech the same checkpoint went from 17.1% to 15.5%. FLEURS is read
audiobook-style speech; if it had been the only benchmark, the entire training
run would have looked like a waste. This is why the
[spontaneous benchmark](https://huggingface.co/datasets/rustam1221/uzbek-asr-benchmark-spontaneous)
was built before the training budget was spent.

**Data returns saturate fast, and cheaply-measured.** LLM-transcribed YouTube
speech was the obvious way to buy in-domain data. Rather than train on all 660
hours and find out, a 120-hour capped run went first: it bought 1.0 WER point.
The full 660 hours then bought roughly 0.1 more on the benchmark. Five and a
half times the data, a twentieth of the gain — the cap experiment paid for
itself several times over.

**Capacity picked up where data stopped.** With 220M saturated, 600M was the
remaining lever, and it moved the number the data no longer could. Note also
that the 600M model reached 11.08% validation WER in 4,000 steps from the raw
base model — better than the 220M ladder managed in 22,000.

**Part of the remaining gap was measurement, not recognition.** Uzbek clitics
(`da`, `ku`, `chi`, `mi`, …) are written both joined and separated in real
text, and neither spelling is wrong. Scoring them as errors charged the model
0.6 WER points for an orthographic convention. Canonicalizing both sides before
scoring is the honest measurement; it was worth checking, and it was worth
knowing that it explained only 0.6 points rather than the 3–4 first estimated.

## Training data

916 hours of training audio across seven public corpora — UzbekVoice, Uzbek
Speech Corpus, Common Voice, FLEURS, and three YouTube collections (974 h
including the validation and test splits) — plus roughly 660 hours of
LLM-transcribed spontaneous YouTube speech. Filtering, text normalization,
speaker-disjoint splitting and the exact row-level manifests are documented in
[`uzbek-asr-train-manifests`](https://huggingface.co/datasets/rustam1221/uzbek-asr-train-manifests).

The 315 benchmark rows are excluded from training and the exclusion was
verified — zero leaked rows.

## Limitations

- **`large_full_600m` is not converged** — the run stopped at 6,000 of 8,000
  planned steps.
- **The error tail is overlapping speech.** The worst 30 benchmark clips carry
  24% of all errors, and they are 25–30 s multi-speaker segments where turns
  overlap. Half the substitution errors are within two characters of the
  reference — spoken-form spelling variants rather than misheard words.
- **Proper nouns stay lowercase.** Only sentence-start capitalization is
  applied; a named-entity layer is not implemented.
- **Comma placement follows written grammar**, inherited from the training
  transcripts, and does not always match the speaker's pauses.
- **Code-switching into Russian and English** is the weakest category after
  overlap: roughly 7% of substitutions.
- **No language model and no streaming.** Greedy CTC decoding over whole
  utterances; a KenLM pass and a streaming decoder are both open.

## Repository layout

```
checkpoints/<run>/best.pt   weights, plus the run config and vocabulary
configs/<run>.yaml          the exact training config each run used
inference.py                standalone transcription script
requirements.txt
```

Each `best.pt` carries the full training config, step count, best validation
WER and character vocabulary, so a checkpoint is self-describing — the
`configs/` files are extracted from the checkpoints themselves, which is how
the 600M configs survived the loss of the training machine.

## License and provenance

Weights are MIT, inherited from the GigaAM-Multilingual base model. The
training data is a mixture of upstream licenses (CC0, CC-BY, Apache-2.0 and
YouTube-derived material) — see the manifest repository before commercial use.

## Citation

```bibtex
@misc{uzbek_asr_gigaam,
  title  = {Uzbek ASR: GigaAM-Multilingual CTC fine-tunes with punctuation},
  author = {Nuriddinov, Rustamjon},
  year   = {2026},
  url    = {https://huggingface.co/rustam1221/uzbek-asr-gigaam}
}
```
