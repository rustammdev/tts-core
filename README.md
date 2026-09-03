# uz-tts — Uzbek speech stack

Speech recognition and, next, expressive speech synthesis for Uzbek — a
language with roughly 35 million speakers and almost no open speech tooling.

The ASR half is done and public. Six fine-tunes of GigaAM-Multilingual, a
974-hour curated training corpus, and a hand-verified benchmark for
spontaneous speech, all published:

| | |
|---|---|
| Model | [`rustam1221/uzbek-asr-gigaam`](https://huggingface.co/rustam1221/uzbek-asr-gigaam) |
| Training manifests | [`rustam1221/uzbek-asr-train-manifests`](https://huggingface.co/datasets/rustam1221/uzbek-asr-train-manifests) |
| Benchmark | [`rustam1221/uzbek-asr-benchmark-spontaneous`](https://huggingface.co/datasets/rustam1221/uzbek-asr-benchmark-spontaneous) |

## Results

311 human-verified clips of real Uzbek conversation — podcasts and interviews
with overlapping speakers, fillers and Russian code-switching. Corpus-level
WER, punctuation stripped, apostrophes canonicalized.

| Model | Params | WER | CER |
|---|---|---|---|
| **`large_full_600m`** (ours) | 600M | **12.2%** | **3.4%** |
| `gemini_full_220m` (ours) | 220M | 13.9% | 3.8% |
| GigaAM-large — untuned baseline | 600M | 16.1% | 5.8% |
| whisper-large-v3-turbo-uzbek | 809M | 39.6% | 19.4% |

Our models also emit `. , ? !` and sentence capitalization, which none of the
baselines do. Target is 10% WER; the 600M run stopped at 6,000 of 8,000
planned steps, so that number is a floor.

## What is in here

| Package | What it does | Status |
|---|---|---|
| `uztts_asr` | GigaAM fine-tune: corpus prep, streaming trainer, eval, Hub release | 6 checkpoints shipped |
| `uztts_data` | data contract, manifests, YouTube ingest, VAD segmentation, transcription | in use |
| `uztts_text` | Uzbek text frontend: Cyrillic→Latin, apostrophes, numbers, clitics | MVP |
| `uztts_events` | audio event tags (`[kulgu]`, `[musiqa]`) from a separate classifier | MVP |
| `uztts_serve` | local STT demo UI — upload or YouTube link, model picker, timestamps | demo |
| `hub/` | the published Hugging Face cards and standalone scripts | — |

TTS training (`uztts_train`, `uztts_eval`) is the next phase; the data
pipeline and text frontend were built model-agnostic for it.

## Three things worth looking at

**The benchmark exists because the public one measures the wrong thing.**
FLEURS is read speech.
After the first fine-tune, FLEURS said 9.4% against the baseline's 9.5% — no
gain worth reporting — while on real conversation the same checkpoint went
17.1% → 15.5%. So a 2-hour spontaneous-speech test set was built and
hand-verified before any serious training budget was spent. Verification was
targeted, not exhaustive: clips were ranked by disagreement between two
independent transcripts, and the 122 riskiest were reviewed word by word.
About 4 annotator-hours instead of 20, with the residual error rate measured
rather than assumed.

**Manifests are pointers, not copies.** A training row names a parquet file
and a row index in the upstream dataset; audio is decoded when the batch is
built. The corpus definition stays a 200 MB reviewable artifact instead of a
terabyte of duplicated audio, and every run records the config, the git commit
and the manifest hash, so a checkpoint traces back to the exact rows it saw.

**Punctuation cost nothing.** Instead of a separate restoration model, `. , ? !`
were added to the CTC vocabulary — four extra rows in the head's output
convolution, zero weight and bias −8 so they start silent, warm-started from
the base fine-tune. Every punctuation run reported WER with and without
symbols; the without-symbols number is the regression guard, and it improved
(8.26% → 8.03%) while the model learned to punctuate from prosody alone.

## Quickstart

```bash
make setup                  # uv sync
make check                  # ruff + mypy --strict + pytest

uv run uztts-text normalize "Мен 25 ёшдаман"     # men yigirma besh yoshdaman
uv run uztts-asr eval --model gigaam-large       # WER/CER on FLEURS
UZTTS_DATA_ROOT=~/uztts-data uv run uztts-serve  # local STT demo on :7860
```

Transcribing with the shipped model needs no checkout — `inference.py` in the
model repository is standalone:

```bash
wget https://huggingface.co/rustam1221/uzbek-asr-gigaam/raw/main/inference.py
python inference.py interview.mp3
```

## Pipeline

Each stage is a separate idempotent CLI that reads a manifest and writes a
new one; rerunning never redoes finished work.

```
ingest → segment → transcribe → filter → (train)
```

`uztts-ingest` pulls channel-level YouTube audio with per-channel hour caps,
`uztts-segment` cuts 2–20 s speech spans with silero-VAD and reports how many
hours survived, `uztts-transcribe` writes text plus the diagnostics the filter
stage needs (`asr_avg_logprob`, `asr_compression_ratio`, `lang_prob`).

## Conventions

Python 3.11 with `uv`. `ruff` + `mypy --strict` + `pytest`, all behind
`make check`. Code and identifiers in English; design documents in Uzbek under
`docs/`. Decisions are recorded as numbered files in `docs/decisions/` —
context, decision, alternatives rejected, consequences — and evaluation runs
are logged in `docs/eval/` with numbers, not impressions. `CLAUDE.md` is the
single source of truth for the project's direction.

## License

Apache-2.0. Model weights are MIT, inherited from GigaAM-Multilingual.
Training data is a mixture of upstream licenses — see the manifest repository
before commercial use. Raw scraped audio is never redistributed.
