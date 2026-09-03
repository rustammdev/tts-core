---
language:
- uz
license: other
license_name: derived-annotations-only
task_categories:
- automatic-speech-recognition
tags:
- uzbek
- asr
- benchmark
- spontaneous-speech
- human-verified
pretty_name: Uzbek Spontaneous Speech ASR Benchmark
size_categories:
- n<1K
---

# Uzbek Spontaneous Speech ASR Benchmark

A 311-clip, 1.98-hour test set for Uzbek speech recognition, built from
spontaneous YouTube speech: podcasts, interviews and multi-speaker
conversation with overlapping turns, fillers, code-switching into Russian, and
dialect spelling. Every transcript that an automatic difficulty check flagged
as possibly wrong was corrected by hand — 118 of the 311 — and the protocol
below says exactly which ones and why.

It exists because the public Uzbek ASR benchmarks do not measure the thing we
ship. FLEURS is read speech, and models score two to three times better on it
than on real conversation. On this benchmark the same GigaAM-large baseline
that scores 6.7% WER on FLEURS scores 16.1%. Anything tuned against FLEURS
alone is tuned against the easy case.

| | |
|---|---|
| Clips | 311 |
| Audio | 1.98 h |
| Reference words | 12,385 |
| Clip length | 3.2–30.0 s (median 28.5 s) |
| Human-verified clips | 118 of 311 (see protocol) |
| Domain | spontaneous YouTube speech (podcast, interview, vlog) |
| Script | Latin Uzbek, punctuated |
| Audio included | no — reconstructed from the upstream source (see below) |

## Why the transcripts can be trusted

The source corpus (`Abduqayum/Uzbek-STT-Dataset-780h`) ships Gemini-generated
transcripts. Machine transcripts cannot be used to score machine transcripts,
so every clip went through a verification pass:

1. **Sampling.** 315 clips drawn with a fixed seed. Clips containing music,
   Cyrillic script, or digits were excluded before sampling — those are text
   normalization problems, not recognition problems, and they would have made
   the numbers about the wrong thing.
2. **Difficulty scoring.** Each clip got an `asr_wer` field: the disagreement
   between the Gemini transcript and GigaAM-large. High disagreement means at
   least one of the two is wrong.
3. **Targeted human review.** A 7-clip random spot check of the low-disagreement
   band came back 100% correct, so manual review was spent where it pays:
   all 122 clips with `asr_wer >= 0.18` were reviewed word by word in a
   purpose-built annotation UI. 118 were corrected or confirmed, 4 were
   discarded as unusable. The remaining 193 clips were auto-accepted.
4. **Result.** 311 clips: 118 human-verified, 193 auto-accepted behind a
   measured error rate.

This is a deliberate trade. Full manual transcription of 2 hours costs roughly
20 annotator-hours; this protocol spent about 4 and put them where the
uncertainty was.

## Files

| File | What it is |
|---|---|
| `final_etalon.jsonl` | **the benchmark** — 311 rows, `id`, `source_row`, `duration`, `text`, `reviewed`, `asr_wer` |
| `etalon.jsonl` | all 315 sampled rows before discards, with the original Gemini text |
| `corrections.jsonl` | the 122 human review decisions (`ok` / `discard` + corrected text) |
| `source_map.json` | `id` → upstream parquet file + row, for rebuilding the audio |
| `exclude_rows.json` | the same rows keyed by parquet file, to remove them from training data |
| `predictions/*.jsonl` | reference/hypothesis pairs for every model in the table below |
| `scoring.py` | normalization + WER/CER, standalone |
| `reconstruct_benchmark_audio.py` | rebuilds the audio from upstream |

`etalon.jsonl` and `corrections.jsonl` are kept so the verification pass is
auditable: anyone can diff the machine transcript against the human decision.

## Getting the audio

Audio is not redistributed here. Every row carries its upstream location, and
the helper script fetches only the parquet row groups that hold benchmark
clips rather than the full 780-hour corpus:

```bash
pip install pyarrow huggingface_hub
python reconstruct_benchmark_audio.py ./benchmark_audio
```

Output is `<id>.wav`, mono 16 kHz, matching `final_etalon.jsonl` by `id`.

## Results

Corpus-level WER/CER with `jiwer`. Text on both sides is lowercased, mapped to
canonical Uzbek apostrophes (ʻ/ʼ) and stripped of punctuation before scoring.

| Model | Params | WER | CER |
|---|---|---|---|
| [`large_full_600m`](https://huggingface.co/rustam1221/uzbek-asr-gigaam) | 600M | **12.2%** | **3.4%** |
| [`gemini_full_220m`](https://huggingface.co/rustam1221/uzbek-asr-gigaam) | 220M | 13.9% | 3.8% |
| `gemini_bench_220m` | 220M | 14.0% | 3.8% |
| `punct_220m` | 220M | 15.0% | 4.1% |
| `base_220m` | 220M | 15.5% | 4.5% |
| GigaAM-large — untuned | 600M | 16.1% | 5.8% |
| GigaAM — untuned | 220M | 17.1% | 5.4% |
| whisper-large-v3-turbo-uzbek | 809M | 39.6% | 19.4% |

For reference, the same untuned GigaAM-large scores 6.7% WER on FLEURS. The
gap between 6.7% and 16.1% is the entire reason this benchmark exists.

Under canonical scoring the two shipped checkpoints are 11.7% and 13.3%.
Their per-clip predictions are in `predictions/`, so any row of this table can
be recomputed or disputed.

**Canonical scoring.** Uzbek clitics (`da`, `ku`, `chi`, `mi`, `yu`, `ya`,
`ta`) are written both joined and separated in real text, and neither is
wrong. Scoring them as errors measures orthographic convention, not
recognition. The canonical variant joins clitics on both sides before scoring;
it is worth about 0.6 WER points and is the number to compare against.

## Reproducing a score

`scoring.py` in this repository implements the normalization and both WER
variants, with no other dependency than `jiwer`:

```python
import json
from huggingface_hub import hf_hub_download
from scoring import score

rows = [
    json.loads(line)
    for line in open(
        hf_hub_download(
            "rustam1221/uzbek-asr-benchmark-spontaneous",
            "final_etalon.jsonl",
            repo_type="dataset",
        )
    )
]

references = [row["text"] for row in rows]
hypotheses = [your_model.transcribe(f"benchmark_audio/{row['id']}.wav") for row in rows]
print(score(references, hypotheses))
# {'wer': ..., 'cer': ..., 'wer_canonical': ..., 'cer_canonical': ...}
```

Already have predictions on disk? `python scoring.py predictions/large_full_600m.jsonl`
prints all four numbers for any JSONL with `ref` and `hyp` fields.

## Known limits

- **Long clips.** Median length is 28.5 s, near the source corpus cap. Errors
  concentrate in the longest multi-speaker clips, so a model that segments
  well before recognizing has an advantage this benchmark does not isolate.
- **Auto-accepted tail.** The 193 auto-accepted clips inherit any Gemini error
  that GigaAM-large happened to agree with. The spot check bounds this, it
  does not eliminate it.
- **Single source corpus.** All clips come from one upstream YouTube
  collection, so channel and topic diversity is whatever that collection has.
- **Size.** 1.98 h and 12.4k words: differences under roughly half a WER point
  are not meaningful.

## Licensing

The annotations in this repository (corrections, review decisions, row
mapping) are released for research and evaluation use. The underlying audio
belongs to its original YouTube publishers and is **not** redistributed here —
it is fetched from the upstream dataset, whose terms apply to that audio.

## Citation

```bibtex
@misc{uzbek_spontaneous_asr_benchmark,
  title  = {Uzbek Spontaneous Speech ASR Benchmark},
  author = {Nuriddinov, Rustamjon},
  year   = {2026},
  url    = {https://huggingface.co/datasets/rustam1221/uzbek-asr-benchmark-spontaneous}
}
```
