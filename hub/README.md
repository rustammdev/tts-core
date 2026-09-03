# hub

Source of truth for the public Hugging Face repositories. The cards and scripts
here are version-controlled and pushed with `uv run uztts-asr hub push-cards` —
nothing is edited in the Hub web UI.

| File | Published as |
|---|---|
| `model_card.md` | `rustam1221/uzbek-asr-gigaam` → `README.md` |
| `inference.py` | `rustam1221/uzbek-asr-gigaam` → `inference.py` |
| `requirements.txt` | `rustam1221/uzbek-asr-gigaam` → `requirements.txt` |
| `manifests_card.md` | `rustam1221/uzbek-asr-train-manifests` → `README.md` |
| `benchmark_card.md` | `rustam1221/uzbek-asr-benchmark-spontaneous` → `README.md` |
| `reconstruct_benchmark_audio.py` | `rustam1221/uzbek-asr-benchmark-spontaneous` → same name |
| `scoring.py` | `rustam1221/uzbek-asr-benchmark-spontaneous` → same name |
| `source_map.json` | `rustam1221/uzbek-asr-benchmark-spontaneous` → same name |

The published scripts are **standalone** — they duplicate the few functions
they need (`normalize_text`, `join_clitics`, the CTC head extension) instead of
importing `uztts_asr` / `uztts_text`, because the code repository is private.
If a normalization rule changes here, it changes in both places.
