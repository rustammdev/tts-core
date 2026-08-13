from __future__ import annotations

from typing import TYPE_CHECKING, Any

from uztts_events.labels import resolve_class_ids
from uztts_events.schema import EventLabel

if TYPE_CHECKING:
    import numpy as np

SAMPLE_RATE = 16_000
CHUNK_SECONDS = 10.0
MIN_CHUNK_SECONDS = 0.25


class CedScreener:
    def __init__(
        self,
        model_name: str = "mispeech/ced-small",
        revision: str = "",
        device: str = "cpu",
    ) -> None:
        import torch
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

        kwargs: dict[str, Any] = {"trust_remote_code": True}
        if revision:
            kwargs["revision"] = revision
        self._extractor = AutoFeatureExtractor.from_pretrained(  # type: ignore[no-untyped-call]
            model_name, **kwargs
        )
        model = AutoModelForAudioClassification.from_pretrained(model_name, **kwargs)
        self._model = model.to(device).eval()
        self._device = device
        self._torch = torch
        id2label = {int(key): value for key, value in model.config.id2label.items()}
        class_names = [id2label[index] for index in sorted(id2label)]
        self._class_ids = resolve_class_ids(class_names)

    def screen(self, waveform: np.ndarray) -> dict[EventLabel, float]:
        torch = self._torch
        scores = dict.fromkeys(self._class_ids, 0.0)
        chunk_samples = int(CHUNK_SECONDS * SAMPLE_RATE)
        min_samples = int(MIN_CHUNK_SECONDS * SAMPLE_RATE)
        for offset in range(0, max(len(waveform), 1), chunk_samples):
            chunk = waveform[offset : offset + chunk_samples]
            if len(chunk) < min_samples and offset > 0:
                break
            audio = torch.from_numpy(chunk).float()[None, :]
            inputs = self._extractor(
                audio, sampling_rate=SAMPLE_RATE, return_tensors="pt"
            )
            features = inputs["input_values"].to(self._device)
            with torch.no_grad():
                probs = self._model(input_values=features).logits[0].cpu()
            for label, class_ids in self._class_ids.items():
                value = float(probs[class_ids].max())
                scores[label] = max(scores[label], value)
        return scores
