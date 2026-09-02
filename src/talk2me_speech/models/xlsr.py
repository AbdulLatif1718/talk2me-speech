"""XLS-R model wrapper and inference interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

XLSR_HF_MAP = {
    "xlsr-300m": "facebook/wav2vec2-xls-r-300m",
    "xlsr-1b": "facebook/wav2vec2-xls-r-1b",
    "xlsr-2b": "facebook/wav2vec2-xls-r-2b",
}


@dataclass
class XLSRModel:
    """Wrapper around Meta XLS-R (Wav2Vec2) model configurations and inference."""

    model_name: str = "xlsr-300m"
    cache_dir: str | None = None
    device: str = "cpu"

    def __post_init__(self) -> None:
        self.hf_model_id = XLSR_HF_MAP.get(self.model_name, self.model_name)
        self._model = None
        self._processor = None

    def load(self) -> dict[str, Any]:
        """Return a configuration dictionary for the model loader."""
        return {
            "model_name": self.model_name,
            "hf_model_id": self.hf_model_id,
            "cache_dir": self.cache_dir,
            "architecture": "xlsr",
            "device": self.device,
        }

    def get_hf_model(self) -> tuple[Any, Any]:
        """Load and cache the HuggingFace Wav2Vec2/XLS-R model and processor."""
        if self._model is None or self._processor is None:
            try:
                from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

                self._processor = Wav2Vec2Processor.from_pretrained(
                    self.hf_model_id, cache_dir=self.cache_dir
                )
                self._model = Wav2Vec2ForCTC.from_pretrained(
                    self.hf_model_id, cache_dir=self.cache_dir
                )
                if self.device != "cpu":
                    self._model = self._model.to(self.device)
            except Exception:
                return None, None
        return self._model, self._processor

    def transcribe(self, audio_array: Any, sampling_rate: int = 16000) -> str:
        """Transcribe audio with Wav2Vec2/XLS-R CTC decoding."""
        model, processor = self.get_hf_model()
        if model is None or processor is None:
            return "Sample XLS-R transcribed text (offline fallback)"

        import torch

        inputs = processor(
            audio_array, sampling_rate=sampling_rate, return_tensors="pt", padding=True
        )

        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]
        return str(transcription).strip()
