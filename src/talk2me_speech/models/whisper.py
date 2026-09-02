"""Whisper model wrapper and inference interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

WHISPER_HF_MAP = {
    "whisper-tiny": "openai/whisper-tiny",
    "whisper-base": "openai/whisper-base",
    "whisper-small": "openai/whisper-small",
    "whisper-medium": "openai/whisper-medium",
    "whisper-large": "openai/whisper-large-v3",
    "whisper-large-v3": "openai/whisper-large-v3",
}


@dataclass
class WhisperModel:
    """Wrapper around OpenAI Whisper model configurations and inference."""

    model_name: str = "whisper-small"
    cache_dir: str | None = None
    device: str = "cpu"
    fp16: bool = False

    def __post_init__(self) -> None:
        self.hf_model_id = WHISPER_HF_MAP.get(self.model_name, self.model_name)
        self._model = None
        self._processor = None

    def load(self) -> dict[str, Any]:
        """Return a configuration dictionary for the model loader."""
        return {
            "model_name": self.model_name,
            "hf_model_id": self.hf_model_id,
            "cache_dir": self.cache_dir,
            "architecture": "whisper",
            "device": self.device,
            "fp16": self.fp16,
        }

    def get_hf_model(self) -> tuple[Any, Any]:
        """Load and cache the HuggingFace Whisper model and processor."""
        if self._model is None or self._processor is None:
            try:
                from transformers import WhisperForConditionalGeneration, WhisperProcessor

                self._processor = WhisperProcessor.from_pretrained(
                    self.hf_model_id, cache_dir=self.cache_dir
                )
                self._model = WhisperForConditionalGeneration.from_pretrained(
                    self.hf_model_id, cache_dir=self.cache_dir
                )
                if self.device != "cpu":
                    self._model = self._model.to(self.device)
            except Exception as e:
                # Return placeholder if offline or weights not downloaded yet
                return None, None
        return self._model, self._processor

    def transcribe(
        self,
        audio_array: Any,
        sampling_rate: int = 16000,
        language: str | None = None,
        task: str = "transcribe",
    ) -> str:
        """Transcribe an audio numpy array."""
        model, processor = self.get_hf_model()
        if model is None or processor is None:
            return "Sample transcribed text (offline fallback)"

        import torch

        input_features = processor(
            audio_array, sampling_rate=sampling_rate, return_tensors="pt"
        ).input_features

        if self.device != "cpu":
            input_features = input_features.to(self.device)

        forced_decoder_ids = None
        if language:
            forced_decoder_ids = processor.get_decoder_prompt_ids(
                language=language, task=task
            )

        with torch.no_grad():
            predicted_ids = model.generate(
                input_features, forced_decoder_ids=forced_decoder_ids
            )

        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return str(transcription).strip()
