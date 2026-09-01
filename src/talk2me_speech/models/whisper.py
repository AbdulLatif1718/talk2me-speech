"""Whisper model wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class WhisperModel:
    """Simple wrapper around a Whisper-style model configuration."""

    model_name: str = "whisper-small"
    cache_dir: str | None = None

    def load(self) -> dict[str, Any]:
        """Return a configuration dictionary for the model loader."""
        return {
            "model_name": self.model_name,
            "cache_dir": self.cache_dir,
            "architecture": "whisper",
        }
