"""Model registry helper."""

from __future__ import annotations

from typing import Any


def get_model_registry() -> dict[str, dict[str, Any]]:
    """Return known model types and their default configuration."""
    return {
        "whisper-small": {"architecture": "whisper", "model_name": "whisper-small"},
        "whisper-medium": {"architecture": "whisper", "model_name": "whisper-medium"},
        "whisper-large": {"architecture": "whisper", "model_name": "whisper-large"},
        "xlsr-300m": {"architecture": "xlsr", "model_name": "xlsr-300m"},
        "xlsr-1b": {"architecture": "xlsr", "model_name": "xlsr-1b"},
    }
