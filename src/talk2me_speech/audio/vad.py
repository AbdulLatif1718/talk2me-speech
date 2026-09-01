"""Simple voice activity detection helper."""

from __future__ import annotations

import numpy as np


def voice_activity_mask(audio: np.ndarray, threshold: float = 0.02) -> np.ndarray:
    """Return a boolean mask where the signal is above a simple energy threshold."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1)
    energy = np.abs(audio)
    return energy > threshold
