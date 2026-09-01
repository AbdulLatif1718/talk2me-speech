"""Audio normalization utilities."""

from __future__ import annotations

import numpy as np


def normalize_audio(audio: np.ndarray, target_dbfs: float = -20.0) -> np.ndarray:
    """Normalize audio to a target dBFS level."""
    if audio.size == 0:
        return audio

    peak = np.max(np.abs(audio))
    if peak == 0:
        return audio

    target_peak = 10 ** (target_dbfs / 20.0)
    return audio * (target_peak / peak)
