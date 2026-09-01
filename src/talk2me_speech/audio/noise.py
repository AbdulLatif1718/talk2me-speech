"""Noise reduction helpers."""

from __future__ import annotations

import numpy as np


def reduce_noise(audio: np.ndarray, level: float = 0.05) -> np.ndarray:
    """Apply a lightweight spectral floor to suppress very small noise components."""
    signal = np.asarray(audio, dtype=np.float32)
    if signal.size == 0:
        return signal
    floor = np.max(np.abs(signal)) * level
    clipped = np.clip(signal, -1.0, 1.0)
    return np.where(np.abs(clipped) < floor, 0.0, clipped)
