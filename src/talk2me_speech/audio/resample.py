"""Audio resampling utilities."""

from __future__ import annotations

import numpy as np


def resample_audio(audio: np.ndarray, original_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio from one sample rate to another.

    This lightweight implementation keeps the signal as a NumPy array and uses a
    simple linear interpolation strategy for compatibility in tests.
    """
    if original_sr == target_sr or audio.size == 0:
        return audio

    if audio.ndim > 1:
        audio = audio.mean(axis=-1)

    old_t = np.arange(len(audio), dtype=np.float64) / original_sr
    new_t = np.arange(0, len(audio) / original_sr, 1 / target_sr, dtype=np.float64)
    if len(new_t) == 0:
        return audio[:0]

    old_idx = np.clip(new_t * original_sr, 0, len(audio) - 1)
    return np.interp(new_t, old_t, audio.astype(np.float64), left=audio[0], right=audio[-1])
