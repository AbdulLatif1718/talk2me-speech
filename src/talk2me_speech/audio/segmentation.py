"""Audio segmentation utilities."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def segment_audio(audio: np.ndarray, segment_length: int = 16000, hop_length: int = 8000) -> list[np.ndarray]:
    """Split audio into segments of fixed length."""
    signal = np.asarray(audio, dtype=np.float32)
    if signal.size == 0:
        return []

    segments: list[np.ndarray] = []
    for start in range(0, len(signal), hop_length):
        end = start + segment_length
        segment = signal[start:end]
        if len(segment) < segment_length:
            pad = np.zeros(segment_length - len(segment), dtype=np.float32)
            segment = np.concatenate([segment, pad])
        segments.append(segment)
    return segments
