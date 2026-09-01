"""Batch inference helpers."""

from __future__ import annotations

from typing import Iterable


def batch_transcribe(items: Iterable[str], batch_size: int = 8) -> list[list[str]]:
    """Chunk input items into batches."""
    values = list(items)
    return [values[i : i + batch_size] for i in range(0, len(values), batch_size)]
