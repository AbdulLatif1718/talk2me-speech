"""Training data collation helpers."""

from __future__ import annotations

from typing import Any


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Convert a batch of record dicts into a simple dictionary structure."""
    return {
        "items": list(batch),
        "batch_size": len(batch),
    }
