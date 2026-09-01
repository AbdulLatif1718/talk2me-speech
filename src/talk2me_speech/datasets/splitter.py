"""Dataset splitting utilities."""

from __future__ import annotations

from typing import Iterable


def split_records(records: Iterable[dict], train_ratio: float = 0.8) -> tuple[list[dict], list[dict]]:
    """Split a list of records into training and validation sets."""
    items = list(records)
    split_index = max(1, int(len(items) * train_ratio))
    return items[:split_index], items[split_index:]
