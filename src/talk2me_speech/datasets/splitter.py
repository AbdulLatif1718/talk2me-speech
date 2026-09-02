"""Dataset splitting utilities with speaker-safe partition support."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Iterable


def split_records(
    records: Iterable[dict[str, Any]], train_ratio: float = 0.8
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split a list of records into training and validation sets sequentially."""
    items = list(records)
    if not items:
        return [], []
    split_index = max(1, int(len(items) * train_ratio)) if len(items) > 1 else 1
    return items[:split_index], items[split_index:]


def split_speaker_safe(
    records: Iterable[dict[str, Any]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Partition dataset records into train, validation, and test sets with strict speaker isolation.

    Ensures that no speaker present in the test/validation set appears in the training set,
    preventing data leakage and acoustic overfitting.
    """
    items = list(records)
    if not items:
        return [], [], []

    # Group records by speaker_id
    speaker_to_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for idx, item in enumerate(items):
        spk = item.get("speaker_id")
        spk_key = str(spk) if spk is not None else f"__unassigned_spk_{idx}__"
        speaker_to_records[spk_key].append(item)

    speakers = list(speaker_to_records.keys())
    rng = random.Random(seed)
    rng.shuffle(speakers)

    total_records = len(items)
    target_train_count = int(total_records * train_ratio)
    target_val_count = int(total_records * val_ratio)

    train_records: list[dict[str, Any]] = []
    val_records: list[dict[str, Any]] = []
    test_records: list[dict[str, Any]] = []

    for spk in speakers:
        spk_records = speaker_to_records[spk]
        if len(train_records) < target_train_count:
            train_records.extend(spk_records)
        elif len(val_records) < target_val_count:
            val_records.extend(spk_records)
        else:
            test_records.extend(spk_records)

    # If test is empty due to small number of speakers, ensure non-empty partitions if possible
    if not test_records and len(val_records) > 1 and len(speakers) >= 3:
        test_records.append(val_records.pop())

    return train_records, val_records, test_records


def verify_speaker_isolation(
    train: list[dict[str, Any]],
    val: list[dict[str, Any]],
    test: list[dict[str, Any]],
) -> bool:
    """Verify that there is zero speaker overlap across train, validation, and test splits."""
    def get_speakers(split: list[dict[str, Any]]) -> set[str]:
        return {
            str(item["speaker_id"])
            for item in split
            if item.get("speaker_id") is not None
        }

    train_spks = get_speakers(train)
    val_spks = get_speakers(val)
    test_spks = get_speakers(test)

    train_val_overlap = train_spks & val_spks
    train_test_overlap = train_spks & test_spks
    val_test_overlap = val_spks & test_spks

    return not (train_val_overlap or train_test_overlap or val_test_overlap)
