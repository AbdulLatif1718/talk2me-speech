"""Dataset validation helpers."""

from __future__ import annotations

from pathlib import Path


def validate_dataset_root(root: str | Path) -> bool:
    """Return True if a dataset root exists and looks valid."""
    dataset_root = Path(root)
    if not dataset_root.exists():
        return False
    required = ["train", "validation", "test"]
    return any((dataset_root / name).exists() for name in required)
