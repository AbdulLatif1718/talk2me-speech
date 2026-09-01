#!/usr/bin/env python3
"""Validate dataset path structure."""

from __future__ import annotations

from pathlib import Path

from talk2me_speech.datasets.validator import validate_dataset_root


def main() -> None:
    dataset_root = Path("./data")
    status = validate_dataset_root(dataset_root)
    print(f"Dataset validation status: {status}")


if __name__ == "__main__":
    main()
