"""Manifest creation utilities."""

from __future__ import annotations

import json
from pathlib import Path


def write_manifest(path: str | Path, records: list[dict]) -> str:
    """Write a list of records to a JSONL manifest file."""
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return str(manifest_path)
