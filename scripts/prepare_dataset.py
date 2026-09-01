#!/usr/bin/env python3
"""Prepare a dataset manifest."""

from __future__ import annotations

from pathlib import Path

from talk2me_speech.datasets.loader import load_manifest


def main() -> None:
    root = Path("./data")
    records = [{"audio": "sample.wav", "text": "hello world"}]
    manifest_path = root / "manifests" / "example.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(str(record).replace("'", '"') + "\n")

    loaded = load_manifest(manifest_path)
    print(f"Prepared {len(loaded)} dataset records at {manifest_path}")


if __name__ == "__main__":
    main()
