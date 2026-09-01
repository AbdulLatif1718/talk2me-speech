#!/usr/bin/env python3
"""Model export helper."""

from __future__ import annotations

from pathlib import Path


def main() -> None:
    output_dir = Path("./models/exported")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Prepared export directory: {output_dir}")


if __name__ == "__main__":
    main()
