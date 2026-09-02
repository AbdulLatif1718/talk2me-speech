#!/usr/bin/env python3
"""Prepare a dataset manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from talk2me_speech.datasets.loader import load_manifest
from talk2me_speech.datasets.manifest import write_manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a dataset manifest from audio files and transcripts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/prepare_dataset.py --output data/manifests/test.jsonl
  python scripts/prepare_dataset.py --source commonvoice --language en
        """,
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/manifests/example.jsonl",
        help="Output manifest file path (JSONL format)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="example",
        help="Dataset source name",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Language code (e.g., 'en', 'tw', 'ha'). Leave empty for multilingual.",
    )

    args = parser.parse_args()

    # Create sample records (placeholder - actual data loading implemented in Phase 2)
    records = [
        {
            "id": "sample_001",
            "audio_path": "audio/sample1.wav",
            "transcript": "Chale yɛbɛ deploy no tomorrow",
            "duration": 2.5,
            "source": args.source,
            "languages": [args.language] if args.language else None,
        }
    ]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    write_manifest(output_path, records)
    loaded = load_manifest(output_path)

    print(f"✓ Prepared {len(loaded)} dataset records")
    print(f"  Output: {output_path}")
    print(f"  Source: {args.source}")
    if args.language:
        print(f"  Language: {args.language}")


if __name__ == "__main__":
    main()
