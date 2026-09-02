#!/usr/bin/env python3
"""Prepare a dataset manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from talk2me_speech.datasets.adapters import get_adapter
from talk2me_speech.datasets.loader import load_manifest
from talk2me_speech.datasets.manifest import write_manifest
from talk2me_speech.datasets.splitter import split_speaker_safe, verify_speaker_isolation
from talk2me_speech.datasets.validator import validate_manifest_records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare and canonicalize dataset manifests from raw audio and metadata",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/prepare_dataset.py --source commonvoice --input-dir data/raw/cv-corpus-15.0 --output data/manifests/cv_ghana.jsonl
  python scripts/prepare_dataset.py --source kasaspeech --input-dir data/raw/kasaspeech --split-speaker-safe
  python scripts/prepare_dataset.py --output data/manifests/example.jsonl
        """,
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        help="Path to raw dataset directory containing audio and metadata files",
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
        help="Dataset source name ('commonvoice', 'kasaspeech', or 'example')",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Language code (e.g., 'en', 'tw', 'ha'). Leave empty for multilingual.",
    )
    parser.add_argument(
        "--split-speaker-safe",
        action="store_true",
        help="Partition the dataset into speaker-disjoint train/val/test split manifests",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Train split ratio for speaker-safe splitting (default: 0.70)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation split ratio for speaker-safe splitting (default: 0.15)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Test split ratio for speaker-safe splitting (default: 0.15)",
    )

    args = parser.parse_args()

    records = []

    if args.input_dir and Path(args.input_dir).exists() and args.source.lower() not in ("example", "default"):
        try:
            adapter = get_adapter(args.source)
            records = adapter.convert(
                input_dir=args.input_dir,
                language=args.language,
            )
            print(f"✓ Imported {len(records)} records using {adapter.__class__.__name__}")
        except Exception as err:
            print(f"⚠️ Adapter import error ({err}). Falling back to sample records.")

    if not records:
        # Canonical sample records for demonstration / testing
        records = [
            {
                "id": "sample_001",
                "audio_path": "data/raw/audio/sample1.wav",
                "transcript": "Chale yɛbɛ deploy no tomorrow",
                "duration": 2.5,
                "sample_rate": 16000,
                "source": args.source,
                "speaker_id": "spk_gh_001",
                "primary_language": args.language or "en",
                "languages": [args.language] if args.language else ["en", "tw"],
                "country": "GH",
                "verified": True,
            },
            {
                "id": "sample_002",
                "audio_path": "data/raw/audio/sample2.wav",
                "transcript": "Me pɛ sɛ me kɔ fie",
                "duration": 3.1,
                "sample_rate": 16000,
                "source": args.source,
                "speaker_id": "spk_gh_002",
                "primary_language": "tw",
                "languages": ["tw"],
                "country": "GH",
                "verified": True,
            },
            {
                "id": "sample_003",
                "audio_path": "data/raw/audio/sample3.wav",
                "transcript": "We are going to make it work chale",
                "duration": 2.8,
                "sample_rate": 16000,
                "source": args.source,
                "speaker_id": "spk_gh_003",
                "primary_language": "en",
                "languages": ["en"],
                "country": "GH",
                "verified": True,
            },
        ]

    # Validate schema
    valid, errors = validate_manifest_records(records)
    if not valid:
        print("⚠️ Schema validation warnings:")
        for err in errors[:5]:
            print(f"  - {err}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.split_speaker_safe:
        train_recs, val_recs, test_recs = split_speaker_safe(
            records,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
        )
        isolated = verify_speaker_isolation(train_recs, val_recs, test_recs)

        stem = output_path.stem
        parent = output_path.parent
        suffix = output_path.suffix or ".jsonl"

        train_path = parent / f"{stem}_train{suffix}"
        val_path = parent / f"{stem}_val{suffix}"
        test_path = parent / f"{stem}_test{suffix}"

        write_manifest(train_path, train_recs)
        write_manifest(val_path, val_recs)
        write_manifest(test_path, test_recs)

        print("=" * 60)
        print(" Speaker-Safe Dataset Partitioning Completed")
        print("=" * 60)
        print(f"  Train split:      {len(train_recs)} records -> {train_path}")
        print(f"  Validation split: {len(val_recs)} records -> {val_path}")
        print(f"  Test split:       {len(test_recs)} records -> {test_path}")
        print(f"  Speaker isolation: {'✓ Verified (0% speaker overlap)' if isolated else '✗ Overlap detected'}")
        print("=" * 60)
    else:
        write_manifest(output_path, records)
        loaded = load_manifest(output_path)
        print("=" * 60)
        print(f"✓ Prepared {len(loaded)} dataset records")
        print(f"  Output: {output_path}")
        print(f"  Source: {args.source}")
        if args.language:
            print(f"  Language: {args.language}")
        print("=" * 60)


if __name__ == "__main__":
    main()
