#!/usr/bin/env python3
"""Validate dataset path structure."""

from __future__ import annotations

import argparse
from pathlib import Path

from talk2me_speech.datasets.validator import validate_dataset_root


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate dataset directory structure and contents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/validate_dataset.py --root data/
  python scripts/validate_dataset.py --root data/ --verbose
        """,
    )

    parser.add_argument(
        "--root",
        type=str,
        default="./data",
        help="Dataset root directory path",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed validation results",
    )

    args = parser.parse_args()

    dataset_root = Path(args.root)
    
    if not dataset_root.exists():
        print(f"❌ Dataset root not found: {dataset_root}")
        return

    status = validate_dataset_root(dataset_root)
    
    if args.verbose:
        print(f"Validating: {dataset_root}")
        print(f"Status: {'✓ Valid' if status else '✗ Invalid'}")
        
        # Check for expected subdirectories
        expected_dirs = ["train", "validation", "test"]
        for subdir in expected_dirs:
            subdir_path = dataset_root / subdir
            exists = "✓" if subdir_path.exists() else "✗"
            print(f"  {exists} {subdir}/")
    else:
        result = "✓ Valid" if status else "✗ Invalid"
        print(f"Dataset validation: {result}")


if __name__ == "__main__":
    main()
