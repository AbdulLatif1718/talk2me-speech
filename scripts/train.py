#!/usr/bin/env python3
"""Train a speech recognition model."""

from __future__ import annotations

import argparse
from pathlib import Path

from talk2me_speech.training.train_whisper import build_train_config as build_whisper_config
from talk2me_speech.training.train_whisper import train_whisper
from talk2me_speech.training.train_xlsr import build_train_config as build_xlsr_config
from talk2me_speech.training.train_xlsr import train_xlsr


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a speech recognition model (Whisper or XLS-R)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported models:
  whisper-small (recommended for initial experiments)
  whisper-medium
  whisper-large
  xlsr-300m
  xlsr-1b

Examples:
  python scripts/train.py --model whisper-small --epochs 5 --batch-size 8
  python scripts/train.py --model xlsr-300m --epochs 10 --batch-size 4
  python scripts/train.py --dry-run
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="whisper-small",
        choices=["whisper-small", "whisper-medium", "whisper-large", "xlsr-300m", "xlsr-1b"],
        help="Model architecture to train",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for training",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate for optimizer",
    )
    parser.add_argument(
        "--train-manifest",
        type=str,
        default="data/manifests/example_train.jsonl",
        help="Path to training dataset manifest",
    )
    parser.add_argument(
        "--eval-manifest",
        type=str,
        default="data/manifests/example_val.jsonl",
        help="Path to evaluation dataset manifest",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models/checkpoints",
        help="Directory to save checkpoints and summaries",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configuration and exit without training",
    )

    args = parser.parse_args()

    is_xlsr = "xlsr" in args.model

    if is_xlsr:
        config = build_xlsr_config(
            model_name=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            output_dir=args.output_dir,
            train_manifest=args.train_manifest,
            eval_manifest=args.eval_manifest,
        )
    else:
        config = build_whisper_config(
            model_name=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            output_dir=args.output_dir,
            train_manifest=args.train_manifest,
            eval_manifest=args.eval_manifest,
        )

    if args.dry_run:
        print("=" * 50)
        print(" Training Configuration (Dry Run)")
        print("=" * 50)
        for key, value in config.items():
            print(f"  {key:<18}: {value}")
        print("=" * 50)
        print("✓ Dry-run completed (no training executed)")
        return

    if is_xlsr:
        train_xlsr(config)
    else:
        train_whisper(config)


if __name__ == "__main__":
    main()
