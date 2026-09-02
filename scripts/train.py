#!/usr/bin/env python3
"""Training entrypoint."""

from __future__ import annotations

import argparse

from talk2me_speech.training.train_whisper import build_train_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a speech recognition model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported models:
  whisper-small (recommended for initial experiments)
  whisper-medium
  xlsr-300m
  xlsr-1b

Examples:
  python scripts/train.py --model whisper-small --epochs 5
  python scripts/train.py --model xlsr-300m --batch-size 16
  python scripts/train.py --help
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="whisper-small",
        choices=["whisper-small", "whisper-medium", "whisper-large", "xlsr-300m", "xlsr-1b"],
        help="Model to train",
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
        "--output-dir",
        type=str,
        default="./models/checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML config file (overrides other arguments if provided)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print config without training",
    )

    args = parser.parse_args()

    # Build config
    config = build_train_config(output_dir=args.output_dir)
    
    # Override with command-line arguments
    config["model"] = args.model
    config["epochs"] = args.epochs
    config["batch_size"] = args.batch_size
    config["learning_rate"] = args.learning_rate

    if args.dry_run:
        print("Training configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        print("\n✓ Dry-run mode (no training executed)")
        return

    print("Training config:", config)
    print("\n⚠️  Note: Actual training implementation is planned for Phase 3")
    print("See README.md for experiment roadmap and current status")


if __name__ == "__main__":
    main()
