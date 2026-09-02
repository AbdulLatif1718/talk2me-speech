#!/usr/bin/env python3
"""Model export helper."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export trained model checkpoint for production inference or deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/export_model.py --checkpoint models/checkpoints/W1_whisper --output-dir models/exported/talk2me-whisper-gh-v0.1
  python scripts/export_model.py --model whisper-small --format huggingface --quantize
  python scripts/export_model.py --help
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="whisper-small",
        help="Base model architecture",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to training checkpoint directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models/exported",
        help="Target directory to export model artifacts",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="huggingface",
        choices=["huggingface", "onnx", "torchscript"],
        help="Export format",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Apply 8-bit / dynamic quantization for edge or low-latency deployment",
    )

    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(" Talk2Me Model Exporter")
    print("=" * 60)
    print(f"  Model:        {args.model}")
    print(f"  Checkpoint:   {args.checkpoint or 'Base pretrained'}")
    print(f"  Target Dir:   {output_path}")
    print(f"  Format:       {args.format}")
    print(f"  Quantization: {'Enabled (int8)' if args.quantize else 'Disabled (fp16/fp32)'}")
    print("-" * 60)

    print(f"✓ Model export prepared at: {output_path}")
    print("  Note: Deep model serialization hooks will be activated in Phase 6")


if __name__ == "__main__":
    main()
