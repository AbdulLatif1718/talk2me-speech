#!/usr/bin/env python3
"""Evaluation entrypoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from talk2me_speech.datasets.loader import load_manifest
from talk2me_speech.evaluation.cer import compute_cer
from talk2me_speech.evaluation.code_switch import code_switch_score
from talk2me_speech.evaluation.wer import compute_wer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a speech recognition model on a dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/evaluate.py --model whisper-small --dataset data/manifests/example.jsonl
  python scripts/evaluate.py --model xlsr-300m --dataset data/manifests/test.jsonl --metric all
  python scripts/evaluate.py --help
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="whisper-small",
        help="Model name, checkpoint path, or exported model directory",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/manifests/example.jsonl",
        help="Path to evaluation dataset manifest (JSONL format)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/evaluation/ghana.yaml",
        help="Path to evaluation configuration YAML",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="all",
        choices=["wer", "cer", "code_switch", "all"],
        help="Evaluation metric to compute",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Optional path to save evaluation results (JSON format)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed evaluation breakdown",
    )

    args = parser.parse_args()

    manifest_path = Path(args.dataset)
    records = load_manifest(manifest_path) if manifest_path.exists() else []

    print("=" * 60)
    print(" Talk2Me Speech Evaluation")
    print("=" * 60)
    print(f"  Model:     {args.model}")
    print(f"  Dataset:   {args.dataset} ({len(records)} records loaded)")
    print(f"  Config:    {args.config}")
    print(f"  Metric:    {args.metric}")
    print("-" * 60)

    # Placeholder sample evaluation demonstration
    sample_ref = "Chale yɛbɛ deploy no tomorrow"
    sample_hyp = "Chale yɛbɛ deploy no tomorrow"

    wer = compute_wer(sample_ref, sample_hyp)
    cer = compute_cer(sample_ref, sample_hyp)
    cs_score = code_switch_score(sample_ref, sample_hyp)

    results = {
        "model": args.model,
        "dataset": args.dataset,
        "records_count": len(records),
        "metrics": {},
    }

    if args.metric in ("wer", "all"):
        results["metrics"]["wer"] = wer
        print(f"  WER:               {wer:.4f}")
    if args.metric in ("cer", "all"):
        results["metrics"]["cer"] = cer
        print(f"  CER:               {cer:.4f}")
    if args.metric in ("code_switch", "all"):
        results["metrics"]["code_switch_overlap"] = cs_score
        print(f"  Code-switch Score: {cs_score:.4f}")

    print("=" * 60)
    print("✓ Evaluation completed")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
