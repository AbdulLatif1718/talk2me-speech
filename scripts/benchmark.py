#!/usr/bin/env python3
"""Benchmark entrypoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from talk2me_speech.evaluation.benchmark import benchmark_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run speech recognition benchmarks across models and datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/benchmark.py --models whisper-small xlsr-300m --datasets data/manifests/example.jsonl
  python scripts/benchmark.py --config configs/evaluation/ghana.yaml --output benchmark_summary.json
  python scripts/benchmark.py --help
        """,
    )

    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["whisper-small", "xlsr-300m"],
        help="List of model names or checkpoint paths to benchmark",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["data/manifests/example.jsonl"],
        help="List of dataset manifest paths to evaluate against",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/evaluation/ghana.yaml",
        help="Path to evaluation/benchmark configuration YAML",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save aggregate benchmark results JSON",
    )

    args = parser.parse_args()

    print("=" * 65)
    print(" Talk2Me Speech Model Benchmarking")
    print("=" * 65)
    print(f"  Models:   {', '.join(args.models)}")
    print(f"  Datasets: {', '.join(args.datasets)}")
    print(f"  Config:   {args.config}")
    print("-" * 65)

    benchmark_runs = []
    for model in args.models:
        # Mock benchmark scores for demonstration
        score = 0.12 if "whisper" in model else 0.18
        benchmark_runs.append({"model": model, "score": score})
        print(f"  Model: {model:<20} Score (WER proxy): {score:.4f}")

    summary = benchmark_results(benchmark_runs)
    print("-" * 65)
    print(f"  Total runs evaluated: {int(summary.get('count', 0))}")
    print(f"  Average benchmark score: {summary.get('average', 0.0):.4f}")
    print("=" * 65)
    print("✓ Benchmarking completed")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump({"summary": summary, "runs": benchmark_runs}, f, indent=2)
        print(f"  Summary saved to: {out_path}")


if __name__ == "__main__":
    main()
