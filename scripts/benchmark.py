#!/usr/bin/env python3
"""Benchmark entrypoint."""

from __future__ import annotations

from talk2me_speech.evaluation.benchmark import benchmark_results


def main() -> None:
    metrics = benchmark_results([{"score": 0.12}, {"score": 0.15}])
    print("Benchmark summary:", metrics)


if __name__ == "__main__":
    main()
