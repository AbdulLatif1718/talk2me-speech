"""Benchmarking utilities for evaluation pipelines."""

from __future__ import annotations

from typing import Any


def benchmark_results(results: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate a list of benchmark result dictionaries into summary metrics."""
    if not results:
        return {"count": 0.0, "average": 0.0}

    values = [float(item.get("score", 0.0)) for item in results]
    return {
        "count": float(len(values)),
        "average": sum(values) / len(values),
    }
