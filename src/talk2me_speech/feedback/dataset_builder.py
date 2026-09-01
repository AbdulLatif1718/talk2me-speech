"""Utilities for building feedback datasets."""

from __future__ import annotations


def build_feedback_dataset(records: list[dict[str, str]]) -> list[dict[str, str]]:
    """Return a normalized list of feedback examples."""
    return [
        {
            "original": item.get("original", ""),
            "corrected": item.get("corrected", ""),
        }
        for item in records
    ]
