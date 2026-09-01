"""Feedback collection helpers."""

from __future__ import annotations


class FeedbackCollector:
    """Collect and store correction feedback for transcripts."""

    def __init__(self) -> None:
        self.records: list[dict[str, str]] = []

    def add(self, original: str, corrected: str) -> None:
        self.records.append({"original": original, "corrected": corrected})

    def export(self) -> list[dict[str, str]]:
        return list(self.records)
