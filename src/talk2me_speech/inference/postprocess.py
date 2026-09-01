"""Post-processing for transcripts."""

from __future__ import annotations


def postprocess_transcript(text: str) -> str:
    """Normalize whitespace and punctuation in a transcript."""
    return " ".join(text.strip().split())
