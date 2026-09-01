"""User correction processing."""

from __future__ import annotations


def correct_transcript(transcript: str, corrections: dict[str, str]) -> str:
    """Apply a dictionary of token replacements to a transcript."""
    corrected = transcript
    for original, replacement in corrections.items():
        corrected = corrected.replace(original, replacement)
    return corrected
