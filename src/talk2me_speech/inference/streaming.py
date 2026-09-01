"""Streaming inference helpers."""

from __future__ import annotations

from typing import Any


def stream_transcript(audio_chunks: list[Any]) -> list[str]:
    """Convert an audio chunk list into a transcript placeholder list."""
    return [str(chunk) for chunk in audio_chunks]
