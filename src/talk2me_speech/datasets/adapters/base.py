"""Base dataset adapter definition and canonical schema helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseDatasetAdapter(ABC):
    """Abstract base class for dataset adapters converting raw formats to canonical schema."""

    def __init__(self, source_name: str) -> None:
        self.source_name = source_name

    @abstractmethod
    def convert(
        self,
        input_dir: str | Path,
        output_manifest: str | Path | None = None,
        language: str | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Convert a raw dataset directory into a list of canonical Talk2Me records."""
        raise NotImplementedError

    def build_record(
        self,
        record_id: str,
        audio_path: str,
        transcript: str,
        duration: float = 0.0,
        sample_rate: int = 16000,
        speaker_id: str | None = None,
        primary_language: str | None = None,
        languages: list[str] | None = None,
        country: str | None = None,
        verified: bool = True,
        **metadata: Any,
    ) -> dict[str, Any]:
        """Construct a canonical Talk2Me record dictionary."""
        record: dict[str, Any] = {
            "id": str(record_id),
            "audio_path": str(audio_path),
            "transcript": str(transcript).strip(),
            "duration": float(duration),
            "sample_rate": int(sample_rate),
            "source": self.source_name,
            "speaker_id": str(speaker_id) if speaker_id is not None else None,
            "primary_language": str(primary_language) if primary_language is not None else None,
            "languages": languages or ([primary_language] if primary_language else None),
            "country": country,
            "verified": bool(verified),
        }
        if metadata:
            record["metadata"] = metadata
        return record

