"""KasaSpeech dataset adapter for Ghanaian speech and code-switching data."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from talk2me_speech.datasets.adapters.base import BaseDatasetAdapter
from talk2me_speech.datasets.manifest import write_manifest


class KasaSpeechAdapter(BaseDatasetAdapter):
    """Adapter for KasaSpeech datasets (Ghanaian English, Twi, and Code-Switching)."""

    def __init__(self) -> None:
        super().__init__(source_name="kasaspeech")

    def convert(
        self,
        input_dir: str | Path,
        output_manifest: str | Path | None = None,
        language: str | None = "twi",
        metadata_file: str = "metadata.csv",
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Parse KasaSpeech metadata formats (CSV, TSV, or JSONL) and audio clips."""
        in_path = Path(input_dir)
        meta_path = in_path / metadata_file if (in_path / metadata_file).exists() else in_path
        audio_dir = in_path / "audio" if (in_path / "audio").exists() else in_path

        records: list[dict[str, Any]] = []

        if not meta_path.exists() or meta_path.is_dir():
            return records

        # Handle CSV / TSV metadata
        if meta_path.suffix in (".csv", ".tsv"):
            delimiter = "\t" if meta_path.suffix == ".tsv" else ","
            with meta_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter=delimiter)
                for i, row in enumerate(reader):
                    audio_filename = row.get("audio_file") or row.get("file_name") or row.get("audio") or f"clip_{i}.wav"
                    transcript = row.get("transcript") or row.get("text") or row.get("sentence") or ""
                    if not transcript:
                        continue

                    speaker_id = row.get("speaker_id") or row.get("speaker") or row.get("speaker_name")
                    languages_str = row.get("languages") or row.get("language") or language or "twi"
                    languages = [lang.strip() for lang in languages_str.split(",")] if languages_str else ["twi"]
                    
                    record_id = row.get("id") or f"kasa_{Path(audio_filename).stem}"
                    audio_path = audio_dir / audio_filename

                    record = self.build_record(
                        record_id=record_id,
                        audio_path=str(audio_path),
                        transcript=transcript,
                        duration=float(row.get("duration", 0.0) or 0.0),
                        sample_rate=kwargs.get("sample_rate", 16000),
                        speaker_id=speaker_id,
                        primary_language=languages[0] if languages else "twi",
                        languages=languages,
                        country="GH",
                        verified=True,
                        code_switching=len(languages) > 1 or row.get("code_switching", "false").lower() == "true",
                    )
                    records.append(record)

        elif meta_path.suffix == ".jsonl":
            with meta_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    record_id = item.get("id") or f"kasa_{len(records)}"
                    audio_path = audio_dir / item.get("audio_path", item.get("audio", ""))
                    transcript = item.get("transcript", item.get("text", ""))
                    languages = item.get("languages") or [item.get("language", language or "twi")]

                    record = self.build_record(
                        record_id=record_id,
                        audio_path=str(audio_path),
                        transcript=transcript,
                        duration=float(item.get("duration", 0.0)),
                        sample_rate=kwargs.get("sample_rate", 16000),
                        speaker_id=item.get("speaker_id"),
                        primary_language=item.get("primary_language", languages[0]),
                        languages=languages,
                        country="GH",
                        verified=bool(item.get("verified", True)),
                    )
                    records.append(record)

        if output_manifest:
            write_manifest(output_manifest, records)

        return records

