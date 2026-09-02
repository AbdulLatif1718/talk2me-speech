"""Mozilla Common Voice dataset adapter."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from talk2me_speech.datasets.adapters.base import BaseDatasetAdapter
from talk2me_speech.datasets.manifest import write_manifest


class CommonVoiceAdapter(BaseDatasetAdapter):
    """Adapter for importing and canonicalizing Mozilla Common Voice TSV releases."""

    def __init__(self) -> None:
        super().__init__(source_name="commonvoice")

    def convert(
        self,
        input_dir: str | Path,
        output_manifest: str | Path | None = None,
        language: str | None = "en",
        tsv_filename: str = "validated.tsv",
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Parse Common Voice TSV files (e.g. validated.tsv, train.tsv) and clips.

        Common Voice TSV format typically includes columns:
        client_id, path, sentence, up_votes, down_votes, age, gender, accent, locale, segment
        """
        in_path = Path(input_dir)
        tsv_path = in_path / tsv_filename if (in_path / tsv_filename).exists() else in_path
        clips_dir = in_path / "clips" if (in_path / "clips").exists() else in_path

        records: list[dict[str, Any]] = []

        if not tsv_path.exists() or tsv_path.is_dir():
            return records

        with tsv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                sentence = row.get("sentence", "").strip()
                if not sentence:
                    continue

                clip_name = row.get("path", "")
                audio_file = clips_dir / clip_name
                client_id = row.get("client_id")
                accent = row.get("accent") or row.get("locale") or language
                country = kwargs.get("country")
                if not country and accent and "ghana" in str(accent).lower():
                    country = "GH"

                record_id = f"cv_{clip_name.replace('.mp3', '').replace('.wav', '')}"
                
                record = self.build_record(
                    record_id=record_id,
                    audio_path=str(audio_file),
                    transcript=sentence,
                    duration=float(row.get("duration", 0.0) or 0.0),
                    sample_rate=kwargs.get("sample_rate", 16000),
                    speaker_id=client_id,
                    primary_language=language,
                    languages=[language] if language else None,
                    country=country,
                    verified=int(row.get("up_votes", 0) or 0) >= int(row.get("down_votes", 0) or 0),
                    age=row.get("age"),
                    gender=row.get("gender"),
                    accent=accent,
                )
                records.append(record)

        if output_manifest:
            write_manifest(output_manifest, records)

        return records

