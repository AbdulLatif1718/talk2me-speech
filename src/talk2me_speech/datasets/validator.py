"""Dataset validation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

REQUIRED_FIELDS = {"id", "audio_path", "transcript"}


def validate_dataset_root(root: str | Path) -> bool:
    """Return True if a dataset root exists and has at least one valid subfolder or manifest."""
    dataset_root = Path(root)
    if not dataset_root.exists():
        return False
    expected_subdirs = ["train", "validation", "test", "manifests", "raw", "processed"]
    return any((dataset_root / name).exists() for name in expected_subdirs)


def validate_record_schema(record: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate a single record against the canonical Talk2Me schema."""
    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in record or record[field] is None:
            errors.append(f"Missing required field: '{field}'")
        elif isinstance(record[field], str) and not record[field].strip():
            errors.append(f"Field '{field}' cannot be empty")

    if "duration" in record and record["duration"] is not None:
        try:
            if float(record["duration"]) < 0:
                errors.append("Field 'duration' cannot be negative")
        except (ValueError, TypeError):
            errors.append("Field 'duration' must be a numeric value")

    return len(errors) == 0, errors


def validate_manifest_records(records: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    """Validate a list of records for canonical compliance."""
    all_errors: list[str] = []
    if not records:
        return False, ["Manifest contains no records"]

    for idx, record in enumerate(records):
        valid, errors = validate_record_schema(record)
        if not valid:
            for err in errors:
                all_errors.append(f"Record index {idx} ({record.get('id', 'unknown')}): {err}")

    return len(all_errors) == 0, all_errors
