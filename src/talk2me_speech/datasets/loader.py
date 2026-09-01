"""Dataset loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    """Load a manifest file expected to be a JSONL or JSON list."""
    manifest_path = Path(path)
    if not manifest_path.exists():
        return []

    content = manifest_path.read_text(encoding="utf-8").strip()
    if not content:
        return []

    if content.startswith("["):
        import json

        return json.loads(content)

    entries = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        import json

        entries.append(json.loads(line))
    return entries
