"""Named-entity evaluation helpers."""

from __future__ import annotations


def extract_named_entities(text: str) -> list[str]:
    """Return a naive token list of text chunks that look like entities."""
    tokens = [token.strip(".,;:!?\"'") for token in text.split()]
    return [token for token in tokens if len(token) > 2 and token[0].isupper()]
