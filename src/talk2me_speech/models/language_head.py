"""Language adaptation head helpers."""

from __future__ import annotations


class LanguageHead:
    """Placeholder for language-specific classifier or adapter logic."""

    def __init__(self, vocab_size: int = 5000, languages: list[str] | None = None):
        self.vocab_size = vocab_size
        self.languages = languages or ["en", "fr", "twi", "mixed"]

    def build(self) -> dict[str, object]:
        return {
            "vocab_size": self.vocab_size,
            "languages": self.languages,
        }
