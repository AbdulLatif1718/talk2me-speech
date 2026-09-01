"""XLS-R model wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class XLSRModel:
    """Simple wrapper around an XLS-R-style model configuration."""

    model_name: str = "xlsr-300m"
    cache_dir: str | None = None

    def load(self) -> dict[str, Any]:
        """Return a configuration dictionary for the model loader."""
        return {
            "model_name": self.model_name,
            "cache_dir": self.cache_dir,
            "architecture": "xlsr",
        }
