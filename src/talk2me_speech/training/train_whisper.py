"""Entry point for Whisper training configuration."""

from __future__ import annotations

from pathlib import Path


def build_train_config(output_dir: str | Path = "./models/checkpoints") -> dict[str, object]:
    """Return a config dictionary for a Whisper training run."""
    return {
        "model_type": "whisper",
        "epochs": 5,
        "batch_size": 8,
        "learning_rate": 5e-5,
        "output_dir": str(output_dir),
    }
