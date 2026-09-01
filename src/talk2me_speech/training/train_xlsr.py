"""Entry point for XLS-R training configuration."""

from __future__ import annotations

from pathlib import Path


def build_train_config(output_dir: str | Path = "./models/checkpoints") -> dict[str, object]:
    """Return a config dictionary for an XLS-R training run."""
    return {
        "model_type": "xlsr",
        "epochs": 10,
        "batch_size": 4,
        "learning_rate": 3e-5,
        "output_dir": str(output_dir),
    }
