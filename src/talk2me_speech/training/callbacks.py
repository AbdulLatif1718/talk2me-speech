"""Training callbacks."""

from __future__ import annotations


class TrainingCallback:
    """A lightweight callback interface used by training pipelines."""

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> None:
        """Invoked at the end of each epoch."""
        _ = (epoch, metrics)

    def on_train_end(self) -> None:
        """Invoked at the end of training."""
        return None
