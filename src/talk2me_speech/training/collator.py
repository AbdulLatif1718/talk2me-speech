"""Training data collation helpers for Speech Seq2Seq (Whisper) and CTC (XLS-R)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Convert a batch of record dicts into a simple dictionary structure."""
    return {
        "items": list(batch),
        "batch_size": len(batch),
    }


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Data collator for Whisper speech Seq2Seq training with dynamic padding."""

    processor: Any

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate speech features and labels with padding."""
        import torch

        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )

        # Replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # If bos token is appended in previous tokenization step, cut it here
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


@dataclass
class DataCollatorCTCWithPadding:
    """Data collator for Wav2Vec2 / XLS-R CTC loss training."""

    processor: Any
    padding: bool | str = True

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate audio input values and CTC target labels."""
        import torch

        input_values = [{"input_values": feature["input_values"]} for feature in features]
        batch = self.processor.pad(
            input_values,
            padding=self.padding,
            return_tensors="pt",
        )

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            return_tensors="pt",
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels
        return batch
