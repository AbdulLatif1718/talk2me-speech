"""Code-switch evaluation helpers."""

from __future__ import annotations


def code_switch_score(reference: str, hypothesis: str) -> float:
    """Return a simple score based on mismatched language tokens in a mixed-language transcript."""
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    if not ref_tokens and not hyp_tokens:
        return 0.0

    overlap = sum(1 for token in set(ref_tokens) & set(hyp_tokens))
    union = len(set(ref_tokens) | set(hyp_tokens))
    return 0.0 if union == 0 else overlap / union
