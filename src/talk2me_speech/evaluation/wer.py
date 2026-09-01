"""Word error rate utilities."""

from __future__ import annotations


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute a simple WER approximation as a normalized Levenshtein distance."""
    if reference == hypothesis:
        return 0.0

    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    if not ref_tokens and not hyp_tokens:
        return 0.0

    max_len = max(len(ref_tokens), len(hyp_tokens))
    if max_len == 0:
        return 0.0

    matrix = [[0 for _ in range(len(hyp_tokens) + 1)] for _ in range(len(ref_tokens) + 1)]
    for i in range(len(ref_tokens) + 1):
        matrix[i][0] = i
    for j in range(len(hyp_tokens) + 1):
        matrix[0][j] = j

    for i in range(1, len(ref_tokens) + 1):
        for j in range(1, len(hyp_tokens) + 1):
            cost = 0 if ref_tokens[i - 1] == hyp_tokens[j - 1] else 1
            matrix[i][j] = min(
                matrix[i - 1][j] + 1,
                matrix[i][j - 1] + 1,
                matrix[i - 1][j - 1] + cost,
            )

    return matrix[len(ref_tokens)][len(hyp_tokens)] / max_len
