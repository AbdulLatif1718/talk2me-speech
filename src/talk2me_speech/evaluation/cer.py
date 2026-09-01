"""Character error rate utilities."""

from __future__ import annotations


def compute_cer(reference: str, hypothesis: str) -> float:
    """Compute a simple CER approximation as a normalized edit distance."""
    if reference == hypothesis:
        return 0.0

    ref_chars = list(reference)
    hyp_chars = list(hypothesis)
    if not ref_chars and not hyp_chars:
        return 0.0

    max_len = max(len(ref_chars), len(hyp_chars))
    if max_len == 0:
        return 0.0

    matrix = [[0 for _ in range(len(hyp_chars) + 1)] for _ in range(len(ref_chars) + 1)]
    for i in range(len(ref_chars) + 1):
        matrix[i][0] = i
    for j in range(len(hyp_chars) + 1):
        matrix[0][j] = j

    for i in range(1, len(ref_chars) + 1):
        for j in range(1, len(hyp_chars) + 1):
            cost = 0 if ref_chars[i - 1] == hyp_chars[j - 1] else 1
            matrix[i][j] = min(
                matrix[i - 1][j] + 1,
                matrix[i][j - 1] + 1,
                matrix[i - 1][j - 1] + cost,
            )

    return matrix[len(ref_chars)][len(hyp_chars)] / max_len
