"""Evaluation metrics for QA datasets."""

from __future__ import annotations

import math
from typing import Iterable, List


def _tokenize(text: str) -> List[str]:
    return [tok for tok in text.lower().split() if tok]


def _lcs_length(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0

    dp = [0] * (len(b) + 1)
    for token_a in a:
        prev = 0
        for j, token_b in enumerate(b, start=1):
            temp = dp[j]
            if token_a == token_b:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp
    return dp[-1]


def rouge_l_f1(reference: str, prediction: str) -> float:
    """Compute ROUGE-L F1 between reference and prediction."""
    ref_tokens = _tokenize(reference)
    pred_tokens = _tokenize(prediction)
    if not ref_tokens or not pred_tokens:
        return 0.0

    lcs = _lcs_length(ref_tokens, pred_tokens)
    precision = lcs / len(pred_tokens) if pred_tokens else 0.0
    recall = lcs / len(ref_tokens) if ref_tokens else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def cosine_similarity(vec_a: Iterable[float], vec_b: Iterable[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a = list(vec_a)
    b = list(vec_b)
    if not a or not b or len(a) != len(b):
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
