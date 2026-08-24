"""Retrieval evaluation metrics: precision@k, recall@k, and MRR."""
from __future__ import annotations

from typing import Iterable, List


def precision_at_k(retrieved_ids: List[str], relevant_ids: Iterable[str], k: int) -> float:
    """Fraction of the top-k retrieved docs that are relevant."""
    if k <= 0:
        raise ValueError("k must be a positive integer")
    relevant = set(relevant_ids)
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for doc_id in top_k if doc_id in relevant)
    return hits / len(top_k)


def recall_at_k(retrieved_ids: List[str], relevant_ids: Iterable[str], k: int) -> float:
    """Fraction of all relevant docs that appear in the top-k retrieved."""
    if k <= 0:
        raise ValueError("k must be a positive integer")
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    top_k = set(retrieved_ids[:k])
    hits = len(top_k & relevant)
    return hits / len(relevant)


def reciprocal_rank(retrieved_ids: List[str], relevant_ids: Iterable[str]) -> float:
    """1 / rank of the first relevant doc in the ranked list (0 if none found)."""
    relevant = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0
