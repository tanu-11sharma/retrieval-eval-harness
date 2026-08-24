import pytest

from app.metrics import precision_at_k, recall_at_k, reciprocal_rank


def test_precision_at_k_basic():
    retrieved = ["a", "b", "c", "d"]
    relevant = {"a", "c"}
    assert precision_at_k(retrieved, relevant, k=4) == 0.5
    assert precision_at_k(retrieved, relevant, k=2) == 0.5
    assert precision_at_k(retrieved, relevant, k=1) == 1.0


def test_precision_at_k_no_hits():
    assert precision_at_k(["x", "y"], {"a"}, k=2) == 0.0


def test_precision_at_k_rejects_non_positive_k():
    with pytest.raises(ValueError):
        precision_at_k(["a"], {"a"}, k=0)


def test_recall_at_k_basic():
    retrieved = ["a", "b", "c"]
    relevant = {"a", "c", "z"}
    # 2 of 3 relevant docs (a, c) found within top 3
    assert recall_at_k(retrieved, relevant, k=3) == pytest.approx(2 / 3)


def test_recall_at_k_empty_relevant():
    assert recall_at_k(["a", "b"], [], k=2) == 0.0


def test_reciprocal_rank_first_position():
    assert reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0


def test_reciprocal_rank_third_position():
    assert reciprocal_rank(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)


def test_reciprocal_rank_no_match():
    assert reciprocal_rank(["x", "y"], {"a"}) == 0.0
