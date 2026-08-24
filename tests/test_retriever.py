from pathlib import Path

from app.retriever import TfidfRetriever, load_corpus

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _retriever():
    return TfidfRetriever(load_corpus(DATA_DIR / "corpus.json"))


def test_retrieve_returns_k_results():
    r = _retriever()
    results = r.retrieve("retrieval augmented generation", k=3)
    assert len(results) == 3
    assert all(res.score >= 0 for res in results)


def test_top_result_is_topically_relevant():
    r = _retriever()
    results = r.retrieve("What is BM25 lexical ranking?", k=1)
    assert results[0].doc_id == "d3"


def test_results_sorted_descending_by_score():
    r = _retriever()
    results = r.retrieve("vector database embeddings", k=5)
    scores = [res.score for res in results]
    assert scores == sorted(scores, reverse=True)


def test_rejects_non_positive_k():
    r = _retriever()
    try:
        r.retrieve("anything", k=0)
        assert False, "expected ValueError"
    except ValueError:
        pass
