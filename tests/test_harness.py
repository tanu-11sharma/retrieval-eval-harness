from app.harness import build_retriever, load_qrels, run_eval, DATA_DIR


def test_run_eval_produces_report_for_all_queries():
    retriever = build_retriever()
    qrels = load_qrels(DATA_DIR / "qrels.json")
    report = run_eval(retriever, qrels, k=5)

    assert report.num_queries == len(qrels)
    assert 0.0 <= report.mean_precision_at_k <= 1.0
    assert 0.0 <= report.mean_recall_at_k <= 1.0
    assert 0.0 <= report.mean_reciprocal_rank <= 1.0
    assert len(report.per_query) == len(qrels)


def test_easy_query_scores_perfectly_at_top1():
    # q7 has a single, unambiguous relevant doc (d13, guardrails) -- the
    # retriever should surface it at rank 1 even with k=1.
    retriever = build_retriever()
    qrels = load_qrels(DATA_DIR / "qrels.json")
    q7 = next(q for q in qrels if q.query_id == "q7")
    report = run_eval(retriever, [q7], k=1)
    result = report.per_query[0]
    assert result.retrieved_doc_ids == ["d13"]
    assert result.precision_at_k == 1.0
    assert result.reciprocal_rank == 1.0


def test_report_serializes_to_dict():
    retriever = build_retriever()
    qrels = load_qrels(DATA_DIR / "qrels.json")
    report = run_eval(retriever, qrels, k=3)
    d = report.to_dict()
    assert d["k"] == 3
    assert "per_query" in d and isinstance(d["per_query"], list)
