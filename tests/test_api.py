from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_retrieve_endpoint():
    resp = client.post("/retrieve", json={"query": "BM25 lexical ranking", "k": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 3
    assert body[0]["doc_id"] == "d3"


def test_retrieve_rejects_empty_query():
    resp = client.post("/retrieve", json={"query": "   ", "k": 3})
    assert resp.status_code == 400


def test_evaluate_endpoint():
    resp = client.get("/evaluate", params={"k": 5})
    assert resp.status_code == 200
    body = resp.json()
    assert body["k"] == 5
    assert body["num_queries"] >= 1
    assert 0.0 <= body["mean_precision_at_k"] <= 1.0


def test_corpus_endpoint():
    resp = client.get("/corpus")
    assert resp.status_code == 200
    assert len(resp.json()) == 15
