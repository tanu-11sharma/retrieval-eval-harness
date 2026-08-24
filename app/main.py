"""FastAPI app exposing the retriever and the evaluation harness.

This is a self-contained demo: it runs entirely against the bundled
synthetic corpus and labeled query set in data/, no external API keys
or live services required.
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from app.harness import build_retriever, load_qrels, run_eval, DATA_DIR

app = FastAPI(
    title="Retrieval Eval Harness",
    description="Evaluates a retriever's precision/recall/MRR against a labeled query set.",
    version="0.1.0",
)

_retriever = build_retriever()
_qrels = load_qrels(DATA_DIR / "qrels.json")


class RetrieveRequest(BaseModel):
    query: str
    k: int = 5


class ScoredDocOut(BaseModel):
    doc_id: str
    score: float


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/retrieve", response_model=list[ScoredDocOut])
def retrieve(req: RetrieveRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")
    results = _retriever.retrieve(req.query, k=req.k)
    return [ScoredDocOut(doc_id=r.doc_id, score=r.score) for r in results]


@app.get("/evaluate")
def evaluate(k: int = Query(default=5, ge=1, le=15)):
    report = run_eval(_retriever, _qrels, k=k)
    return report.to_dict()


@app.get("/corpus")
def corpus():
    return [
        {"id": d.id, "title": d.title, "text": d.text} for d in _retriever.documents
    ]
