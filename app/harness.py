"""Runs the retriever against a labeled query set and aggregates metrics."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

from app.metrics import precision_at_k, recall_at_k, reciprocal_rank
from app.retriever import Document, TfidfRetriever, load_corpus

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass
class QueryLabel:
    query_id: str
    query: str
    relevant_doc_ids: List[str]


@dataclass
class QueryResult:
    query_id: str
    query: str
    retrieved_doc_ids: List[str]
    relevant_doc_ids: List[str]
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float


@dataclass
class EvalReport:
    k: int
    num_queries: int
    mean_precision_at_k: float
    mean_recall_at_k: float
    mean_reciprocal_rank: float
    per_query: List[QueryResult]

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def load_qrels(path: Path) -> List[QueryLabel]:
    raw = json.loads(Path(path).read_text())
    return [QueryLabel(**r) for r in raw]


def build_retriever(corpus_path: Path = DATA_DIR / "corpus.json") -> TfidfRetriever:
    docs: List[Document] = load_corpus(corpus_path)
    return TfidfRetriever(docs)


def run_eval(
    retriever: TfidfRetriever,
    qrels: List[QueryLabel],
    k: int = 5,
) -> EvalReport:
    per_query: List[QueryResult] = []
    for label in qrels:
        results = retriever.retrieve(label.query, k=k)
        retrieved_ids = [r.doc_id for r in results]
        p = precision_at_k(retrieved_ids, label.relevant_doc_ids, k)
        r = recall_at_k(retrieved_ids, label.relevant_doc_ids, k)
        rr = reciprocal_rank(retrieved_ids, label.relevant_doc_ids)
        per_query.append(
            QueryResult(
                query_id=label.query_id,
                query=label.query,
                retrieved_doc_ids=retrieved_ids,
                relevant_doc_ids=label.relevant_doc_ids,
                precision_at_k=p,
                recall_at_k=r,
                reciprocal_rank=rr,
            )
        )

    n = len(per_query) or 1
    mean_p = sum(q.precision_at_k for q in per_query) / n
    mean_r = sum(q.recall_at_k for q in per_query) / n
    mean_rr = sum(q.reciprocal_rank for q in per_query) / n

    return EvalReport(
        k=k,
        num_queries=len(per_query),
        mean_precision_at_k=mean_p,
        mean_recall_at_k=mean_r,
        mean_reciprocal_rank=mean_rr,
        per_query=per_query,
    )


def run_default_eval(k: int = 5) -> EvalReport:
    retriever = build_retriever()
    qrels = load_qrels(DATA_DIR / "qrels.json")
    return run_eval(retriever, qrels, k=k)
