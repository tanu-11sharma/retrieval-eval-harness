# Retrieval Eval Harness

A small, self-contained harness for evaluating a RAG retriever's quality —
precision@k, recall@k, and Mean Reciprocal Rank (MRR) — against a hand-labeled
query set, exposed both as a FastAPI service and a CLI.

## Why this exists

Most RAG demos show a retriever answering *some* query and stop there. The
harder, more production-relevant question is: **how do you know your
retriever is actually good, and whether a change made it better or worse?**
That requires a labeled query set (qrels) and standard information-retrieval
metrics — the same discipline used to evaluate search and recommendation
systems before LLMs existed. This project builds that offline evaluation
loop end to end: a retriever, a labeled query set, metric implementations,
and a report.

## What's inside

- `app/retriever.py` — a TF-IDF + cosine-similarity retriever (scikit-learn)
  over a small synthetic corpus of 15 short documents about RAG/AI-engineering
  topics (`data/corpus.json`).
- `app/metrics.py` — `precision_at_k`, `recall_at_k`, and `reciprocal_rank`,
  implemented from scratch and unit-tested against hand-computed values.
- `app/harness.py` — runs the retriever against a labeled query set
  (`data/qrels.json`, 8 queries with ground-truth relevant doc ids) and
  aggregates per-query and mean metrics.
- `app/main.py` — a FastAPI app exposing `/retrieve`, `/evaluate`, and
  `/corpus` endpoints.
- `app/cli.py` — a CLI that runs the same harness and prints a report table.
- `tests/` — 20 tests covering the metric formulas, retriever behavior, the
  harness end to end, and the API endpoints.

Everything runs against bundled synthetic data. No external API keys, no
network calls, no live services required.

## Setup

```bash
pip install -r requirements.txt
```

## Run the CLI report

```bash
python -m app.cli --k 5
```

Example output:

```
Retrieval Eval Report (k=5, queries=8)
------------------------------------------------------------------------
query_id  precision@k   recall@k    MRR
q1        0.40          0.67        1.00
q2        0.40          1.00        1.00
...
------------------------------------------------------------------------
mean precision@k=0.375  mean recall@k=0.917  mean MRR=0.875
```

(Actual numbers are computed live from the bundled data each run — the
figures above are illustrative of the report shape, not a claimed benchmark.)

## Run the API

```bash
uvicorn app.main:app --reload
```

Example usage:

```bash
curl -X POST http://localhost:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "What is BM25 lexical ranking?", "k": 3}'

curl "http://localhost:8000/evaluate?k=5"
```

## Run the tests

```bash
pytest -q
```

## Extending this

Swap `TfidfRetriever` for a dense embedding retriever (e.g. sentence
embeddings + FAISS/Qdrant) or a hybrid retriever, keep `data/qrels.json` (or
a larger labeled set) fixed, and re-run the harness — the metrics module and
report format stay the same, which is the point of separating the retriever
from the evaluation harness.

## Disclaimer

This is a demo/learning project using a small synthetic corpus and labeled
query set. It is not a benchmark of any production system and does not
represent real user traffic or metrics.
