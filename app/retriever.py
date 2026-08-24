"""A small TF-IDF retriever used as the system under test for the eval harness."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Document:
    id: str
    title: str
    text: str

    @property
    def full_text(self) -> str:
        return f"{self.title}. {self.text}"


@dataclass
class ScoredDoc:
    doc_id: str
    score: float


class TfidfRetriever:
    """Cosine-similarity retriever over TF-IDF vectors.

    Deliberately simple (no external services, no API keys) so the whole
    demo runs offline against the bundled synthetic corpus.
    """

    def __init__(self, documents: List[Document]):
        if not documents:
            raise ValueError("documents must be non-empty")
        self.documents = documents
        self._ids = [d.id for d in documents]
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = self._vectorizer.fit_transform([d.full_text for d in documents])

    def retrieve(self, query: str, k: int = 5) -> List[ScoredDoc]:
        if k <= 0:
            raise ValueError("k must be a positive integer")
        query_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self._matrix)[0]
        ranked = sorted(zip(self._ids, sims), key=lambda pair: pair[1], reverse=True)
        return [ScoredDoc(doc_id=doc_id, score=float(score)) for doc_id, score in ranked[:k]]


def load_corpus(path: Path) -> List[Document]:
    raw = json.loads(Path(path).read_text())
    return [Document(id=r["id"], title=r["title"], text=r["text"]) for r in raw]
