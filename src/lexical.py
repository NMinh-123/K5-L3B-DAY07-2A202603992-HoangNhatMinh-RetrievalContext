"""BM25 lexical retrieval and hybrid (dense + BM25) fusion over the same Day 6 chunks.

BM25 (Okapi, k1=1.5, b=0.75) indexes syllable unigrams plus adjacent-syllable
bigrams: Vietnamese words are often two syllables ("kháng sinh", "thương hàn"),
so bigrams keep some word identity without a segmenter. Tokens: NFC, casefold, \\w+.

Hybrid uses Reciprocal Rank Fusion: score = sum over retrievers of 1 / (k + rank),
k=60, on the top-`depth` candidates of each retriever. RRF needs no score
calibration between cosine and BM25.

Filters (user-provided metadata, e.g. {"disease": "cum_mua"}) are applied
before ranking in BM25 and inside Qdrant for dense, so both see the same subset.
"""
import math
import re
import time
import unicodedata
from collections import Counter
from typing import Any


def tokens(text: str) -> list[str]:
    words = re.findall(r"\w+", unicodedata.normalize("NFC", text).casefold())
    return words + [f"{a}_{b}" for a, b in zip(words, words[1:])]


def matches(chunk: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    for key, wanted in (filters or {}).items():
        actual = chunk.get("source_id" if key == "document_id" else key) \
            if key in ("document_id", "source", "page") else (chunk.get("metadata") or {}).get(key)
        if isinstance(wanted, (list, tuple, set)):
            if actual not in wanted:
                return False
        elif actual != wanted:
            return False
    return True


class BM25Retriever:
    name = "bm25"

    def __init__(self, chunks: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks, self.k1, self.b = chunks, k1, b
        self.tf = [Counter(tokens(c["text"])) for c in chunks]
        self.len = [sum(t.values()) for t in self.tf]
        self.avg = sum(self.len) / len(self.len)
        df = Counter(term for t in self.tf for term in t)
        n = len(chunks)
        self.idf = {term: math.log(1 + (n - d + 0.5) / (d + 0.5)) for term, d in df.items()}
        self.last_timing: dict[str, float] = {}

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        if top_k < 1:
            raise ValueError("top_k phải >= 1")
        t0 = time.perf_counter()
        q = [t for t in dict.fromkeys(tokens(query)) if t in self.idf]
        scored = []
        for i, c in enumerate(self.chunks):
            if not matches(c, filters):
                continue
            tf, norm = self.tf[i], self.k1 * (1 - self.b + self.b * self.len[i] / self.avg)
            s = sum(self.idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + norm) for t in q if t in tf)
            if s > 0:
                scored.append((s, c))
        scored.sort(key=lambda x: (-x[0], x[1]["id"]))
        self.last_timing = {"search_ms": (time.perf_counter() - t0) * 1000}
        return [{**c, "score": s, "rank": r, "retriever": self.name}
                for r, (s, c) in enumerate(scored[:top_k], start=1)]

    def describe(self) -> dict[str, Any]:
        return {"retriever": self.name, "k1": self.k1, "b": self.b, "tokens": "syllable unigram + bigram, NFC casefold",
                "chunks": len(self.chunks), "cost_usd": 0}


class HybridRetriever:
    name = "hybrid"

    def __init__(self, dense: Any, lexical: BM25Retriever, depth: int = 20, k: int = 60) -> None:
        self.dense, self.lexical, self.depth, self.k = dense, lexical, depth, k

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        if top_k < 1:
            raise ValueError("top_k phải >= 1")
        fused: dict[str, dict[str, Any]] = {}
        for retriever in (self.dense, self.lexical):
            for c in retriever.search(query, filters=filters, top_k=self.depth):
                entry = fused.setdefault(c["id"], {**c, "score": 0.0, "sources": {}})
                entry["score"] += 1 / (self.k + c["rank"])
                entry["sources"][retriever.name] = c["rank"]
        ranked = sorted(fused.values(), key=lambda c: (-c["score"], c["id"]))[:top_k]
        return [{**c, "rank": r, "retriever": self.name} for r, c in enumerate(ranked, start=1)]

    def describe(self) -> dict[str, Any]:
        return {"retriever": self.name, "fusion": "RRF", "k": self.k, "depth": self.depth,
                "dense": self.dense.describe(), "lexical": self.lexical.describe(), "cost_usd": 0}
