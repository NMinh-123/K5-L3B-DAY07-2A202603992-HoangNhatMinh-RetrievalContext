"""Dense retrieval on the real Day 6 Qdrant index.

The query embedding must match how Day 6 embedded the chunks, so the settings
below copy Day 6 `embeddings.py` exactly: intfloat/multilingual-e5-large at a
pinned revision, prefix "query: ", L2-normalized, cosine distance in Qdrant.
A mismatch (other model, revision, prefix or dimension) is an error, never a
silent fallback.
"""
import os
import time
from typing import Any

MODEL = "intfloat/multilingual-e5-large"
REVISION = "3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3"
QUERY_PREFIX = "query: "
DIMENSIONS = 1024


def _device() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


class E5QueryEncoder:
    def __init__(self, model: str = MODEL, revision: str = REVISION, device: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer
        self.model_name, self.revision = model, revision
        self.device = device or _device()
        self.model = SentenceTransformer(model, revision=revision, device=self.device)
        get_dim = getattr(self.model, "get_embedding_dimension", None) \
            or self.model.get_sentence_embedding_dimension  # older sentence-transformers
        dims = get_dim()
        if dims != DIMENSIONS:
            raise RuntimeError(f"{model} trả {dims} chiều, index Day 6 dùng {DIMENSIONS}")

    def encode(self, query: str) -> list[float]:
        if not query or not query.strip():
            raise ValueError("query rỗng")
        vec = self.model.encode([QUERY_PREFIX + query], normalize_embeddings=True, convert_to_numpy=True)
        return vec[0].tolist()


class DenseRetriever:
    """Search one Day 6 strategy (index) in Qdrant; returns Day 7 chunk dicts with score and rank."""

    name = "dense"

    def __init__(self, strategy: str = "D", url: str | None = None, collection: str = "day6_chunks",
                 encoder: E5QueryEncoder | None = None, exact: bool = True) -> None:
        from qdrant_client import QdrantClient
        self.strategy, self.collection, self.exact = strategy, collection, exact
        self.url = url or os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
        self.client = QdrantClient(url=self.url, timeout=30)
        info = self.client.get_collection(collection).config.params.vectors
        if info.size != DIMENSIONS or str(info.distance).lower().find("cosine") < 0:
            raise RuntimeError(f"Collection {collection}: size={info.size}, distance={info.distance}; "
                               f"cần {DIMENSIONS} chiều, cosine")
        self.encoder = encoder or E5QueryEncoder()
        self.last_timing: dict[str, float] = {}

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        from qdrant_client import models
        if top_k < 1:
            raise ValueError("top_k phải >= 1")
        t0 = time.perf_counter()
        vector = self.encoder.encode(query)
        t1 = time.perf_counter()
        must = [models.FieldCondition(key="strategy", match=models.MatchValue(value=self.strategy))]
        for key, value in (filters or {}).items():
            field = key if key in ("document_id", "source", "page") else f"metadata.{key}"
            match = (models.MatchAny(any=list(value)) if isinstance(value, (list, tuple, set))
                     else models.MatchValue(value=value))
            must.append(models.FieldCondition(key=field, match=match))
        resp = self.client.query_points(self.collection, query=vector, query_filter=models.Filter(must=must),
                                        limit=top_k, with_payload=True,
                                        search_params=models.SearchParams(exact=self.exact))
        t2 = time.perf_counter()
        self.last_timing = {"embed_ms": (t1 - t0) * 1000, "search_ms": (t2 - t1) * 1000}
        out = []
        for rank, point in enumerate(resp.points, start=1):
            p = point.payload or {}
            out.append({"id": p["chunk_id"], "text": p["text"], "source_id": p["document_id"],
                        "source": p["source"], "page": p["page"], "metadata": p.get("metadata", {}),
                        "score": float(point.score), "rank": rank, "retriever": self.name})
        return out

    def describe(self) -> dict[str, Any]:
        return {"retriever": self.name, "model": MODEL, "revision": REVISION, "query_prefix": QUERY_PREFIX,
                "dimensions": DIMENSIONS, "normalized": True, "device": self.encoder.device,
                "vector_store": "qdrant", "url": self.url, "collection": self.collection,
                "strategy": self.strategy, "distance": "cosine", "exact": self.exact, "cost_usd": 0}
