"""Retrieval-only benchmark: Hit@1, Evidence Hit@3 and Context Evidence Rate per retriever.

Only answerable questions of the chosen split are scored (unanswerable ones are
not in the retrieval denominator, per BENCHMARK.md). Gold is used only after the
search to mark hits; it never reaches the retriever or the query.

    python src/eval_retrieval.py --retriever dense --split dev --output results/retrieval/dev-dense
    python src/eval_retrieval.py --retriever tfidf --split dev --output results/retrieval/dev-tfidf
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "starter"))
sys.path.insert(0, str(ROOT / "src"))
from baseline import TfidfIndex, pack_context  # noqa: E402
from evaluation import context_has_evidence, is_hit  # noqa: E402

DATA = ROOT / "data" / "day6"


class TfidfRetriever:
    name = "tfidf"

    def __init__(self, chunks):
        self.index = TfidfIndex(chunks)

    def search(self, query, *, filters=None, top_k=3):
        if filters:
            raise ValueError("TF-IDF baseline không hỗ trợ filter")
        return [{**c, "rank": i, "retriever": self.name} for i, c in enumerate(self.index.search(query, top_k), 1)]

    def describe(self):
        return {"retriever": self.name, "tokenizer": r"\w+ casefold NFC", "cost_usd": 0}


RETRIEVERS = ("tfidf", "bm25", "dense", "hybrid")


def build(name: str, strategy: str):
    """All retrievers search the same corpus: Day 6 index `strategy` (exported by prepare_data.py)."""
    chunks = json.loads((DATA / f"chunks_{strategy}.json").read_text(encoding="utf-8"))
    if name == "tfidf":
        return TfidfRetriever(chunks)
    from lexical import BM25Retriever, HybridRetriever
    if name == "bm25":
        return BM25Retriever(chunks)
    from dense import DenseRetriever
    if name == "dense":
        return DenseRetriever(strategy=strategy)
    if name == "hybrid":
        return HybridRetriever(DenseRetriever(strategy=strategy), BM25Retriever(chunks))
    raise ValueError(f"retriever không hỗ trợ: {name}")


def frac(n: int, d: int) -> str:
    return f"{n}/{d} ({n / d:.0%})" if d else "0/0 (không áp dụng)"


def main() -> None:
    ap = argparse.ArgumentParser(description="Chấm retrieval trên một split")
    ap.add_argument("--retriever", required=True, choices=RETRIEVERS)
    ap.add_argument("--split", required=True, choices=("dev", "test"))
    ap.add_argument("--strategy", default="D", help="Index Day 6 (corpus) dùng chung cho mọi retriever")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--budget", type=int, default=800, help="Context budget, đơn vị ký tự (gồm header nguồn)")
    ap.add_argument("--use-filter", action="store_true", help="Dùng filter người dùng cung cấp trong câu hỏi")
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    if args.top_k < 3:
        raise ValueError("cần top_k >= 3 để báo Evidence Hit@3")

    questions = [q for q in json.loads((DATA / "questions.json").read_text(encoding="utf-8"))
                 if q["split"] == args.split]
    t0 = time.perf_counter()
    retriever = build(args.retriever, args.strategy)
    setup_s = time.perf_counter() - t0

    traces, errors = [], []
    for q in questions:
        filters = q.get("filter") if args.use_filter else None
        start = time.perf_counter()
        try:
            results = retriever.search(q["question"], filters=filters, top_k=args.top_k)
        except Exception as exc:  # record, never replace with fake results
            errors.append({"query_id": q["id"], "error": repr(exc)})
            continue
        ms = (time.perf_counter() - start) * 1000
        context, ids = pack_context(results, budget=args.budget)
        in_context = [c for c in results if c["id"] in ids]
        marked = [{"rank": c["rank"], "id": c["id"], "page": c["page"], "score": round(c["score"], 4),
                   "hit": is_hit(c, q["gold"]), "in_context": c["id"] in ids} for c in results]
        traces.append({"query_id": q["id"], "split": q["split"], "query": q["question"], "filter": filters,
                       "gold_pages": [g["page"] for g in q["gold"]], "results": marked,
                       "context_ids": ids, "context_chars": len(context), "elapsed_ms": round(ms, 1),
                       "hit1": bool(marked) and marked[0]["hit"], "hit3": any(m["hit"] for m in marked[:3]),
                       "context_evidence": context_has_evidence(in_context, q["gold"])})

    n = len(questions)
    summary = {
        "retriever": retriever.describe(), "split": args.split, "corpus_index": args.strategy,
        "top_k": args.top_k, "budget_chars": args.budget, "use_filter": args.use_filter,
        "N": n, "errors": len(errors),
        "hit_at_1": frac(sum(t["hit1"] for t in traces), n),
        "evidence_hit_at_3": frac(sum(t["hit3"] for t in traces), n),
        "context_evidence_rate": frac(sum(t["context_evidence"] for t in traces), n),
        "avg_context_chars": round(statistics.mean(t["context_chars"] for t in traces), 1) if traces else 0,
        "median_ms": round(statistics.median(t["elapsed_ms"] for t in traces), 1) if traces else 0,
        "setup_s": round(setup_s, 2), "cost_usd": 0,
        "misses_at_3": [t["query_id"] for t in traces if not t["hit3"]],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for name, obj in (("summary.json", summary), ("traces.json", traces), ("errors.json", errors)):
        (args.output / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("split", "N", "errors", "hit_at_1", "evidence_hit_at_3",
                                               "context_evidence_rate", "median_ms", "misses_at_3")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
