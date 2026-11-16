"""Export the Day 6 index and benchmark into the Day 7 data contract.

Chunks are read from the live Qdrant collection (not from a JSONL copy), so the
lexical retrievers and the dense retriever see exactly the same corpus.

Usage (Qdrant from Day 6 must be running):
    python src/prepare_data.py --day6 ../K5-L3B-DAY06-...-DataIngestion --strategy D

Outputs in data/day6/:
    chunks_<strategy>.json  Day 7 chunk schema (id, text, source_id, source, page, metadata)
    questions.json          answerable questions with `gold` (for retrieval metrics)
    cases.json              every case with `answerable` (for agent metrics)
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "starter"))
sys.path.insert(0, str(ROOT / "src"))
from convert_day6 import convert_question  # noqa: E402
from evaluation import gold_report  # noqa: E402

OUT = ROOT / "data" / "day6"


def export_chunks(url: str, collection: str, strategy: str) -> list[dict]:
    from qdrant_client import QdrantClient, models

    client = QdrantClient(url=url, timeout=30)
    flt = models.Filter(must=[models.FieldCondition(key="strategy", match=models.MatchValue(value=strategy))])
    chunks, offset = [], None
    while True:
        points, offset = client.scroll(collection, scroll_filter=flt, limit=256, offset=offset,
                                       with_payload=True, with_vectors=False)
        for p in points:
            pl = p.payload
            chunks.append({"id": pl["chunk_id"], "text": pl["text"], "source_id": pl["document_id"],
                           "source": pl["source"], "page": pl["page"], "metadata": pl.get("metadata", {})})
        if offset is None:
            break
    if not chunks:
        raise RuntimeError(f"Không có chunk nào cho strategy={strategy!r} trong {collection}")
    # Stable order: document, page, then chunk id.
    return sorted(chunks, key=lambda c: (c["source_id"], c["page"], c["id"]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--day6", required=True, type=Path, help="Thư mục repo Day 6")
    ap.add_argument("--strategy", default="D", help="Index Day 6 cần xuất (A, B hoặc D)")
    ap.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://127.0.0.1:6333"))
    ap.add_argument("--collection", default="day6_chunks")
    args = ap.parse_args()

    chunks = export_chunks(args.qdrant_url, args.collection, args.strategy)
    raw = json.loads((args.day6 / "benchmark" / "questions.json").read_text(encoding="utf-8"))
    questions = [convert_question(q) for q in raw]
    unanswerable = json.loads((OUT / "unanswerable.json").read_text(encoding="utf-8"))

    # Every gold evidence must exist in this index. Evidence split across two chunks
    # of the gold page is kept (it can still reach the context) but reported, since
    # single-chunk Hit@k cannot count it.
    report = gold_report(chunks, questions)
    if report["missing"]:
        raise ValueError(f"Evidence không có trong index {args.strategy}: {report['missing']}")
    ids = [q["id"] for q in questions] + [u["id"] for u in unanswerable]
    if len(set(ids)) != len(ids):
        raise ValueError("ID câu hỏi bị trùng giữa câu có đáp án và câu thiếu bằng chứng")

    cases = [{"id": q["id"], "split": q["split"], "question": q["question"], "answerable": True,
              "expected_behavior": "search_and_answer", **({"filter": q["filter"]} if "filter" in q else {})}
             for q in questions]
    cases += [{k: u[k] for k in ("id", "split", "question", "answerable", "expected_behavior", "absence_reason")}
              for u in unanswerable]

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"gold_check_{args.strategy}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, data in ((f"chunks_{args.strategy}.json", chunks), ("questions.json", questions),
                       ("cases.json", cases)):
        (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def count(split, answerable):
        return sum(c["split"] == split and c["answerable"] == answerable for c in cases)
    print(f"chunks_{args.strategy}: {len(chunks)} chunk, {len({c['page'] for c in chunks})} trang")
    for split in ("dev", "test"):
        print(f"{split}: {count(split, True)} có đáp án + {count(split, False)} thiếu bằng chứng")
    print("evidence bị cắt qua 2 chunk:", report["split_across_chunks"] or "không có")


if __name__ == "__main__":
    main()
