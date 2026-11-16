"""Build the advanced-challenge corpus and its dense index.

1. Merge data/advanced/extracted/text_chunks.json and visual_chunks.json into
   data/advanced/chunks.json (Day 7 chunk schema). Visual chunks longer than
   `--max-chars` are split into parts; each part repeats the figure/table
   header line so it stays findable on its own (E5 reads at most 512 tokens).
2. Embed every chunk with the same E5 model/revision/prefix as Day 6 and
   upsert into a NEW Qdrant collection (default `day7_advanced`, cosine, 1024
   dims, payload strategy="ADV"), so src/dense.py can search it unchanged.

Run with the Day 6 venv (sentence-transformers, qdrant-client):
  <day6>/starter/.venv/bin/python src/advanced/index.py
"""
import argparse
import json
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXTRACTED = ROOT / "data" / "advanced" / "extracted"
NAMESPACE = uuid.UUID("2b1f0c8e-7d4a-4c1b-9a51-6f0d2c3e4b5a")
STRATEGY = "ADV"


def split_visual(chunk: dict, max_chars: int) -> list:
    text = chunk["text"]
    if len(text) <= max_chars:
        return [chunk]
    header, _, body = text.partition("\n")
    parts, buf = [], ""
    for line in body.splitlines():
        if buf and len(buf) + len(line) + 1 > max_chars - len(header) - 1:
            parts.append(buf)
            buf = ""
        buf += line + "\n"
    if buf.strip():
        parts.append(buf)
    out = []
    for k, part in enumerate(parts):
        c = json.loads(json.dumps(chunk))
        c["id"] = f"{chunk['id']}#{k + 1}"
        c["text"] = f"{header}\n{part.strip()}"
        c["metadata"]["part"] = f"{k + 1}/{len(parts)}"
        out.append(c)
    return out


def build_corpus(max_chars: int) -> list:
    text = json.loads((EXTRACTED / "text_chunks.json").read_text(encoding="utf-8"))
    visual = json.loads((EXTRACTED / "visual_chunks.json").read_text(encoding="utf-8"))
    chunks = text + [p for v in visual for p in split_visual(v, max_chars)]
    ids = [c["id"] for c in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("chunk ID trùng")
    return chunks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-chars", type=int, default=1500)
    ap.add_argument("--collection", default="day7_advanced")
    ap.add_argument("--qdrant-url", default="http://127.0.0.1:6333")
    ap.add_argument("--skip-embed", action="store_true")
    args = ap.parse_args()

    chunks = build_corpus(args.max_chars)
    (ROOT / "data" / "advanced" / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=1),
                                                           encoding="utf-8")
    n_vis = sum(c["metadata"]["modality"] == "visual" for c in chunks)
    print(f"corpus: {len(chunks)} chunk ({len(chunks) - n_vis} text, {n_vis} visual), "
          f"{len({c['page'] for c in chunks})} trang")
    if args.skip_embed:
        return

    sys.path.insert(0, str(ROOT / "src"))
    from dense import DIMENSIONS, MODEL, REVISION, _device
    from qdrant_client import QdrantClient, models
    from sentence_transformers import SentenceTransformer

    t0 = time.perf_counter()
    model = SentenceTransformer(MODEL, revision=REVISION, device=_device())
    vectors = model.encode(["passage: " + c["text"] for c in chunks], batch_size=16, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=False)
    tok = model.tokenizer
    truncated = sum(len(tok("passage: " + c["text"])["input_ids"]) > model.max_seq_length for c in chunks)
    embed_s = time.perf_counter() - t0

    client = QdrantClient(url=args.qdrant_url, timeout=60)
    if client.collection_exists(args.collection):
        client.delete_collection(args.collection)
    client.create_collection(args.collection, vectors_config=models.VectorParams(size=DIMENSIONS,
                                                                                  distance=models.Distance.COSINE))
    for field, kind in (("strategy", models.PayloadSchemaType.KEYWORD), ("page", models.PayloadSchemaType.INTEGER),
                        ("metadata.modality", models.PayloadSchemaType.KEYWORD)):
        client.create_payload_index(args.collection, field_name=field, field_schema=kind, wait=True)
    points = [models.PointStruct(id=str(uuid.uuid5(NAMESPACE, c["id"])), vector=v.tolist(),
                                 payload={"chunk_id": c["id"], "document_id": c["source_id"], "source": c["source"],
                                          "page": c["page"], "text": c["text"], "strategy": STRATEGY,
                                          "metadata": c["metadata"]})
              for c, v in zip(chunks, vectors)]
    for i in range(0, len(points), 128):
        client.upsert(args.collection, points=points[i:i + 128], wait=True)
    count = client.count(args.collection, exact=True).count
    stats = {"collection": args.collection, "points": count, "chunks": len(chunks), "visual_chunks": n_vis,
             "truncated_over_512_tokens": truncated, "embed_seconds": round(embed_s, 1), "model": MODEL,
             "revision": REVISION, "max_chars_visual_part": args.max_chars}
    (ROOT / "data" / "advanced" / "index_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2),
                                                                encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
