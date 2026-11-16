import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lexical import BM25Retriever, HybridRetriever, tokens  # noqa: E402


def chunk(cid, text, disease):
    return {"id": cid, "text": text, "source_id": "doc", "source": "x.pdf", "page": 1,
            "metadata": {"disease": disease}}


CHUNKS = [chunk("a", "Ciprofloxacin 15 mg/kg/ngày điều trị thương hàn", "thuong_han"),
          chunk("b", "Oseltamivir 75 mg điều trị cúm mùa", "cum_mua"),
          chunk("c", "Kháng sinh điều trị lỵ trực khuẩn trong 5 ngày", "ly_truc_khuan")]


class Fake:
    name = "dense"

    def __init__(self, ids):
        self.ids = ids

    def search(self, query, *, filters=None, top_k=3):
        rows = [c for c in CHUNKS if c["id"] in self.ids][:top_k]
        return [{**c, "score": 1.0, "rank": i} for i, c in enumerate(rows, 1)]


class LexicalTest(unittest.TestCase):
    def test_tokens_add_syllable_bigrams(self):
        self.assertEqual(tokens("Kháng Sinh"), ["kháng", "sinh", "kháng_sinh"])

    def test_bm25_ranks_and_ranks_are_1_based(self):
        out = BM25Retriever(CHUNKS).search("liều ciprofloxacin thương hàn", top_k=2)
        self.assertEqual(out[0]["id"], "a")
        self.assertEqual([r["rank"] for r in out], list(range(1, len(out) + 1)))

    def test_bm25_filter_restricts_candidates(self):
        out = BM25Retriever(CHUNKS).search("điều trị", filters={"disease": "cum_mua"}, top_k=3)
        self.assertEqual([c["id"] for c in out], ["b"])

    def test_no_overlap_returns_empty(self):
        self.assertEqual(BM25Retriever(CHUNKS).search("zzz"), [])

    def test_hybrid_rrf_prefers_chunks_found_by_both(self):
        h = HybridRetriever(Fake(["b", "a"]), BM25Retriever(CHUNKS), depth=3)
        out = h.search("ciprofloxacin thương hàn", top_k=2)
        self.assertEqual(out[0]["id"], "a")
        self.assertEqual(set(out[0]["sources"]), {"dense", "bm25"})


if __name__ == "__main__":
    unittest.main()
