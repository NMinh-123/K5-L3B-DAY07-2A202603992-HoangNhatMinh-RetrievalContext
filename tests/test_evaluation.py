import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluation import context_has_evidence, gold_report, is_hit, norm  # noqa: E402

GOLD = [{"source_id": "doc", "page": 5, "evidence": "dùng thuốc trong 10-14 ngày"}]


def chunk(cid, text, page=5, source_id="doc"):
    return {"id": cid, "text": text, "page": page, "source_id": source_id, "source": "x.pdf", "metadata": {}}


class EvaluationTest(unittest.TestCase):
    def test_norm_collapses_line_breaks_and_case(self):
        self.assertEqual(norm("Dùng  thuốc\ntrong"), "dùng thuốc trong")

    def test_hit_requires_page_source_and_full_evidence(self):
        self.assertTrue(is_hit(chunk("doc:p005:D00", "... Dùng thuốc\ntrong 10-14 ngày."), GOLD))
        self.assertFalse(is_hit(chunk("doc:p006:D00", "dùng thuốc trong 10-14 ngày", page=6), GOLD))
        self.assertFalse(is_hit(chunk("doc:p005:D00", "dùng thuốc trong 10"), GOLD))

    def test_context_accepts_evidence_split_over_two_chunks_of_gold_page(self):
        parts = [chunk("doc:p005:D02", "trong 10-14 ngày."), chunk("doc:p005:D01", "Bắt đầu dùng thuốc")]
        self.assertFalse(any(is_hit(c, GOLD) for c in parts))
        self.assertTrue(context_has_evidence(parts, GOLD))
        self.assertFalse(context_has_evidence(parts[:1], GOLD))

    def test_gold_report_classifies_questions(self):
        chunks = [chunk("doc:p005:D01", "dùng thuốc"), chunk("doc:p005:D02", "trong 10-14 ngày")]
        report = gold_report(chunks, [{"id": "q1", "gold": GOLD},
                                      {"id": "q2", "gold": [{**GOLD[0], "evidence": "không có"}]}])
        self.assertEqual(report, {"single_chunk": [], "split_across_chunks": ["q1"], "missing": ["q2"]})


if __name__ == "__main__":
    unittest.main()
