"""Offline checks of the advanced agent loop with scripted fakes (no model quality measured)."""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "src" / "advanced")]
from agent_adv import AdvancedAgent, DOC_PREFIX  # noqa: E402

TXT = {"id": DOC_PREFIX + "p038:t01", "text": "Việt Nam xuất khẩu 562 triệu US$.", "page": 38, "score": 1.0, "rank": 1,
       "source_id": "d", "source": "x.pdf", "metadata": {"modality": "text"}}
FIG = {"id": DOC_PREFIX + "p082:fig35", "text": "Hình 35: giá điện", "page": 82, "score": 0.9, "rank": 2,
       "source_id": "d", "source": "x.pdf", "metadata": {"modality": "visual", "label": "Hình 35"}}


def call(name, **a):
    return {"id": f"c-{name}", "type": "function", "function": {"name": name, "arguments": json.dumps(a, ensure_ascii=False)}}


class FakeModel:
    model = "gpt-4.1-mini"

    def __init__(self, replies):
        self.replies, self.messages, self.tools = iter(replies), [], []

    def complete(self, messages, tools):
        self.messages.append(list(messages)); self.tools.append([t["function"]["name"] for t in tools])
        r = next(self.replies)
        msg = {"content": r} if isinstance(r, str) else {"content": None, "tool_calls": [r]}
        return {"message": msg, "usage": {"prompt_tokens": 10, "completion_tokens": 2}, "latency_ms": 1.0}


class FakeRetriever:
    def __init__(self, rows): self.rows, self.queries = rows, []
    def search(self, q, *, filters=None, top_k=4): self.queries.append(q); return self.rows[:top_k]


class FakeInspector:
    def __init__(self): self.calls = []
    def inspect(self, chunk, focus):
        self.calls.append((chunk["id"], focus))
        return {"text": "Việt Nam 8,1", "usage": {}, "latency_ms": 1, "images": 1, "cost_usd_est": 0.001}


class AdvancedAgentTest(unittest.TestCase):
    def test_seed_search_uses_question_and_short_ids_are_accepted(self):
        m = FakeModel([call("answer", answer="562 triệu USD", citations=["p038:t01"])])
        r = FakeRetriever([TXT])
        out = AdvancedAgent(m, r, None, seed_with_question=True).run("Xuất khẩu bao nhiêu?")
        self.assertEqual(r.queries, ["Xuất khẩu bao nhiêu?"])
        self.assertTrue(out["events"][0]["seeded_with_question"])
        self.assertEqual(out["status"], "answered")
        self.assertEqual(out["citations"], [DOC_PREFIX + "p038:t01"])
        self.assertIn("[p038:t01]", out["answer"])  # inline marker added by the harness
        self.assertIn("auto_inline_citation", [e["event"] for e in out["events"]])
        self.assertIn("[p038:t01]", m.messages[0][-1]["content"])  # model sees the short ID

    def test_citation_outside_context_is_repaired_then_abstains(self):
        bad = call("answer", answer="x [p999:t00]", citations=["p999:t00"])
        out = AdvancedAgent(FakeModel([call("search_documents", query="q"), bad, bad]), FakeRetriever([TXT]), None).run("q")
        self.assertEqual(out["status"], "abstain")

    def test_inspect_only_offered_for_visual_chunks_and_reading_is_kept(self):
        insp = FakeInspector()
        m = FakeModel([call("search_documents", query="giá điện"), call("inspect_figure", chunk_id="p082:fig35", focus="VN"),
                       call("answer", answer="8,1 [p082:fig35]", citations=["p082:fig35"])])
        out = AdvancedAgent(m, FakeRetriever([FIG]), insp, max_inspects=1).run("Giá điện VN?")
        self.assertNotIn("inspect_figure", m.tools[0])
        self.assertIn("inspect_figure", m.tools[1])
        self.assertEqual(insp.calls, [(FIG["id"], "VN")])
        self.assertEqual(out["n_inspects"], 1)
        self.assertIn("Đọc lại hình: Việt Nam 8,1", out["context_chunks"][0]["text"])

    def test_no_tool_call_is_nudged_once(self):
        m = FakeModel([call("search_documents", query="q"), "Câu trả lời dạng văn bản", call("abstain", reason="thiếu")])
        out = AdvancedAgent(m, FakeRetriever([TXT]), None, nudge_no_tool=True).run("q")
        self.assertEqual((out["status"], out["reason"]), ("abstain", "thiếu"))
        m2 = FakeModel([call("search_documents", query="q"), "văn bản"])
        self.assertEqual(AdvancedAgent(m2, FakeRetriever([TXT]), None).run("q")["reason"], "model không gọi tool")


if __name__ == "__main__":
    unittest.main()
