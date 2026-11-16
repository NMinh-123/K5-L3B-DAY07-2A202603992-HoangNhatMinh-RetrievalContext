"""Offline checks of the agent loop with scripted fakes; they do not measure model quality."""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agent import Agent  # noqa: E402

C1 = {"id": "doc:p051:D01", "text": "Oseltamivir 75 mg x 2 lần/ngày trong 5 ngày.", "source_id": "doc",
      "source": "doc.pdf", "page": 51, "metadata": {"section": "CÚM"}, "score": 0.9, "rank": 1}
C2 = {**C1, "id": "doc:p052:D00", "page": 52, "text": "Đoạn khác " * 40, "score": 0.8, "rank": 2}


def call(name, **args):
    return {"id": f"call-{name}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


class FakeModel:
    model = "gpt-4.1-mini"

    def __init__(self, calls):
        self.calls, self.seen_tools, self.seen_messages = iter(calls), [], []

    def complete(self, messages, tools):
        self.seen_tools.append([t["function"]["name"] for t in tools])
        self.seen_messages.append(list(messages))
        c = next(self.calls)
        return {"message": {"content": None, "tool_calls": [c] if c else []},
                "usage": {"prompt_tokens": 100, "completion_tokens": 10}, "latency_ms": 1.0}


class FakeRetriever:
    def __init__(self, *batches):
        self.batches, self.calls = list(batches), []

    def search(self, query, *, filters=None, top_k=3):
        self.calls.append((query, filters, top_k))
        return self.batches.pop(0) if self.batches else []


class AgentTest(unittest.TestCase):
    def test_search_then_answer_with_valid_citation(self):
        m = FakeModel([call("search_documents", query="oseltamivir"),
                       call("answer", answer="75 mg x 2 lần/ngày [doc:p051:D01]", citations=["doc:p051:D01"])])
        r = FakeRetriever([C1])
        out = Agent(m, r).run("Liều oseltamivir?", filters={"disease": "cum_mua"})
        self.assertEqual(out["status"], "answered")
        self.assertEqual(out["citations"], ["doc:p051:D01"])
        self.assertEqual(r.calls, [("oseltamivir", {"disease": "cum_mua"}, 3)])
        self.assertEqual(out["n_searches"], 1)
        self.assertEqual(out["n_llm_calls"], 2)
        self.assertAlmostEqual(out["cost_usd_est"], (200 * 0.4 + 20 * 1.6) / 1e6)

    def test_citation_not_in_context_gets_one_repair_then_abstains(self):
        bad = call("answer", answer="75 mg [x:p1]", citations=["x:p1"])
        out = Agent(FakeModel([call("search_documents", query="q"), bad, bad]), FakeRetriever([C1])).run("q")
        self.assertEqual(out["status"], "abstain")
        self.assertEqual([e["event"] for e in out["events"]].count("invalid_citations"), 2)

    def test_citation_must_appear_inline(self):
        m = FakeModel([call("search_documents", query="q"),
                       call("answer", answer="75 mg", citations=["doc:p051:D01"]),
                       call("answer", answer="75 mg [doc:p051:D01]", citations=["doc:p051:D01"])])
        self.assertEqual(Agent(m, FakeRetriever([C1])).run("q")["status"], "answered")

    def test_abstain_and_missing_tool_call(self):
        out = Agent(FakeModel([call("search_documents", query="q"), call("abstain", reason="không có")]),
                    FakeRetriever([])).run("q")
        self.assertEqual((out["status"], out["reason"]), ("abstain", "không có"))
        self.assertEqual(Agent(FakeModel([None]), FakeRetriever()).run("q")["status"], "abstain")

    def test_search_limit_removes_search_tool(self):
        m = FakeModel([call("search_documents", query="a"), call("search_documents", query="b"),
                       call("abstain", reason="x")])
        out = Agent(m, FakeRetriever([C1], [C2]), max_searches=2).run("q")
        self.assertEqual(out["n_searches"], 2)
        self.assertNotIn("search_documents", m.seen_tools[2])

    def test_budget_drops_chunks_and_never_exceeds(self):
        m = FakeModel([call("search_documents", query="a"), call("abstain", reason="x")])
        out = Agent(m, FakeRetriever([C1, C2]), budget_chars=120).run("q")
        search = out["events"][1]
        self.assertEqual(search["added_ids"], ["doc:p051:D01"])
        self.assertEqual(search["dropped_for_budget"], ["doc:p052:D00"])
        self.assertLessEqual(out["context_chars"], 120)

    def test_turn_limit_abstains(self):
        m = FakeModel([call("search_documents", query=str(i)) for i in range(3)])
        out = Agent(m, FakeRetriever([C1], [C2], []), max_searches=5, max_turns=3).run("q")
        self.assertEqual(out["status"], "abstain")
        self.assertEqual(out["events"][-1]["event"], "turn_limit")


if __name__ == "__main__":
    unittest.main()
