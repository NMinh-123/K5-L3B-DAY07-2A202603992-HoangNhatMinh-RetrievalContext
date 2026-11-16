"""Run the agent with one config on one split; save traces, automatic summary and a review template.

    python src/run_agent.py --config configs/baseline.json --split dev

Outputs in starter/results/agent/<config>-<split>/:
    traces.json            one record per case (agent_metrics.py compatible + full trace)
    summary.json           automatic numbers (tool use, statuses, context evidence, time, tokens, cost)
    reviews_template.json  fill by hand, save as reviews.json, then run starter/agent_metrics.py

Gold labels are read only AFTER each run to compute Context Evidence Rate;
the agent receives the question text and the user-provided filter only.
"""
import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from agent import Agent, OpenAIChat, load_env  # noqa: E402
from eval_retrieval import build  # noqa: E402
from evaluation import context_has_evidence  # noqa: E402

DATA = ROOT / "data" / "day6"


def frac(n, d):
    return {"n": n, "N": d, "rate": round(n / d, 3) if d else None}


def main() -> None:
    ap = argparse.ArgumentParser(description="Chạy agent trên một split")
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--split", required=True, choices=("dev", "test"))
    ap.add_argument("--only", nargs="*", help="Chỉ chạy các case ID này (để debug)")
    ap.add_argument("--output", type=Path)
    ap.add_argument("--retry-errors", action="store_true",
                    help="Giữ các câu đã chạy xong trong traces.json cũ, chỉ chạy lại câu có status error")
    args = ap.parse_args()

    load_env(ROOT / ".env")
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    out = args.output or ROOT / "starter" / "results" / "agent" / f"{cfg['name']}-{args.split}"
    cases = [c for c in json.loads((DATA / "cases.json").read_text(encoding="utf-8")) if c["split"] == args.split]
    if args.only:
        cases = [c for c in cases if c["id"] in set(args.only)]
    gold = {q["id"]: q["gold"] for q in json.loads((DATA / "questions.json").read_text(encoding="utf-8"))}

    model = OpenAIChat(model=cfg.get("model"), temperature=cfg.get("temperature", 0.0))
    retriever = build(cfg["retriever"], cfg.get("index", "D"))
    agent = Agent(model, retriever, top_k=cfg["top_k"], max_searches=cfg["max_searches"],
                  max_turns=cfg["max_turns"], budget_chars=cfg["budget_chars"])

    previous = {}
    if args.retry_errors:
        previous = {t["query_id"]: t for t in json.loads((out / "traces.json").read_text(encoding="utf-8"))}

    traces = []
    for case in cases:
        old = previous.get(case["id"])
        if old and old["status"] != "error":
            traces.append(old)
            continue
        filters = case.get("filter") if cfg.get("use_filter") else None
        t0 = time.perf_counter()
        try:
            r = agent.run(case["question"], filters=filters)
            error = None
        except Exception as exc:  # recorded as error, excluded from quality metrics, never faked
            r, error = {"status": "error", "answer": None, "citations": [], "events": [], "context_ids": [],
                        "context_chunks": [], "context_chars": 0, "n_searches": 0, "n_llm_calls": 0,
                        "usage": {}, "cost_usd_est": None, "reason": None}, repr(exc)
        elapsed = time.perf_counter() - t0
        ctx_ev = (context_has_evidence(r["context_chunks"], gold[case["id"]])
                  if case["answerable"] and r["status"] != "error" else None)
        traces.append({"query_id": case["id"], "split": case["split"], "question": case["question"],
                       "answerable": case["answerable"], "config": cfg["name"], "filter": filters,
                       "status": r["status"], "answer": r["answer"], "citations": r["citations"],
                       "reason": r.get("reason"), "error": error, "context_evidence": ctx_ev,
                       "context_ids": r["context_ids"], "context_chars": r["context_chars"],
                       "context": [{"id": c["id"], "page": c["page"], "text": c["text"]} for c in r["context_chunks"]],
                       "n_searches": r["n_searches"], "n_llm_calls": r["n_llm_calls"], "usage": r["usage"],
                       "cost_usd_est": r["cost_usd_est"], "latency_s": round(elapsed, 2), "events": r["events"],
                       # Provider errors of earlier attempts stay visible after a retry.
                       "previous_errors": (old.get("previous_errors", []) + [old["error"]]) if old else []})
        mark = {"answered": "A", "abstain": "-", "error": "E"}[r["status"]]
        print(f"{case['id']} {mark} searches={r['n_searches']} ctx_ev={ctx_ev} {elapsed:.1f}s"
              + (f" ERROR {error}" if error else ""), flush=True)

    ok = [t for t in traces if t["status"] != "error"]
    ans = [t for t in ok if t["answerable"]]
    una = [t for t in ok if not t["answerable"]]
    costs = [t["cost_usd_est"] for t in ok if t["cost_usd_est"] is not None]
    summary = {
        "config": cfg, "model": model.describe(), "retriever": retriever.describe(), "split": args.split,
        "cases": len(traces), "errors": len(traces) - len(ok),
        "status": dict(Counter(t["status"] for t in traces)),
        "tool_use": frac(sum(t["n_searches"] > 0 for t in ok), len(ok)),
        "searches_per_case": dict(sorted(Counter(t["n_searches"] for t in ok).items())),
        "llm_calls_per_case": dict(sorted(Counter(t["n_llm_calls"] for t in ok).items())),
        "context_evidence_rate": frac(sum(bool(t["context_evidence"]) for t in ans), len(ans)),
        "answered_answerable": frac(sum(t["status"] == "answered" for t in ans), len(ans)),
        "abstained_unanswerable_auto": frac(sum(t["status"] == "abstain" for t in una), len(una)),
        "avg_context_chars": round(statistics.mean(t["context_chars"] for t in ok), 1) if ok else 0,
        "latency_s": {"median": round(statistics.median(t["latency_s"] for t in ok), 2),
                      "total": round(sum(t["latency_s"] for t in ok), 1)} if ok else None,
        "tokens": {"prompt": sum(t["usage"].get("prompt_tokens", 0) for t in ok),
                   "completion": sum(t["usage"].get("completion_tokens", 0) for t in ok)},
        "cost_usd_est_total": round(sum(costs), 5) if costs else None,
        "cost_note": "Ước tính theo giá niêm yết OpenAI; giá thực tế qua endpoint proxy có thể khác.",
        "note": "Grounded answer, citation support và abstention đúng cần chấm tay (reviews.json).",
    }
    template = [{"query_id": t["query_id"], "citation_support": [None] * len(t["citations"]),
                 **({"answer_grade": 0 if t["status"] == "abstain" else None} if t["answerable"]
                    else {"abstention_correct": None}), "note": ""}
                for t in ok]
    out.mkdir(parents=True, exist_ok=True)
    for name, obj in (("traces.json", traces), ("summary.json", summary), ("reviews_template.json", template)):
        (out / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("status", "errors", "tool_use", "searches_per_case",
                                               "context_evidence_rate", "answered_answerable",
                                               "abstained_unanswerable_auto", "latency_s", "tokens",
                                               "cost_usd_est_total")}, ensure_ascii=False))
    print(f"Đã lưu: {out}")


if __name__ == "__main__":
    main()
