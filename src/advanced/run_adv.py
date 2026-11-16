"""Run the advanced agent (or retrieval only) on one split; save traces, metrics and a review sheet.

  python src/advanced/run_adv.py --config configs/advanced/<name>.json --split dev
  python src/advanced/run_adv.py --retrieval-only --retriever hybrid --split dev

Gold (benchmark evidence, key facts) is read only AFTER each run to score it:
- hit: a chunk on a gold page that contains the gold quote (text evidence) or is a
  visual chunk whose label matches the gold figure/table label (visual evidence);
- Hit@1 / Evidence Hit@3 on the original question; Context Evidence = every gold
  evidence item of the question is in the final context (multi-hop needs all);
- auto_grade (draft only): all key facts appear in the answer (number separators
  normalised) and at least one citation points to a gold page; abstention for
  unanswerable. Final pass/fail is a human label in reviews.json.
"""
import argparse
import json
import re
import statistics
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "advanced"))
from agent import OpenAIChat, load_env  # noqa: E402

DATA = ROOT / "data" / "advanced"
RESULTS = ROOT / "starter" / "results" / "advanced"
LABEL = re.compile(r"(Hình|Bảng|Hộp)\s*([A-Z]?\d+(?:\.\d+)?)")


def norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s or "").lower()
    s = re.sub(r"(?<=\d)[.,](?=\d)", "·", s)          # 4.4 == 4,4 ; 12.772 == 12772 handled below
    s = re.sub(r"(?<=\d)·(?=\d{3}\b)", "", s)          # thousands separator
    s = re.sub(r"-\s+", "-", s)
    return " ".join(s.replace("−", "-").split())


def label_core(text: str | None) -> str | None:
    m = LABEL.search(text or "")
    return f"{m.group(1)} {m.group(2)}" if m else None


def ev_hit(chunk: dict, ev: dict) -> bool:
    if chunk.get("page") != ev["page"]:
        return False
    if "quote" in ev:
        return norm(ev["quote"]) in norm(chunk.get("text", ""))
    md = chunk.get("metadata") or {}
    want = label_core(ev.get("label"))
    return md.get("modality") == "visual" and (want is None or label_core(md.get("label")) == want
                                                or label_core(md.get("title")) == want)


def build_retriever(name: str):
    chunks = json.loads((DATA / "chunks.json").read_text(encoding="utf-8"))
    from lexical import BM25Retriever, HybridRetriever
    if name == "bm25":
        return BM25Retriever(chunks)
    from dense import DenseRetriever
    dense = DenseRetriever(strategy="ADV", collection="day7_advanced")
    return dense if name == "dense" else HybridRetriever(dense, BM25Retriever(chunks))


def retrieval_only(args, cases):
    r = build_retriever(args.retriever)
    rows = []
    for q in cases:
        if not q["answerable"]:
            continue
        t0 = time.perf_counter()
        res = r.search(q["question"], filters=None, top_k=max(3, args.top_k))
        ms = (time.perf_counter() - t0) * 1000
        evs = q["evidence"]
        hit = [any(ev_hit(c, ev) for ev in evs) for c in res]
        ctx = res[:args.top_k]
        rows.append({"id": q["id"], "group": q["group"], "hit1": bool(hit[:1] and hit[0]), "hit3": any(hit[:3]),
                     "context_all": all(any(ev_hit(c, ev) for c in ctx) for ev in evs), "ms": round(ms, 1),
                     "top": [(c["id"].split(":", 1)[1], c["page"]) for c in res[:args.top_k]]})
    n = len(rows)
    summ = {"retriever": args.retriever, "split": args.split, "top_k_context": args.top_k, "N": n,
            "hit_at_1": f"{sum(x['hit1'] for x in rows)}/{n}", "evidence_hit_at_3": f"{sum(x['hit3'] for x in rows)}/{n}",
            "context_evidence_all": f"{sum(x['context_all'] for x in rows)}/{n}",
            "median_ms": statistics.median(x["ms"] for x in rows),
            "by_group": {g: f"{sum(x['hit3'] for x in rows if x['group'] == g)}/{sum(x['group'] == g for x in rows)}"
                         for g in sorted({x['group'] for x in rows})},
            "misses_at_3": [x["id"] for x in rows if not x["hit3"]]}
    out = RESULTS / "retrieval" / f"{args.split}-{args.retriever}-k{args.top_k}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "traces.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False))


def auto_grade(q: dict, t: dict) -> dict:
    if t["status"] == "error":
        return {"pass": False, "reason": "lỗi hệ thống"}
    if not q["answerable"]:
        ok = t["status"] == "abstain"
        return {"pass": ok, "reason": "từ chối" if ok else "trả lời câu thiếu bằng chứng"}
    if t["status"] != "answered":
        return {"pass": False, "reason": "từ chối câu có đáp án"}
    a = norm(t["answer"])
    # A key fact may list equivalent spellings separated by "|" (see data/advanced/BENCHMARK_CHANGES.md).
    def variants(alt: str) -> list:
        # "-8,2%" is also written "giảm 8,2%" / "âm 8,2%" in Vietnamese answers
        return [alt] + ([f"giảm {alt[1:]}", f"âm {alt[1:]}"] if alt.startswith("-") else [])
    missing = [k for k in q["key_facts"] if not any(norm(v) in a for alt in k.split("|") for v in variants(alt))]
    gold_pages = {ev["page"] for ev in q["evidence"]}
    cite_pages = {t["context_pages"].get(c) for c in t["citations"]}
    cites_ok = gold_pages <= cite_pages
    return {"pass": not missing and cites_ok, "missing_key_facts": missing, "citation_pages": sorted(p for p in cite_pages if p),
            "gold_pages": sorted(gold_pages), "citations_cover_gold_pages": cites_ok}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path)
    ap.add_argument("--split", required=True, choices=("dev", "test"))
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--retriever", default="hybrid", choices=("bm25", "dense", "hybrid"))
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--retry-errors", action="store_true")
    ap.add_argument("--regrade", action="store_true", help="Chấm lại auto_grade từ traces đã lưu, không gọi model")
    args = ap.parse_args()
    bench = json.loads((DATA / "benchmark.json").read_text(encoding="utf-8"))
    if args.regrade:
        byid = {q["id"]: q for q in bench}
        d = RESULTS / "agent" / f"{json.loads(args.config.read_text(encoding='utf-8'))['name']}-{args.split}"
        traces = json.loads((d / "traces.json").read_text(encoding="utf-8"))
        for t in traces:
            t["auto_grade"] = auto_grade(byid[t["query_id"]], t)
        summ = json.loads((d / "summary.json").read_text(encoding="utf-8"))
        groups = sorted({t["group"] for t in traces})
        summ["auto_pass"] = f"{sum(t['auto_grade']['pass'] for t in traces)}/{len(traces)}"
        summ["auto_pass_by_group"] = {g: f"{sum(t['auto_grade']['pass'] for t in traces if t['group'] == g)}/"
                                         f"{sum(t['group'] == g for t in traces)}" for g in groups}
        (d / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=1), encoding="utf-8")
        (d / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2), encoding="utf-8")
        print(d.name, summ["auto_pass"], summ["auto_pass_by_group"])
        return
    cases = [q for q in bench if q["split"] == args.split and (not args.only or q["id"] in args.only)]
    if args.retrieval_only:
        return retrieval_only(args, cases)

    load_env(ROOT / ".env")
    from agent_adv import AdvancedAgent, FigureInspector
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    out = RESULTS / "agent" / f"{cfg['name']}-{args.split}"
    out.mkdir(parents=True, exist_ok=True)
    # The proxy endpoint intermittently answers 503 "no available channel": retry 5xx/429 up to 5 times with
    # 5-40 s waits; a call that still fails is recorded as an error, never replaced.
    model = OpenAIChat(model=cfg["model"], temperature=cfg.get("temperature", 0.0), timeout=120,
                       retries=4, backoff=(5, 10, 20, 40))
    inspector = FigureInspector(model=cfg["inspect_model"]) if cfg.get("inspect_model") else None
    agent = AdvancedAgent(model, build_retriever(cfg["retriever"]), inspector, top_k=cfg["top_k"],
                          max_searches=cfg["max_searches"], max_inspects=cfg.get("max_inspects", 0),
                          max_turns=cfg["max_turns"], budget_chars=cfg["budget_chars"],
                          seed_with_question=cfg.get("seed_with_question", False),
                          nudge_no_tool=cfg.get("nudge_no_tool", False))
    previous = {}
    if args.retry_errors and (out / "traces.json").exists():
        previous = {t["query_id"]: t for t in json.loads((out / "traces.json").read_text(encoding="utf-8"))}
    traces = []
    for q in cases:
        old = previous.get(q["id"])
        if old and old["status"] != "error":
            traces.append(old)
            continue
        t0 = time.perf_counter()
        try:
            r, err = agent.run(q["question"]), None   # only the question text reaches the agent
        except Exception as exc:
            r, err = {"status": "error", "answer": None, "citations": [], "events": [], "context_ids": [],
                      "context_chunks": [], "context_chars": 0, "n_searches": 0, "n_inspects": 0, "n_llm_calls": 0,
                      "usage": {}, "cost_usd_est": None, "reason": None}, repr(exc)
        ctx = r["context_chunks"]
        t = {"query_id": q["id"], "split": q["split"], "group": q["group"], "question": q["question"],
             "answerable": q["answerable"], "config": cfg["name"], "status": r["status"], "answer": r["answer"],
             "citations": r["citations"], "reason": r.get("reason"), "error": err,
             "context_ids": r["context_ids"], "context_pages": {c["id"]: c["page"] for c in ctx},
             "context_chars": r["context_chars"], "n_searches": r["n_searches"], "n_inspects": r["n_inspects"],
             "n_llm_calls": r["n_llm_calls"], "usage": r["usage"], "cost_usd_est": r["cost_usd_est"],
             "latency_s": round(time.perf_counter() - t0, 2), "events": r["events"],
             "previous_errors": (old.get("previous_errors", []) + [old["error"]]) if old else []}
        if q["answerable"] and r["status"] != "error":
            t["context_evidence_all"] = all(any(ev_hit(c, ev) for c in ctx) for ev in q["evidence"])
        t["auto_grade"] = auto_grade(q, t)
        traces.append(t)
        print(f"{q['id']} {q['group']:12s} {r['status']:8s} s={r['n_searches']} i={r['n_inspects']} "
              f"auto={'ĐẠT' if t['auto_grade']['pass'] else 'trượt'} {t['latency_s']}s" + (f" ERR {err[:120]}" if err else ""),
              flush=True)

    ok = [t for t in traces if t["status"] != "error"]
    groups = sorted({t["group"] for t in traces})
    costs = [t["cost_usd_est"] for t in ok if t["cost_usd_est"] is not None]
    summary = {"config": cfg, "split": args.split, "cases": len(traces), "errors": len(traces) - len(ok),
               "auto_pass": f"{sum(t['auto_grade']['pass'] for t in traces)}/{len(traces)}",
               "auto_pass_by_group": {g: f"{sum(t['auto_grade']['pass'] for t in traces if t['group'] == g)}/"
                                         f"{sum(t['group'] == g for t in traces)}" for g in groups},
               "context_evidence_all": f"{sum(bool(t.get('context_evidence_all')) for t in ok if t['answerable'])}/"
                                       f"{sum(t['answerable'] for t in ok)}",
               "status": dict(Counter(t["status"] for t in traces)),
               "searches": dict(sorted(Counter(t["n_searches"] for t in ok).items())),
               "inspects": dict(sorted(Counter(t["n_inspects"] for t in ok).items())),
               "latency_s_median": statistics.median(t["latency_s"] for t in ok) if ok else None,
               "cost_usd_est_total": round(sum(costs), 4) if costs else None,
               "note": "auto_pass là chấm nháp tự động; kết quả chính thức là nhãn người chấm trong reviews.json."}
    (out / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("auto_pass", "auto_pass_by_group", "context_evidence_all", "errors",
                                               "searches", "inspects", "latency_s_median", "cost_usd_est_total")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
