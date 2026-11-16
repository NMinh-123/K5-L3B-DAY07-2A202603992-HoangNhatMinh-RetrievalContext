"""Collect project results into dashboard/data.json for the local dashboard.

    python3 dashboard/build_data.py
Missing result folders are reported as "chưa chạy" instead of failing.
"""
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "starter" / "results"


def load(path, default=None):
    p = ROOT / path if not isinstance(path, Path) else path
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def frac(s):
    """'12/19 (63%)' or {'n':..,'N':..} -> [n, N]."""
    if isinstance(s, dict):
        return [s.get("n"), s.get("N")]
    m = re.match(r"\s*(\d+)\s*/\s*(\d+)", str(s or ""))
    return [int(m.group(1)), int(m.group(2))] if m else None


def basic():
    retrieval = []
    for name, label in (("tfidf", "TF-IDF"), ("bm25", "BM25"), ("dense", "Dense"), ("hybrid", "Hybrid RRF"),
                        ("bm25-filter", "BM25 + filter"), ("dense-filter", "Dense + filter"),
                        ("hybrid-filter", "Hybrid + filter")):
        s = load(RES / "retrieval" / f"dev-{name}" / "summary.json")
        if s:
            retrieval.append({"id": name, "label": label, "hit1": frac(s["hit_at_1"]),
                              "hit3": frac(s["evidence_hit_at_3"]), "ctx": frac(s["context_evidence_rate"]),
                              "ms": s["median_ms"]})
    budget = []
    for b, d in ((800, "dev-dense-budget800"), (2000, "dev-dense-budget2000"), (4000, "dev-dense")):
        s = load(RES / "retrieval" / d / "summary.json")
        if s:
            budget.append({"budget": b, "ctx": frac(s["context_evidence_rate"]), "hit3": frac(s["evidence_hit_at_3"])})
    agents = {}
    for split in ("dev", "test"):
        rows = []
        for cfg in ("baseline-v0", "baseline", "hybrid", "dense-filter", "hybrid-filter"):
            d = RES / "agent" / f"{cfg}-{split}"
            s, tr = load(d / "summary.json"), load(d / "traces.json")
            if not s or not tr:
                continue
            ans = [t for t in tr if t["answerable"] and t["status"] != "error"]
            row = {"config": cfg, "cases": len(tr), "errors": sum(t["status"] == "error" for t in tr),
                   "answered_with_evidence": [sum(t["status"] == "answered" and bool(t["context_evidence"]) for t in ans),
                                              sum(t["answerable"] for t in tr)],
                   "ctx": [sum(bool(t["context_evidence"]) for t in ans), sum(t["answerable"] for t in tr)],
                   "abstain_ok": [sum(t["status"] == "abstain" for t in tr if not t["answerable"]),
                                  sum(not t["answerable"] for t in tr)],
                   "median_s": s["latency_s"]["median"], "cost": s["cost_usd_est_total"],
                   "tokens": s["tokens"]}
            m = load(d / "agent_metrics.json")
            if m:
                g = m["grounded_answer"]
                row["grounded"] = {"0": g["0"], "1": g["1"], "2": g["2"], "N": g["N"], "mean": g["mean"]}
                row["citation"] = [m["citation_validity"]["n"], m["citation_validity"]["N"]]
            ctxr = load(d / "context_reviews.json")
            if ctxr:
                row["context_grades"] = dict(Counter(str(r["context_grade"]) for r in ctxr))
            if split == "test":
                rv = {r["query_id"]: r for r in load(d / "reviews.json", [])}
                row["cases_detail"] = [{"id": t["query_id"], "q": t["question"], "status": t["status"],
                                        "answer": t["answer"] or (t.get("error") or "")[:160],
                                        "citations": t["citations"],
                                        "grade": rv.get(t["query_id"], {}).get("answer_grade"),
                                        "abst": rv.get(t["query_id"], {}).get("abstention_correct"),
                                        "note": rv.get(t["query_id"], {}).get("note", "")} for t in tr]
            rows.append(row)
        agents[split] = rows
    sel = load("configs/selected.json", {})
    cases = load("data/day6/cases.json", [])
    return {"retrieval_dev": retrieval, "budget": budget, "agents": agents, "selected": sel.get("selected"),
            "benchmark": {s: {"answerable": sum(c["split"] == s and c["answerable"] for c in cases),
                              "unanswerable": sum(c["split"] == s and not c["answerable"] for c in cases)}
                          for s in ("dev", "test")},
            "corpus": {"name": "QĐ 5642/QĐ-BYT (2015) – Hướng dẫn chẩn đoán và điều trị bệnh truyền nhiễm",
                       "pages": 66, "chunks": len(load("data/day6/chunks_D.json", [])), "index": "Day 6 index D (semantic)"}}


def advanced():
    stats = load("data/advanced/index_stats.json", {})
    visual = load("data/advanced/extracted/visual_chunks.json", [])
    bench = load("data/advanced/benchmark.json", [])
    kinds = Counter(v["metadata"]["kind"] for v in visual)
    comp = {}
    for q in bench:
        comp.setdefault(q["group"], {"dev": 0, "test": 0})[q["split"]] += 1
    retr = []
    for r in ("bm25", "dense", "hybrid"):
        s = load(RES / "advanced" / "retrieval" / f"dev-{r}-k4" / "summary.json")
        if s:
            retr.append({"id": r, "label": {"bm25": "BM25", "dense": "Dense", "hybrid": "Hybrid RRF"}[r],
                         "hit1": frac(s["hit_at_1"]), "hit3": frac(s["evidence_hit_at_3"]),
                         "ctx": frac(s["context_evidence_all"]), "ms": s["median_ms"]})
    agents = {}
    for split in ("dev", "test"):
        rows = []
        for d in sorted((RES / "advanced" / "agent").glob(f"*-{split}")) if (RES / "advanced" / "agent").exists() else []:
            s, tr = load(d / "summary.json"), load(d / "traces.json")
            if not s or not tr:
                continue
            reviews = {r["query_id"]: r for r in load(d / "reviews.json", [])}
            rows.append({"config": s["config"]["name"], "description": s["config"].get("description", ""),
                         "human_pass": frac(s.get("human_pass")),
                         "human_by_group": {g: frac(v) for g, v in (s.get("human_pass_by_group") or {}).items()},
                         "auto_pass": frac(s["auto_pass"]), "by_group": {g: frac(v) for g, v in s["auto_pass_by_group"].items()},
                         "ctx": frac(s["context_evidence_all"]), "errors": s["errors"], "median_s": s["latency_s_median"],
                         "cost": s["cost_usd_est_total"], "inspects": s.get("inspects"),
                         "cases_detail": [{"id": t["query_id"], "group": t["group"], "q": t["question"], "status": t["status"],
                                           "answer": t["answer"] or (t.get("error") or "")[:160], "citations": t["citations"],
                                           "pass": reviews[t["query_id"]]["pass"] if t["query_id"] in reviews else t["auto_grade"]["pass"],
                                           "reviewed": t["query_id"] in reviews,
                                           "review": reviews.get(t["query_id"], {}).get("reason", ""),
                                           "missing": t["auto_grade"].get("missing_key_facts", []),
                                           "inspects": t.get("n_inspects", 0)} for t in tr]})
        agents[split] = rows
    total_pages = 117
    return {"document": {"title": "Việt Nam 2045: Nâng cao vị thế thương mại trong một thế giới đang thay đổi",
                         "publisher": "Ngân hàng Thế giới (2024)", "license": "CC BY-NC 3.0 IGO", "pages": total_pages,
                         "url": "https://openknowledge.worldbank.org/entities/publication/c3d2e6ed-e439-4091-be7e-237c99f864ed/full"},
            "index": stats, "visual_kinds": dict(kinds.most_common()),
            "number_match_flagged": sum(1 for v in visual if (v["metadata"]["number_match"]["rate"] is not None
                                                              and v["metadata"]["number_match"]["rate"] < 0.8)),
            "benchmark_composition": comp, "retrieval_dev": retr, "agents": agents,
            "selected": load("configs/advanced/selected.json", {}).get("selected")}


if __name__ == "__main__":
    data = {"basic": basic(), "advanced": advanced()}
    out = ROOT / "dashboard" / "data.json"
    out.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Đã ghi {out.relative_to(ROOT)}: {len(data['basic']['retrieval_dev'])} retriever cơ bản, "
          f"{len(data['advanced']['agents'].get('dev', []))} cấu hình agent nâng cao trên dev")
