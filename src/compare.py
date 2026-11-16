"""Compare agent configs on one split; with --select, apply configs/SELECTION.md and freeze the choice.

    python src/compare.py --split dev --configs baseline hybrid dense-filter hybrid-filter --select
    python src/compare.py --split test --configs baseline <selected>
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "starter" / "results" / "agent"
COMPLEXITY = {"baseline": 0, "dense-filter": 1, "hybrid": 1, "hybrid-filter": 2}


def row(name: str, split: str) -> dict:
    d = RESULTS / f"{name}-{split}"
    traces = json.loads((d / "traces.json").read_text(encoding="utf-8"))
    s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    ok = [t for t in traces if t["status"] != "error"]
    ans = [t for t in ok if t["answerable"]]
    una = [t for t in ok if not t["answerable"]]
    n_ans = sum(t["answerable"] for t in traces)
    n_una = len(traces) - n_ans
    return {
        "config": name, "errors": len(traces) - len(ok),
        "retried": [t["query_id"] for t in traces if t.get("previous_errors")],
        "grounded_proxy": sum(t["status"] == "answered" and t["context_evidence"] for t in ans), "N_ans": n_ans,
        "answered_without_evidence": [t["query_id"] for t in ans if t["status"] == "answered" and not t["context_evidence"]],
        "false_abstain": [t["query_id"] for t in ans if t["status"] == "abstain"],
        "abstained": sum(t["status"] == "abstain" for t in una), "N_una": n_una,
        "context_evidence": sum(bool(t["context_evidence"]) for t in ans),
        "median_s": s["latency_s"]["median"], "cost_usd": s["cost_usd_est_total"],
        "searches": s["searches_per_case"], "tokens": s["tokens"],
    }


def table(rows: list[dict]) -> str:
    # Denominators count every case of the split; a provider error counts as not answered here
    # (agent_metrics.py instead drops errors), so the error column is shown explicitly.
    lines = ["| Cấu hình | Lỗi nhà cung cấp | Trả lời + có evidence | Context Evidence | Từ chối đúng (auto) | Trả lời không có evidence | Từ chối sai | Lượt tìm | Trung vị (s) | Chi phí ước tính ($) |",
             "|---|---:|---:|---:|---:|---|---|---|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['config']} | {r['errors']}{' (chạy lại: ' + ', '.join(r['retried']) + ')' if r['retried'] else ''} | {r['grounded_proxy']}/{r['N_ans']} | {r['context_evidence']}/{r['N_ans']} | "
                     f"{r['abstained']}/{r['N_una']} | {', '.join(r['answered_without_evidence']) or '—'} | "
                     f"{', '.join(r['false_abstain']) or '—'} | {r['searches']} | {r['median_s']} | {r['cost_usd']} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=("dev", "test"))
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--select", action="store_true", help="Áp dụng tiêu chí chọn (chỉ cho dev)")
    args = ap.parse_args()
    rows = [row(c, args.split) for c in args.configs]
    md = table(rows)
    out = ROOT / "starter" / "results" / f"compare-{args.split}.md"
    out.write_text(md + "\n", encoding="utf-8")
    print(md)
    if args.select:
        if args.split != "dev":
            raise SystemExit("Chỉ chọn cấu hình trên dev")
        key = lambda r: (-r["grounded_proxy"], -r["abstained"], -r["context_evidence"], r["median_s"],
                         r["cost_usd"] or 0, COMPLEXITY.get(r["config"], 9))
        ranked = sorted(rows, key=key)
        selected = {"selected": ranked[0]["config"], "ranking_on_dev": [r["config"] for r in ranked],
                    "rule": "configs/SELECTION.md", "frozen_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                    "dev_rows": rows, "test_seen_before_selection": False}
        (ROOT / "configs" / "selected.json").write_text(json.dumps(selected, ensure_ascii=False, indent=2) + "\n",
                                                         encoding="utf-8")
        print("\nChọn:", selected["selected"], "| thứ hạng:", selected["ranking_on_dev"])


if __name__ == "__main__":
    main()
