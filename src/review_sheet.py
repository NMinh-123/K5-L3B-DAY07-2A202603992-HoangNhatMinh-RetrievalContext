"""Build a Markdown sheet for human review of agent answers, citations and context.

For each case: question, expected answer (or expected abstention and why),
agent answer, full text of every cited chunk, and the current labels from
reviews.json / context_reviews.json. Edit the JSON files, not the sheet.

    python src/review_sheet.py --split test --configs baseline hybrid-filter
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "day6"
RESULTS = ROOT / "starter" / "results" / "agent"


def load(path: Path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=("dev", "test"))
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--max-chars", type=int, default=900, help="Cắt đoạn trích dài hơn mức này")
    args = ap.parse_args()

    gold = {q["id"]: q for q in load(DATA / "questions.json")}
    cases = {c["id"]: c for c in load(DATA / "cases.json")}
    out = [f"# Duyệt nhãn {args.split}", "",
           "Sửa nhãn trong `starter/results/agent/<cấu hình>-" + args.split + "/reviews.json` và `context_reviews.json`, "
           "rồi chạy lại `src/review_sheet.py` và `starter/agent_metrics.py`.", "",
           "- `answer_grade`: 0 = sai/thiếu evidence hoặc từ chối câu có đáp án, 1 = đúng một phần, 2 = đúng và đủ.",
           "- `citation_support`: mỗi citation một true/false, true khi đoạn đó hỗ trợ mọi điều kiện của ý được gắn.",
           "- `context_grade`: 0 = context không có evidence, 1 = có nhưng thiếu điều kiện hoặc nguồn, 2 = đủ evidence và nguồn.",
           "- `abstention_correct`: true khi câu thiếu bằng chứng được từ chối và không có khẳng định vô căn cứ.", ""]
    for cfg in args.configs:
        d = RESULTS / f"{cfg}-{args.split}"
        traces = load(d / "traces.json")
        reviews = {r["query_id"]: r for r in load(d / "reviews.json", [])}
        ctx_reviews = {r["query_id"]: r for r in load(d / "context_reviews.json", [])}
        out += [f"## {cfg}", ""]
        for t in traces:
            i = t["query_id"]
            out += [f"### {i} · {t['status']}", f"**Hỏi:** {t['question']}  "]
            if i in gold:
                g = gold[i]
                out.append(f"**Đáp án chuẩn:** {g['expected_answer']} (trang {', '.join(str(x['page']) for x in g['gold'])})  ")
            else:
                out.append(f"**Mong đợi:** từ chối. {cases[i].get('absence_reason', '')}  ")
            if t["status"] == "error":
                out += [f"**Lỗi nhà cung cấp, không chấm:** `{(t['error'] or '')[:200]}`", ""]
                continue
            queries = [e["query"] for e in t["events"] if e["event"] == "search"]
            out.append(f"**Truy vấn agent:** {' · '.join(queries) or '—'}  ")
            out.append(f"**Context cuối:** {', '.join(c.split(':', 1)[1] for c in t['context_ids']) or '—'}  ")
            out.append(f"**Agent:** {t['answer']}  ")
            if t.get("reason"):
                out.append(f"**Lý do từ chối:** {t['reason']}  ")
            ctx = {c["id"]: c for c in t["context"]}
            for cid in t["citations"]:
                text = " ".join(ctx[cid]["text"].split())
                cut = "…" if len(text) > args.max_chars else ""
                out += ["", f"> `{cid}` (trang {ctx[cid]['page']}): {text[:args.max_chars]}{cut}"]
            r, cr = reviews.get(i, {}), ctx_reviews.get(i)
            label = {k: r[k] for k in ("answer_grade", "abstention_correct", "citation_support") if k in r}
            out += ["", f"**Nhãn câu trả lời:** `{json.dumps(label, ensure_ascii=False)}` — {r.get('note', 'chưa chấm')}  "]
            if cr:
                out.append(f"**Nhãn context:** `{cr['context_grade']}` — {cr['note']}")
            out.append("")
    path = ROOT / "starter" / "results" / "agent" / f"review_sheet_{args.split}.md"
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"Đã ghi {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
