"""Extract text and visual content (charts, diagrams, maps, tables) from the advanced PDF.

Text: PyMuPDF text blocks, page header/footer removed, grouped into paragraph
chunks (<= ~1200 chars) that keep page number and the nearest section heading.

Visual: every page whose vector drawings or images form a large region is
rendered (PNG, kept for verification) and sent to a vision model that
transcribes each figure/table on the page into structured text: label (e.g.
"Hình 23"), kind, title, axes, units, legend, every printed value, and tables
as Markdown. Vector charts keep their labels in the PDF text layer, so every
number the model reports is checked against the page text (`number_match`);
low scores are flagged for manual review.

Outputs (data/advanced/extracted/):
  pages/pNNN.png          rendered pages sent to the vision model
  vision/pNNN.json        raw vision answer per page (cache: rerun skips existing)
  text_chunks.json        text chunks
  visual_chunks.json      one chunk per figure/table with page, label, kind, region, number_match

Usage (Day 7 .venv with pymupdf):
  .venv/bin/python src/advanced/extract.py --pdf data/advanced/raw/wb_vn2045_trading_up_vi.pdf [--pages 53,59,105]
"""
import argparse
import base64
import json
import os
import re
import time
import unicodedata
import urllib.request
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "advanced" / "extracted"
DOC_ID = "wb-vn2045-trading-up-vi"
SKIP_PAGES = {1, 116, 117}           # covers: decorative only
HEADER_Y, FOOTER_MARGIN = 95, 60     # pt; coloured band at top, page number at bottom
CAPTION = re.compile(r"^(Hình|Bảng|Hộp)\s+([A-Z]?\d+(?:\.\d+)*[a-z]?)")
HEADING = re.compile(r"^(\d+\.\s+[A-ZĐÂĂÊÔƠƯ]|Phụ lục|PHỤ LỤC|Chương|CHƯƠNG|[A-ZĐ][A-ZĐÀ-Ỹ ,\-]{12,}$)")

VISION_PROMPT = """Bạn trích xuất nội dung trực quan từ một trang của báo cáo Ngân hàng Thế giới "Việt Nam 2045: Nâng cao vị thế thương mại trong một thế giới đang thay đổi" (bản tiếng Việt).

Với MỖI hình (biểu đồ, sơ đồ, bản đồ, infographic) và MỖI bảng trên trang, tạo một phần tử. Bỏ qua dải trang trí đầu trang, số trang, logo và đoạn văn thường.

Trả về JSON: {"items": [{
  "label": "số hiệu đúng như in, ví dụ 'Hình 20', 'Bảng A2.1', 'Hình B1.1'; null nếu không có",
  "kind": "bar_chart | line_chart | area_chart | scatter | bubble_chart | stacked_bar | map | diagram | table | infographic | other",
  "title": "tiêu đề đầy đủ như in",
  "panels": "liệt kê các biểu đồ con A/B nếu có, kèm tiêu đề con",
  "content": "mô tả chi tiết bằng tiếng Việt, đủ để trả lời câu hỏi mà không cần nhìn hình",
  "source_note": "dòng Nguồn/Ghi chú/Dữ liệu dưới hình, chép nguyên văn"
}]}

Quy tắc cho "content":
- Biểu đồ: tên trục và đơn vị; chú giải (màu -> chuỗi); MỌI giá trị có in số trên hình, dạng "chuỗi – hạng mục/năm: giá trị đơn vị"; xu hướng chính. Giá trị không in số mà chỉ đọc từ trục thì ghi "khoảng X (đọc từ trục)".
- Sơ đồ, bản đồ: mọi nút, nhãn, nhóm, mũi tên, thứ tự bước, quốc gia/khu vực và quan hệ giữa chúng (ví dụ nước nào thuộc nhóm/hiệp định nào).
- Bảng: chép đầy đủ dạng bảng Markdown, đủ mọi hàng và cột, giữ nguyên số như in (dấu phẩy thập phân, dấu chấm hàng nghìn, %, dấu âm).
- Cột chồng: số ở đỉnh cột thường là TỔNG; gán từng số vào đúng đoạn màu theo vị trí, không gán tổng vào một chuỗi.
- Đường ngang ngưỡng/mục tiêu: lấy giá trị từ ghi chú hoặc nhãn nếu có; nếu không có thì ghi "không in số".
- Nếu không chắc một số thuộc chuỗi nào, ghi rõ "không chắc" thay vì đoán.
- Chép số đúng như in, không làm tròn, không tự suy đoán số không có trên trang."""


def norm(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def page_text(page) -> str:
    return norm(page.get_text("text"))


def visual_regions(page) -> list:
    """Large drawing clusters/images outside header/footer bands (candidate figures, tables, shaded boxes)."""
    h = page.rect.height
    rects = list(page.cluster_drawings(x_tolerance=8, y_tolerance=8))
    for img in page.get_images(full=True):
        try:
            rects.append(page.get_image_bbox(img))
        except Exception:
            pass
    keep = []
    for r in rects:
        if r.is_empty or r.y1 < HEADER_Y or r.y0 > h - FOOTER_MARGIN or r.width < 120 or r.height < 60:
            continue
        keep.append([round(r.x0, 1), round(r.y0, 1), round(r.x1, 1), round(r.y1, 1)])
    return keep


def text_chunks(doc, max_chars: int = 1200) -> list:
    chunks, section = [], ""
    for page in doc:
        pno = page.number + 1
        if pno in SKIP_PAGES:
            continue
        h = page.rect.height
        buf = []

        def flush():
            if buf:
                text = " ".join(" ".join(buf).split())
                if len(text) >= 40:
                    chunks.append({"id": f"{DOC_ID}:p{pno:03d}:t{len([c for c in chunks if c['page'] == pno]):02d}",
                                   "source_id": DOC_ID, "source": "data/advanced/raw/wb_vn2045_trading_up_vi.pdf",
                                   "page": pno, "text": text,
                                   "metadata": {"modality": "text", "section": section}})
                buf.clear()

        for b in page.get_text("blocks", sort=True):
            if b[6] != 0 or b[1] < HEADER_Y - 10 or b[3] > h - FOOTER_MARGIN + 25:
                continue
            t = norm(b[4]).strip()
            if not t or re.fullmatch(r"[\d\sI|]+", t):
                continue
            first = t.splitlines()[0].strip()
            if HEADING.match(first) and len(first) < 140 and not CAPTION.match(first):
                flush()
                section = " ".join(first.split())
            if sum(len(x) for x in buf) + len(t) > max_chars:
                flush()
            buf.append(t)
        flush()
    return chunks


def region_lines(page, regions: list, pad: float = 30, limit: int = 3500) -> str:
    """Text-layer lines inside (padded) visual regions, with x,y in pt: exact labels/numbers for the model."""
    rects = [pymupdf.Rect(r).__add__((-pad, -pad, pad, pad)) for r in regions]
    out = []
    for b in page.get_text("dict")["blocks"]:
        for line in b.get("lines", []):
            bbox = pymupdf.Rect(line["bbox"])
            if bbox.y1 < HEADER_Y or not any(bbox.intersects(r) for r in rects):
                continue
            text = norm(" ".join(s["text"] for s in line["spans"])).strip()
            if text:
                out.append(f"({bbox.x0:.0f},{bbox.y0:.0f}) {text}")
    return "\n".join(out)[:limit]


def raster_crops(page, out_dir: Path, dpi: int = 220) -> list:
    """Large embedded images (e.g. a diagram drawn as a bitmap) cropped at high resolution."""
    paths = []
    for k, img in enumerate(page.get_images(full=True)):
        try:
            r = page.get_image_bbox(img)
        except Exception:
            continue
        if r.is_empty or r.y1 < HEADER_Y or r.width < 150 or r.height < 100:
            continue
        path = out_dir / f"p{page.number + 1:03d}_img{k}.png"
        if not path.exists():
            page.get_pixmap(dpi=dpi, clip=r).save(path)
        paths.append(path)
    return paths


def call_vision(png: Path, model: str, base_url: str, key: str, hints: str = "", crops: list = (),
                retries: int = 2) -> dict:
    content = [{"type": "text", "text": VISION_PROMPT}]
    if hints:
        content.append({"type": "text", "text": "Các dòng chữ lấy từ lớp text của PDF trong vùng hình/bảng, "
                        "dạng (x,y) theo pt từ góc trên trái. Đây là nhãn và số CHÍNH XÁC đã in; dùng đúng các "
                        "giá trị này, chỉ gán chúng vào đúng chuỗi/hạng mục theo vị trí trên ảnh:\n" + hints})
    for path in [png, *crops]:
        b64 = base64.b64encode(Path(path).read_bytes()).decode()
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}})
    if crops:
        content.append({"type": "text", "text": "Ảnh sau ảnh trang là ảnh cắt độ phân giải cao của hình raster trên trang."})
    body = {"model": model, "temperature": 0, "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": content}]}
    last = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(f"{base_url.rstrip('/')}/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                data = json.loads(r.read())
            msg = data["choices"][0]["message"]["content"]
            return {"model": model, "usage": data.get("usage", {}), "latency_s": round(time.perf_counter() - start, 1),
                    "items": json.loads(msg).get("items", [])}
        except Exception as exc:  # network / 5xx / malformed JSON; recorded, never faked
            last = exc
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"vision thất bại cho {png.name}: {last!r}")


def as_text(value) -> str:
    """The model sometimes returns lists/objects for text fields; flatten them."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(as_text(v) for v in value)
    if isinstance(value, dict):
        return "; ".join(f"{k}: {as_text(v)}" for k, v in value.items())
    return str(value)


NUM = re.compile(r"-?\d+(?:[.,]\d+)*")


def _canon(n: str) -> str:
    # The PDF mixes "4.4" (chart labels) and "4,4" (Vietnamese text); compare with one separator.
    return n.lstrip("-").replace(",", ".")


def number_match(content: str, source_text: str) -> dict:
    """Share of numbers in the model output that also appear in the page text layer."""
    page_nums = {_canon(n) for n in NUM.findall(source_text.replace("−", "-"))}
    nums = [n for n in NUM.findall(content.replace("−", "-")) if len(n.strip("-")) >= 2 or "," in n or "." in n]
    found = [n for n in nums if _canon(n) in page_nums]
    return {"numbers": len(nums), "found_in_page_text": len(found),
            "rate": round(len(found) / len(nums), 3) if nums else None,
            "missing": sorted(set(nums) - set(found))[:30]}


def load_env() -> None:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, type=Path)
    ap.add_argument("--pages", help="Chỉ xử lý các trang này (vd 53,59,105)")
    ap.add_argument("--model", default="gpt-4.1")
    ap.add_argument("--dpi", type=int, default=110)
    args = ap.parse_args()
    load_env()
    key, base = os.environ.get("OPENAI_API_KEY"), os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    if not key:
        raise SystemExit("Thiếu OPENAI_API_KEY trong .env")

    doc = pymupdf.open(args.pdf)
    for d in ("pages", "vision"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    wanted = set(map(int, args.pages.split(","))) if args.pages else None

    if not wanted:
        tc = text_chunks(doc)
        (OUT / "text_chunks.json").write_text(json.dumps(tc, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"text chunks: {len(tc)}")

    visual, errors = [], []
    for page in doc:
        pno = page.number + 1
        if pno in SKIP_PAGES or (wanted and pno not in wanted):
            continue
        regions = visual_regions(page)
        if not regions:
            continue
        png = OUT / "pages" / f"p{pno:03d}.png"
        if not png.exists():
            page.get_pixmap(dpi=args.dpi).save(png)
        hints_path = OUT / "pages" / f"p{pno:03d}_hints.txt"   # reused by the agent's inspect_figure tool
        if not hints_path.exists():
            hints_path.write_text(region_lines(page, regions), encoding="utf-8")
        cache = OUT / "vision" / f"p{pno:03d}.json"
        if cache.exists():
            res = json.loads(cache.read_text(encoding="utf-8"))
        else:
            try:
                res = call_vision(png, args.model, base, key, hints=region_lines(page, regions),
                                  crops=raster_crops(page, OUT / "pages"))
            except Exception as exc:
                errors.append({"page": pno, "error": repr(exc)})
                print(f"p{pno}: LỖI {exc!r}")
                continue
            cache.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        ptext = page_text(page)
        prev = {}
        for k, it in enumerate(res["items"]):
            label = (it.get("label") or "").strip() or prev.get((it.get("title") or "").strip())
            if label:
                prev[(it.get("title") or "").strip()] = label
            slug = re.sub(r"[^0-9A-Za-z.]+", "", (label or "").replace("Hình", "fig").replace("Bảng", "tab").replace("Hộp", "box")) or f"v{k}"
            body = "\n".join(x for x in (f"{label or 'Hình/bảng không đánh số'} ({it.get('kind')}): {as_text(it.get('title'))}",
                                          as_text(it.get("panels")), as_text(it.get("content")),
                                          as_text(it.get("source_note"))) if x)
            visual.append({"id": f"{DOC_ID}:p{pno:03d}:{slug}", "source_id": DOC_ID,
                           "source": "data/advanced/raw/wb_vn2045_trading_up_vi.pdf", "page": pno, "text": body,
                           "metadata": {"modality": "visual", "kind": it.get("kind"), "label": label,
                                        "title": it.get("title"), "page_image": f"data/advanced/extracted/pages/p{pno:03d}.png",
                                        "regions": regions, "extractor": res["model"],
                                        "number_match": number_match(as_text(it.get("content")), ptext)}})
        print(f"p{pno}: {len(res['items'])} mục, {res.get('usage', {}).get('prompt_tokens')}+{res.get('usage', {}).get('completion_tokens')} token, {res.get('latency_s')} s")
    # De-duplicate IDs (two unlabeled items on one page, or a label repeated)
    seen = {}
    for v in visual:
        n = seen.get(v["id"], 0)
        seen[v["id"]] = n + 1
        if n:
            v["id"] += f"-{n}"
    name = "visual_chunks.json" if not wanted else "visual_chunks_sample.json"
    (OUT / name).write_text(json.dumps(visual, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "errors.json").write_text(json.dumps(errors, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"visual chunks: {len(visual)}, lỗi: {len(errors)}")


if __name__ == "__main__":
    main()
