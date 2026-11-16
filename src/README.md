# Code

| File | Công dụng |
|---|---|
| `prepare_data.py` | Xuất chunk từ Qdrant Day 6 và benchmark Day 6 sang schema Day 7, thêm câu thiếu bằng chứng, kiểm tra nhãn |
| `evaluation.py` | Định nghĩa evidence hit (giống Day 6: NFC, chữ thường, gộp khoảng trắng) và context có evidence |
| `dense.py` | Dense retrieval: E5 `multilingual-e5-large` (revision cố định, prefix `query: `) + Qdrant `day6_chunks` |
| `lexical.py` | BM25 (âm tiết đơn + cặp âm tiết liền kề) và hybrid RRF (dense + BM25, k = 60, mỗi bên lấy 20 ứng viên) |
| `eval_retrieval.py` | Chấm Hit@1, Evidence Hit@3, Context Evidence Rate cho một retriever trên một split |
| `agent.py` | Agent tool calling: `search_documents`, `answer`, `abstain`; kiểm tra citation, context budget, trace, token, chi phí |
| `run_agent.py` | Chạy agent theo một file trong `configs/`, lưu trace, summary và mẫu nhãn chấm tay |
| `compare.py` | So sánh các cấu hình agent; với `--select` áp dụng `configs/SELECTION.md` và ghi `configs/selected.json` |
| `review_sheet.py` | Gộp trace, đáp án chuẩn, đoạn được trích và nhãn thành một bảng duyệt Markdown |

## Chuẩn bị

1. Bật Qdrant của Day 6 (`docker compose up -d` trong `starter/` của repo Day 6). Collection `day6_chunks` có 3 index A, B, D (677 điểm).
2. Python 3.11+:

   ```bash
   python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
   ```

   Model E5 (~2.2 GB) tải về cache Hugging Face lần đầu; các lần sau có thể đặt `HF_HUB_OFFLINE=1`.
3. Chép `.env.example` thành `.env` và điền `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `AGENT_MODEL` (endpoint tương thích OpenAI có tool calling). Không commit `.env`.

## Chạy lại toàn bộ

```bash
python src/prepare_data.py --day6 ../K5-L3B-DAY06-HoangNhatMinh-2A202603992-DataIngestion --strategy D
```

Retrieval trên dev (context budget 4000 ký tự), không filter rồi có filter:

```bash
for r in tfidf bm25 dense hybrid; do python src/eval_retrieval.py --retriever $r --split dev --budget 4000 --output starter/results/retrieval/dev-$r; done
```

```bash
for r in bm25 dense hybrid; do python src/eval_retrieval.py --retriever $r --split dev --budget 4000 --use-filter --output starter/results/retrieval/dev-$r-filter; done
```

Ảnh hưởng của context budget (dense, dev):

```bash
for b in 800 2000; do python src/eval_retrieval.py --retriever dense --split dev --budget $b --output starter/results/retrieval/dev-dense-budget$b; done
```

Agent trên dev, so sánh và chốt cấu hình:

```bash
for c in baseline hybrid dense-filter hybrid-filter; do python src/run_agent.py --config configs/$c.json --split dev; done
```

```bash
python src/compare.py --split dev --configs baseline hybrid dense-filter hybrid-filter --select
```

Chỉ sau khi `configs/selected.json` đã được commit, chạy test cho baseline và cấu hình đã chọn:

```bash
for c in baseline hybrid-filter; do python src/run_agent.py --config configs/$c.json --split test; done
```

Câu bị lỗi nhà cung cấp (mạng, 5xx, bộ lọc nội dung) chạy lại riêng bằng `--retry-errors`; lỗi cũ được giữ trong trường `previous_errors` của trace.

Chấm tay (`reviews.json`, `context_reviews.json`), tạo bảng duyệt và tổng hợp:

```bash
python src/review_sheet.py --split test --configs baseline hybrid-filter
```

```bash
for c in baseline hybrid-filter; do python starter/agent_metrics.py --cases data/day6/cases.json --traces starter/results/agent/$c-test/traces.json --reviews starter/results/agent/$c-test/reviews.json --split test --output starter/results/agent/$c-test/agent_metrics.json; done
```

## Kết quả (`starter/results/`)

- `retrieval/<split>-<phương pháp>[-filter]/`: `summary.json`, `traces.json` (top-k có rank, score, hit), `errors.json`.
- `agent/<cấu hình>-<split>/`: `traces.json` (mọi lượt gọi model và tool, câu hỏi viết lại, context cuối, câu trả lời, citation, token, chi phí, thời gian, lỗi), `summary.json`, `reviews.json` (nhãn câu trả lời và citation), `context_reviews.json` (nhãn context 0/1/2), `agent_metrics.json`.
- `compare-<split>.md`: bảng so sánh cấu hình; `review_sheet_<split>.md`: bảng duyệt nhãn.

Thư mục này bị `.gitignore` bỏ qua; khi nộp dùng `git add -f starter/results/...`.

## Dữ liệu (`data/day6/`)

- Corpus: index **D** của Day 6 (semantic chunking, 233 chunk, 66 trang), là index Day 6 đã chọn trên dev. Mọi retriever dùng chung corpus này.
- `questions.json`: 25 câu có đáp án từ Day 6 (19 dev, 6 test), chia theo chủ đề: test chỉ hỏi về sốt mò và dịch hạch.
- `unanswerable.json`: 4 câu thiếu bằng chứng viết cho Day 7 (2 dev, 2 test), mỗi câu ghi lý do và cách đã kiểm tra trên toàn văn.
- `cases.json`: 29 câu cho đánh giá agent, có trường `answerable` và `filter` (nếu có).
- `gold_check_<index>.json`: câu nào có evidence nằm trọn một chunk, câu nào bị cắt qua 2 chunk.

Giới hạn đã biết: trên index D, evidence của **t03** bị cắt qua 2 chunk của trang 63 (tên thuốc ở cuối `p063:D00`, liều ở đầu `p063:D01`), nên Hit@k theo một chunk không tính trúng câu này; Context Evidence Rate vẫn tính được nếu cả hai chunk vào context. Index A (fixed 500/50) cắt mất evidence của d09, d11 nên không dùng.

## Thử thách nâng cao (`src/advanced/`)

| File | Công dụng |
|---|---|
| `extract.py` | PyMuPDF: chunk text theo đoạn; trang có hình/bảng → ảnh trang + nhãn lớp text → vision (gpt-4.1) chép từng hình/bảng; đối chiếu số với lớp text (`.venv` Day 7, cần `pymupdf`) |
| `index.py` | Gộp chunk text + hình/bảng thành `data/advanced/chunks.json`, nhúng E5 vào Qdrant `day7_advanced` (venv Day 6) |
| `agent_adv.py` | Agent: `search_documents`, `inspect_figure` (đọc lại ảnh trang theo câu hỏi), `answer`, `abstain`; ID citation ngắn chứa trang và số hình |
| `run_adv.py` | Chạy agent hoặc chỉ retrieval trên dev/test; chấm nháp tự động; `--regrade`, `--retry-errors` |

Tài liệu và cách tải: `data/advanced/README.md`. Benchmark: `data/advanced/benchmark.json` (thay đổi: `BENCHMARK_CHANGES.md`). Cấu hình: `configs/advanced/` (tiêu chí `SELECTION.md`, lựa chọn `selected.json`).

```bash
.venv/bin/python src/advanced/extract.py --pdf data/advanced/raw/wb_vn2045_trading_up_vi.pdf --dpi 150
```

```bash
<day6>/starter/.venv/bin/python src/advanced/index.py
```

```bash
<day6>/starter/.venv/bin/python src/advanced/run_adv.py --retrieval-only --retriever hybrid --split dev --top-k 4
```

```bash
<day6>/starter/.venv/bin/python src/advanced/run_adv.py --config configs/advanced/adv-base-haiku-v2.json --split test
```

Dashboard: `python3 dashboard/build_data.py` rồi `python3 -m http.server 8765 --bind 127.0.0.1 --directory dashboard`.
