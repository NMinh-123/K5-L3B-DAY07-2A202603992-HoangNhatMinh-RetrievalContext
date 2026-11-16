# Báo cáo Day 7 

## Dữ liệu và benchmark

- **Nguồn, quyền sử dụng, số trang và số chunk:** "Hướng dẫn chẩn đoán và điều trị một số bệnh truyền nhiễm", Bộ Y tế, QĐ 5642/QĐ-BYT ngày 31/12/2015 (văn bản công khai, https://kcb.vn/upload/2005611/20210723//Truyen-nhiem-1.pdf). PDF 86 trang, 66 trang nội dung, 14 bệnh. Corpus Day 7 là **index D** của Day 6 (semantic chunking, 233 chunk; độ dài chunk 154–1439 ký tự, trung vị 437), index Day 6 đã chọn trên dev. Mọi phương pháp retrieval dùng chung corpus này (`data/day6/chunks_D.json`, xuất trực tiếp từ Qdrant).
- **Vector store, embedding model, phiên bản và số chiều:** Qdrant 1.20.0, collection `day6_chunks`, cosine, exact search. `intfloat/multilingual-e5-large` revision `3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3`, 1024 chiều, chuẩn hóa L2, prefix `query: ` cho câu hỏi (giống Day 6), chạy local trên MPS, chi phí 0. Kiểm tra tương thích: dense trên index D cho đúng Hit@1 9/19 và Evidence Hit@3 17/19 như Day 6 ghi trong `selection.json`.
- **Agent model, cách chạy và phân chia dev/test:** `gpt-4.1-mini`, temperature 0, qua endpoint tương thích OpenAI (proxy, cấu hình trong `.env`, không commit). Cách chạy ở `src/README.md`. Chia dev/test **theo chủ đề** kế thừa Day 6: test chỉ hỏi về sốt mò (trang 60–63) và dịch hạch (trang 69–72); dev trải trên 10 bệnh khác.

  | | Có đáp án | Thiếu bằng chứng | Tổng |
  |---|---:|---:|---:|
  | dev | 19 | 2 (du01, du02) | 21 |
  | test | 6 | 2 (tu01, tu02) | 8 |

- **Cách xác định câu có/thiếu evidence và tiêu chí chấm:** 25 câu có đáp án lấy từ Day 6, mỗi câu có nguồn, trang và evidence chép nguyên văn. 4 câu thiếu bằng chứng viết mới, gần chủ đề, mỗi câu ghi lý do và cách đã kiểm tra trên **toàn văn 66 trang**, không dựa vào top-k (`data/day6/unanswerable.json`): Dengue chỉ xuất hiện trong chẩn đoán phân biệt; tài liệu không có chi phí điều trị; sốt mò không có tỷ lệ tử vong; dịch hạch không có lịch tiêm vaccin. **Evidence hit**: chunk đúng nguồn, đúng trang và chứa trọn evidence sau khi chuẩn hóa NFC, chữ thường, gộp khoảng trắng (giống Day 6; PDF có xuống dòng giữa câu). Evidence của **t03** bị cắt qua 2 chunk (`p063:D00` có "Doxycyclin:", `p063:D01` có liều), nên Hit@k theo một chunk không thể tính trúng t03; Context Evidence Rate chấp nhận evidence ghép từ các chunk cùng trang trong context.

## Retrieval và thử nghiệm

Retrieval-only chạy bằng câu hỏi gốc (không viết lại), top-3, context budget 4000 ký tự (đơn vị ký tự, gồm header nguồn), filter `disease` chỉ khi ghi rõ. Thời gian là trung vị mỗi truy vấn trên MacBook (MPS); chi phí retrieval = 0 (local). Cột agent: chạy agent thật trên dev, Context Evidence Rate của context cuối gửi model, thời gian trung vị mỗi câu và tổng chi phí ước tính cho 21 câu.

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| Baseline | Dense E5 + Qdrant, top-3, không filter, không rerank, 4000 ký tự | Phương pháp gốc Day 6 | dev | 9/19 (47%) | 17/19 (89%) | 17/19 (89%); agent 17/19 | 166 ms; agent 6,3 s/câu, $0,033 |
| Phương pháp 2 | BM25 (âm tiết đơn + cặp âm tiết), top-3, không filter | Lexical bắt tên thuốc, liều, tên vi khuẩn | dev | 11/19 (58%) | 15/19 (79%) | 15/19 (79%) | 0,5 ms; $0 |
| Phương pháp 3 | Hybrid RRF (dense + BM25, k = 60, mỗi bên 20 ứng viên), top-3, không filter | Gộp ưu điểm hai bên mà không cần chuẩn hóa score | dev | 10/19 (53%) | 17/19 (89%) | 17/19 (89%); agent 17/19 | 194 ms; agent 6,3 s/câu, $0,033 |
| Tham chiếu | TF-IDF (starter), top-3 | Lexical đơn giản để đối chiếu | dev | 9/19 (47%) | 11/19 (58%) | 11/19 (58%) | 0,3 ms; $0 |
| Cải thiện 1 | Dense + filter `disease`, top-3, 4000 ký tự | Câu mơ hồ không nêu bệnh (d08) bị kéo sang bệnh khác; filter do người dùng cung cấp loại chunk sai bệnh | dev | 12/19 (63%) | 18/19 (95%) | 18/19 (95%); agent 18/19 | 82 ms; agent 7,1 s/câu, $0,035 |
| Cải thiện 2 | Hybrid RRF + filter `disease`, top-3, 4000 ký tự | Gộp cải thiện 1 với BM25 để sửa d13 (liều "15 mg/kg") | dev | 12/19 (63%) | **19/19 (100%)** | **19/19 (100%)**; agent 19/19 | 74 ms; agent 7,2 s/câu, $0,033 |
| Cấu hình đã chọn | Hybrid RRF + filter, như cải thiện 2 | Chốt theo `configs/SELECTION.md` | test | 4/6 (67%) | 5/6 (83%) | 6/6 (100%); agent 6/6 | 92 ms; agent 8,2 s/câu, $0,012 / 8 câu |
| Baseline | Dense, như baseline dev | | test | 3/6 (50%) | 4/6 (67%) | 4/6 (67%); agent 4/5 (1 lỗi nhà cung cấp) | 105 ms; agent 6,6 s/câu, $0,012 / 8 câu |

Thêm: BM25 + filter trên dev đạt 13/19, 17/19, 17/19. Chi phí agent ước tính theo giá niêm yết OpenAI cho `gpt-4.1-mini` ($0,40 / $1,60 mỗi 1M token vào/ra); giá thực tế qua proxy có thể khác. Hybrid và BM25 là phương pháp mới; dense + filter và hybrid + filter là hai cấu hình cải thiện, mỗi cái có giả thuyết và kết quả riêng.

**Ảnh hưởng của context budget** (dense, dev, `starter/results/retrieval/dev-dense-budget*`): Evidence Hit@3 giữ 17/19 nhưng Context Evidence Rate là 5/19 ở 800 ký tự (mặc định của starter), 16/19 ở 2000 và 17/19 ở 4000. Chunk D dài tới 1439 ký tự nên budget nhỏ làm rơi chunk đúng. Chọn 4000 ký tự cho mọi cấu hình agent (context cuối trung bình ~1950–2040 ký tự; trung bình ~970 token prompt mỗi lượt gọi model, ~3 lượt gọi mỗi câu).

**Các lỗi dev đã phân tích:**

1. **Agent từ chối dù có evidence (lần chạy `baseline-v0-dev`).** Chỉ 10/19 câu có đáp án được trả lời dù Context Evidence là 17/19. Trace cho thấy ở 14 câu model đưa vào `citations` các ID có trong context nhưng không ghi `[id]` trong câu trả lời (không câu nào trích ID ngoài context); thông báo sửa lỗi của harness chỉ nói "citation không hợp lệ" nên lần sửa vẫn sai và 9 câu bị chuyển thành từ chối. *Giả thuyết:* model sửa được nếu được nói rõ thiếu gì. *Thay đổi:* tách hai loại lỗi (ID không có trong context / chưa ghi `[id]`) và chỉ cách sửa. *Kết quả:* 19/19 câu được trả lời, từ chối đúng vẫn 2/2, chi phí gần như không đổi ($0,0335 → $0,0334).
2. **d08 — câu mơ hồ không nêu bệnh** ("Kháng sinh diệt vi khuẩn được dùng trong bao lâu và có những thuốc thay thế nào?", gold: uốn ván trang 29). Baseline và hybrid lấy trang 71, 76 (dịch hạch, tiêu chảy) và agent **vẫn trả lời** bằng thông tin sai bệnh. *Giả thuyết:* filter `disease` (ngữ cảnh người dùng cung cấp) giữ retrieval trong đúng bệnh. *Kết quả:* dense + filter và hybrid + filter đều đưa trang 27–30 vào context.
3. **d13 — liều "15 mg/kg/ngày" ciprofloxacin/ofloxacin cho thương hàn** (trang 41). Dense lấy trang 76 (ciprofloxacin cho tiêu chảy) và trang 41 nhưng không đúng đoạn; agent trả lời sai. *Giả thuyết:* BM25 khớp được tên thuốc và đơn vị. *Kết quả:* hybrid đưa đúng đoạn vào context. Nhưng hybrid **không filter** lại làm hỏng d17 (câu cúm mùa không nêu bệnh: context lấy trang 23–24 thủy đậu). Chỉ hybrid + filter sửa được cả d08, d13, d17.

**Tiêu chí chọn** (`configs/SELECTION.md`, ghi trước khi chạy các cấu hình cải thiện): (1) số câu có đáp án được trả lời **và** có evidence trong context, (2) số câu thiếu bằng chứng bị từ chối, (3) Context Evidence Rate, (4) thời gian trung vị, rồi chi phí, (5) cấu hình đơn giản hơn. Kết quả dev: hybrid-filter 19/19, 2/2; dense-filter 18/19; baseline 17/19; hybrid 17/19 → chọn **hybrid-filter** (`configs/selected.json`). Lần áp dụng đầu tiên chạy khi còn 5 câu dev lỗi nhà cung cấp và đã bị hủy; sau khi chạy lại riêng các câu lỗi (`--retry-errors`, lỗi cũ giữ trong trace), tiêu chí được áp dụng lại. Lựa chọn được commit (`baa3081`) trước khi chạy test; không chỉnh gì theo kết quả test.

## Agent và đánh giá

- **Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi:** `tool_choice="required"`, mỗi lượt đúng một tool: `search_documents(query)` (model tự viết truy vấn, có thể tìm lại), `answer(answer, citations)`, `abstain(reason)`. Tối đa 3 lượt tìm (hết lượt thì chỉ còn answer/abstain), tối đa 6 lượt gọi model (hết thì từ chối). Mỗi lượt tìm lấy top-3; context là các chunk đã thấy, **khử trùng theo ID**, thêm theo thứ hạng tới khi chạm budget 4000 ký tự (chunk không vừa bị bỏ và ghi `dropped_for_budget`). Citation hợp lệ khi ID có trong context **và** xuất hiện dạng `[id]` trong câu trả lời; sai thì được sửa 1 lần, sau đó tính từ chối. Lỗi mạng, 5xx, 429 thử lại 2 lần; lỗi 4xx (ví dụ bộ lọc nội dung) không thử lại; lỗi cuối cùng ghi `status: error`, không thay bằng kết quả giả, không tính vào chất lượng. Filter `disease` lấy từ benchmark Day 6 (chỉ 6 câu mơ hồ: d06, d08, d14, d17, t03, t05), coi là ngữ cảnh người dùng cung cấp trước, không suy từ gold.

| Cấu hình / split | Tool use | Số lượt gọi | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| baseline / test | 7/7 (100%); 1 lỗi nhà cung cấp riêng | tìm: 1 lượt ×4, 2 ×1, 3 ×2; model: 2–6 lượt | 1 / 0 / 4, TB 1,60/2 (N = 5) | 6/7 (86%) | 5/5 | 2/2 (100%) | 0/5 |
| hybrid-filter / test | 8/8 (100%) | tìm: 1 lượt ×5, 3 ×3; model: 2–4 lượt | 1 / 1 / 4, TB 1,50/2 (N = 6) | 3/5 (60%) | 5/5 | 2/2 (100%) | 1/6 (t05) |

Hai mẫu số khác nhau vì t04 của baseline bị lỗi. Trên 5 câu chung (t01, t02, t03, t05, t06) cả hai đều được 8/10 điểm.

**Chấm context 0/1/2 trên test** (`context_reviews.json`): baseline 4 câu mức 2, 1 câu mức 0 (t03: không có trang 63 trong context), TB 1,60/2, N = 5; hybrid-filter 6 câu mức 2, TB 2,00/2, N = 6.

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| t01 | Diễn đạt lại | 1 lượt tìm; context p060:D01, p061:D01, p060:D00 | 6–21 ngày, TB 9–12 ngày [p060:D01] | Trả lời 2, citation đúng, context 2 |
| t02 | Trực tiếp | 1 lượt tìm; context p060:D01, p060:D00, p061:D00 | Orientia tsutsugamushi, Gram âm, họ Rickettsiacea [p060:D01] | Trả lời 2, citation đúng, context 2 |
| t03 | Cần filter | 1 lượt tìm (filter sot_mo); context p063:D00, p063:D01, p062:D04 | 0,1 g x 2 viên chia 2 lần/ngày x 5 ngày [p063:D01] | Trả lời 2, context 2; **citation sai**: D01 không có chữ "doxycyclin" (nằm ở D00, không được trích) |
| t04 | Điều kiện | 3 lượt tìm; context gồm p063:D00, p063:D01 (và p050, p036, p062) | Azithromycin 500 mg x 1–3 ngày, thêm ý về doxycyclin và chloramphenicol [p063:D01] | Trả lời **1**: ý chính đúng nhưng phần doxycyclin không được D01 nêu rõ; citation sai; context 2 |
| t05 | Cần filter | 1 lượt tìm (filter dich_hach); context p071:D00, p071:D01, p070:D03 | **Từ chối** sau 1 lần sửa: thiếu `[p071:D01]` trong câu dù nội dung đúng | Trả lời 0 (từ chối câu có đáp án), context 2 — lỗi citation, không phải retrieval |
| t06 | Diễn đạt lại | 1 lượt tìm; context p069:D01, p069:D00, p069:D02 | 1–6 ngày tùy thể [p069:D01] | Trả lời 2, citation đúng, context 2 |
| tu01 | Thiếu bằng chứng | 3 lượt tìm ("tỷ lệ tử vong sốt mò…") | Từ chối: tài liệu không nêu tỷ lệ tử vong | Từ chối đúng, không khẳng định thêm |
| tu02 | Thiếu bằng chứng | 3 lượt tìm (vaccin dịch hạch) | Từ chối: không có số mũi và khoảng cách | Từ chối đúng |
| t03 (baseline) | Cần filter | 3 lượt tìm, context trang 71, 76, 23 | Doxycyclin cho tiêu chảy và dịch hạch [p076:D00, p076:D03, p071:D00] | Trả lời 0 (sai bệnh); citation 2/3 đúng; context 0 |
| t04 (baseline) | Điều kiện | Lỗi HTTP 400 ngay lượt gọi đầu (bộ lọc nội dung của nhà cung cấp chặn câu hỏi), thử lại vẫn bị chặn | — | Lỗi nhà cung cấp, báo riêng, không chấm |

- **Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí:** `starter/results/agent/<cấu hình>-<split>/` (`traces.json` ghi mọi lượt gọi model và tool, truy vấn, top-k có rank/score, context cuối, câu trả lời, citation, token, chi phí, thời gian, lỗi; `reviews.json`, `context_reviews.json`, `agent_metrics.json`); retrieval ở `starter/results/retrieval/`; bảng so sánh `starter/results/compare-dev.md`, `compare-test.md`; lệnh chạy ở `src/README.md`. Tổng chi phí ước tính toàn bộ thí nghiệm agent: dev 5 lần chạy × 21 câu ≈ $0,17, test 2 × 8 câu ≈ $0,024.
- **Kết quả trước/sau, ví dụ lỗi, đánh đổi và giới hạn:**
  - *Ví dụ có bằng chứng:* t01. *Ví dụ thiếu bằng chứng:* tu02. *Ví dụ lỗi:* t05 (từ chối dù context đủ) và t03 baseline (trả lời sai bệnh).
  - *Kết luận:* cấu hình chọn cải thiện rõ ở retrieval và context (test: Evidence Hit@3 5/6 so với 4/6; Context Evidence 6/6 so với 4/6) nhưng **không cải thiện điểm câu trả lời trên test** (8/10 so với 8/10 trên 5 câu chung). Điểm nghẽn còn lại là hành vi citation của model: quên ghi `[id]` (t05) và chỉ trích một nửa evidence bị cắt qua hai chunk (t03, t04). Không sửa trên test vì đã xem kết quả test; cần bộ test mới nếu muốn thử (ví dụ tự gắn `[id]` khi ID đã nằm trong context, hoặc gộp chunk liền kề cùng trang).
  - *Đánh đổi:* hybrid + filter chậm hơn baseline ~0,9 s/câu trên dev (7,2 so với 6,3 s trung vị) dù số lượt gọi model trung bình tương đương (2,95 so với 3,0), nên chênh lệch chủ yếu do độ trễ endpoint giữa các lần chạy; retrieval hybrid thêm BM25 gần như không tốn thời gian (0,5 ms). Chi phí tương đương (~$0,0015/câu). Lợi ích: Context Evidence dev 19/19 so với 17/19, không còn câu trả lời khi context thiếu evidence.
  - *Giới hạn:* lợi thế lớn nhất đến từ filter, mà filter chỉ có ở 6 câu mơ hồ và giả định người dùng cung cấp bệnh; trong thực tế điều này không luôn có. Test chỉ 8 câu và chung một tài liệu với dev, nên kết quả mang tính minh họa. Tiêu chí chọn dựa trên số tự động (trả lời + có evidence), không chấm tay câu trả lời trên dev. Học viên có xem nhãn test khi viết 2 câu thiếu bằng chứng cho test và khi kiểm tra nhãn, nhưng không dùng để chỉnh cấu hình. Model không hoàn toàn tất định dù temperature 0 (d16 ghi đúng `[id]` ở lần chạy thử nhưng quên ở `baseline-v0-dev`). Endpoint proxy gây 5 lỗi tạm thời trên dev (2 lỗi DNS, 3 lỗi HTTP 503 hết kênh model; đã chạy lại riêng, ghi trong `previous_errors`) và bộ lọc nội dung chặn t04 của baseline trên test ở cả 2 lần chạy.

## Thử thách nâng cao (nếu thực hiện)

**Kết quả: 21/30 câu test đạt (70,0%), chưa đạt ngưỡng 90% (27/30) nên không nhận 15 điểm bonus.** Nhãn từng câu do Claude đọc đối chiếu ảnh trang gốc và lớp text PDF, học viên cần xác nhận (`starter/results/advanced/agent/adv-base-haiku-v2-test/reviews.json`).

- **Tài liệu, nguồn, quyền sử dụng, số trang và phiên bản:** Ngân hàng Thế giới, *Việt Nam 2045: Nâng cao vị thế thương mại trong một thế giới đang thay đổi – Con đường dẫn đến tương lai thu nhập cao của Việt Nam* (bản tiếng Việt của *Viet Nam 2045: Trading up in a Changing World*), phát hành 2024-12-06, số báo cáo 194764; 117 trang; giấy phép CC BY-NC 3.0 IGO; link, SHA-256 và lệnh tải lại ở `data/advanced/README.md`. PDF không commit.
- **Cách xử lý text, bảng, ảnh, biểu đồ; phạm vi index và phần chưa xử lý:** PyMuPDF trích text theo đoạn (362 chunk, giữ trang và mục). Trang có cụm nét vẽ/ảnh lớn (67 trang) được dựng ảnh 150 dpi và gửi gpt-4.1 kèm các dòng chữ của lớp text PDF trong vùng hình (nhãn và số chính xác) và ảnh cắt 220 dpi của hình raster; mỗi hình/bảng thành một chunk có số trang, số hình, loại, ảnh trang (82 mục: 22 bảng, 20 cột, 15 đường, 11 phân tán, 3 sơ đồ…; chi phí ≈ 1,2 USD). Mỗi số trong bản trích được đối chiếu tự động với lớp text (trung vị khớp 100%; 9 mục < 80% được đánh dấu). Index: 523 chunk (phần hình/bảng > 1.500 ký tự được tách, lặp tiêu đề), E5 + Qdrant collection `day7_advanced`, BM25 và hybrid RRF; 6 chunk dài hơn 512 token bị E5 cắt. Không xử lý: bìa và trang trống (1, 2, 3, 114-117), dải trang trí đầu trang. **Giới hạn đã thấy:** đối chiếu số không phát hiện được lỗi gán sai số vào chuỗi/hàng; bản trích Bảng 3 lệch một hàng do hàng "Khai mỏ khác" để trống (gây trượt at03, at04, at25); bản mô tả sơ đồ Hình 23 sai thành viên vài khung.
- **Cách chia benchmark, schema evidence, model và cấu hình đã chốt trên dev:** `data/advanced/benchmark.json`, dev 18 câu (15 có đáp án + 3 thiếu bằng chứng), test 30 câu (bảng 6, ảnh/sơ đồ 5, biểu đồ 5, multi-hop 4, text 4, thiếu bằng chứng 6), soạn từ ảnh trang gốc và lớp text, mỗi câu có trang, vùng/ô/hàng/cột hoặc trích dẫn nguyên văn (48 trích dẫn kiểm tra tự động khớp trang). Thay đổi benchmark trước test (chỉ cách viết tương đương cho bộ chấm nháp và làm rõ bảng con B3.1) ghi ở `data/advanced/BENCHMARK_CHANGES.md`. Agent: hybrid RRF top-6, lượt tìm đầu bằng câu hỏi gốc, tối đa 4 lượt tìm, context 9.000 ký tự, citation ID ngắn chứa trang và số hình, `claude-haiku-4-5` điều phối (gpt-4.1-mini hết kênh trên endpoint; học viên chọn đổi model). Tool tùy chọn `inspect_figure` đọc lại ảnh trang theo câu hỏi.

| Cấu hình dev (18 câu) | Thay đổi / giả thuyết | Đạt (đọc tay) | Context đủ evidence | Chi phí | Trung vị |
|---|---|---:|---:|---:|---:|
| adv-base (gpt-4.1-mini, v1) | mô tả hình lúc index, không đọc lại hình | 8/18 (5 lỗi 503) | 9/10 | $0,03 | 7,3 s |
| adv-base-haiku | đổi sang claude-haiku-4-5, ID citation ngắn, tự gắn [ID] hợp lệ | 15/18 | 14/15 | $0,22 | 9,1 s |
| adv-inspect-haiku | + đọc lại ảnh trang theo câu hỏi | 14/18 | 13/15 | $0,39 | 17,1 s |
| **adv-base-haiku-v2**  | + lượt tìm đầu bằng câu hỏi gốc, top-6, nhắc gọi tool | **17/18** | 15/15 | $0,19 | 4,2 s |
| adv-inspect-haiku-v2 | v2 + đọc lại ảnh | 17/18 | 15/15 | $0,35 | 13,9 s |

Chọn `adv-base-haiku-v2` theo `configs/advanced/SELECTION.md` (hòa 17/18 và 15/15, rẻ và nhanh hơn); `configs/advanced/selected.json` ghi thời điểm chốt và SHA-256 cấu hình, benchmark, corpus, code trước khi chạy test (học viên yêu cầu chưa commit).

| Nhóm test | Đạt / tổng | Tỷ lệ | Lỗi / nhận xét |
|---|---|---|---|
| Bảng | 3/6 | 50% | at03, at04: lệch hàng ở bản trích Bảng 3; at06: sai tên chỉ số và năm ở Bảng A2.2 |
| Ảnh, sơ đồ | 2/5 | 40% | at07 thừa nước, at08 thiếu Nga (Hình 23 raster); at10 từ chối vì mô tả Hình B1.1 không nối chặng với bước |
| Biểu đồ | 3/5 | 60% | at13 nhầm mốc trục -0,55% thành giá trị; at14 cộng lẫn nhãn tổng |
| Multi-hop | 4/4 | 100% | at17 có phép tính 13.485/0,93 ≈ 14.505 |
| Text | 4/4 | 100% | |
| Thiếu bằng chứng | 5/6 | 83% | at25 đưa số cho ô trống (lệch hàng Bảng 3) |
| **Toàn bộ** | **21/30** | **70,0%** | 0 lỗi hệ thống; Context đủ evidence 24/24 |

- **Trace, nhãn và evidence từng câu; thời gian, chi phí, lỗi hệ thống:** `starter/results/advanced/agent/adv-base-haiku-v2-test/` (`traces.json`: mọi lượt gọi model và tool, truy vấn, top-k, context, câu trả lời, citation; `reviews.json`: đạt/không đạt và lý do từng câu); bảng duyệt trên dashboard (`python3 -m http.server 8765 --directory dashboard`). Test: chi phí ước tính $0,47 (giá niêm yết), trung vị 4,9 s/câu, 0 lỗi hệ thống. Toàn bộ thử nghiệm nâng cao ≈ 3 USD gồm cả trích xuất vision. Bộ chấm nháp tự động bất đồng với chấm tay ở 4 câu (at13, at14 chấm nhầm đạt; at17, at29 chấm nhầm trượt), nên chỉ dùng làm sàng lọc.
- **Giới hạn, cách tái lập và đường dẫn benchmark:** retrieval không phải điểm nghẽn (Context đủ evidence 24/24); lỗi nằm ở **độ chính xác của bản trích bảng/sơ đồ** và **đọc biểu đồ không in số**. Tool đọc lại ảnh với claude-haiku-4-5 không cải thiện trên dev (sửa vài câu nhưng đọc sai câu khác), nên không được chọn; hướng sửa đáng thử với bộ test mới là trích bảng theo tọa độ từ (hàng trống giữ chỗ) và đọc lại sơ đồ raster bằng model vision mạnh hơn. Endpoint proxy không ổn định (503 "no available channel", hết hạn mức giữa chừng), làm phải đổi model; lần chạy dev đầu tiên lưu ở `*-v1-dev` làm bằng chứng. Tái lập: `src/advanced/extract.py` → `src/advanced/index.py` → `src/advanced/run_adv.py --config configs/advanced/adv-base-haiku-v2.json --split test` (chi tiết trong `src/README.md`).
