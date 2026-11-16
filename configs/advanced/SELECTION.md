# Tiêu chí chọn cấu hình — thử thách nâng cao (ghi trước khi chạy dev, 2026-11-16)

Chỉ dùng **dev** (15 câu có đáp án, 3 câu thiếu bằng chứng; `data/advanced/benchmark.json`). Ứng viên: `adv-base`, `adv-inspect`, `adv-inspect-41` (và cấu hình cải tiến nếu thêm sau, mỗi cấu hình có giả thuyết riêng).

Xếp hạng lần lượt theo:

1. **Số câu dev đạt** theo cách chấm của ADVANCED.md (đúng và đủ, đúng đơn vị/điều kiện/phép tính, citation chỉ đúng trang và hình/bảng/đoạn, từ chối đúng câu thiếu bằng chứng). Nhãn tự động chỉ là nháp; câu trượt tự động được đọc lại trước khi tính.
2. **Context Evidence (đủ mọi evidence)** trên câu có đáp án.
3. **Chi phí ước tính**, rồi **thời gian trung vị** mỗi câu.

Benchmark test (30 câu) chỉ chạy **một lần** với cấu hình đã chốt, sau khi `configs/advanced/selected.json` được ghi. Không chỉnh gì theo kết quả test; nếu phải chỉnh, tạo bộ test mới trước khi báo điểm.

## Cập nhật ứng viên (2026-11-16, trước khi chạy test)

Endpoint báo liên tục "No available channel for model gpt-4.1-mini" nên `adv-inspect` (gpt-4.1-mini) không chạy được. Ứng viên so sánh chính là cặp cùng điều phối gpt-4.1, chỉ khác tool đọc lại hình: `adv-base-41` và `adv-inspect-41`. Kết quả `adv-base` (gpt-4.1-mini) chạy được trước đó giữ làm tham chiếu chi phí. Tiêu chí xếp hạng không đổi.

## Cập nhật ứng viên lần 2 (2026-11-16, trước khi chạy test)

Theo yêu cầu học viên, đổi model điều phối và model đọc lại hình sang `claude-haiku-4-5-20251001` (gpt-4.1-mini hết kênh trên endpoint; gpt-4.1 đắt hơn). Bản trích xuất vision dùng cho index (gpt-4.1) giữ nguyên. Ứng viên chính: `adv-base-haiku` và `adv-inspect-haiku` (chỉ khác tool inspect_figure). Lượt `adv-base-41` bị dừng giữa chừng, không dùng để chọn. Tiêu chí xếp hạng không đổi.

## Cấu hình cải tiến v2 (2026-11-16, sau khi đọc lỗi dev của bản haiku, trước khi chạy test)

Điểm dev đọc tay: `adv-base-haiku` 15/18, `adv-inspect-haiku` 14/18 (ad01 đạt nhưng bộ chấm nháp đòi dấu trừ). Lỗi chính: truy vấn agent tự viết đẩy đoạn văn cần thiết ra ngoài top-4 (ad13), model trả lời bằng văn bản thay vì gọi tool (ad11), đọc sai sơ đồ/biểu đồ không in số (ad05, ad10). Thêm `adv-base-haiku-v2` và `adv-inspect-haiku-v2` (lượt tìm đầu bằng câu hỏi gốc, top-k 6, nhắc gọi tool một lần). Tiêu chí xếp hạng không đổi; tất cả ứng viên dev được so sánh bằng điểm đọc tay.
