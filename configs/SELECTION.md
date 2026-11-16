# Tiêu chí chọn cấu hình (ghi trước khi chạy agent với cấu hình cải thiện, 2026-11-16)

Chỉ dùng **dev** (19 câu có đáp án, 2 câu thiếu bằng chứng). Ứng viên: `baseline`, `hybrid`, `dense-filter`, `hybrid-filter`.
Model, prompt, top-k = 3, tối đa 3 lượt tìm, context 4000 ký tự giữ nguyên; chỉ đổi retriever và filter.

Xếp hạng lần lượt theo:

1. **Grounded proxy** (cao hơn): số câu có đáp án mà agent trả lời **và** context cuối có đủ evidence gold.
   Câu trả lời khi context không có evidence (như d08, d13 của baseline) không được tính.
2. **Abstention** (cao hơn): số câu thiếu bằng chứng bị từ chối.
3. **Context Evidence Rate** (cao hơn).
4. **Thời gian trung vị mỗi câu**, rồi **chi phí ước tính** (thấp hơn).

Nếu vẫn hòa, chọn cấu hình đơn giản hơn (ít thành phần hơn). Chấm tay trên dev dùng để phân tích lỗi, không đổi thứ tự trên.
Sau khi chọn, ghi `configs/selected.json` rồi mới chạy test; không chỉnh gì thêm theo kết quả test.

Filter `disease` là ngữ cảnh người dùng cung cấp trước (Day 6 BENCHMARK, mục Filter), không suy ra từ gold.
