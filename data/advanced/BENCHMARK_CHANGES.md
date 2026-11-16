# Thay đổi benchmark nâng cao

## 2026-11-16 — sau lần chạy dev đầu tiên, trước khi chạy test

Không đổi đáp án, evidence hay chia dev/test. Chỉ hai loại sửa:

1. **Cách viết tương đương cho bộ chấm tự động nháp** (`key_facts`, phân tách bằng `|`): ví dụ `US cent/kWh|US¢/kWh`, `USD năm 2022|2022 USD`, `Úc|Australia`, `bước 4|bước thứ tư`. Lý do: lần chạy dev đầu chấm trượt các câu trả lời đúng chỉ vì cách viết đơn vị/tên khác (ad05, ad08, ad09). Áp dụng cho cả dev và test trước khi test được chạy; nhãn chính thức vẫn do người chấm.
2. **Làm rõ câu hỏi mơ hồ về Bảng B3.1** (ad01, at01, at20): bảng có ba bảng con cùng tiêu đề cột (xuất khẩu ở trên, sản xuất ở giữa, nhập khẩu ở dưới, theo đoạn văn trang 63). Câu hỏi cũ ghi "phần tác động đến xuất khẩu", trùng tiêu đề chung của cả bảng; nay ghi "bảng con trên cùng: xuất khẩu năm 2035". Đáp án không đổi.

## 2026-11-16 — lần 2, trước khi chạy test

Bộ chấm nháp coi `giảm X`/`âm X` tương đương key fact `-X`; thêm `USD (2022)` cho ad08. Lý do: ad01 và ad08 trên dev trả lời đúng nhưng bị chấm trượt vì cách viết. Đáp án, evidence không đổi.
