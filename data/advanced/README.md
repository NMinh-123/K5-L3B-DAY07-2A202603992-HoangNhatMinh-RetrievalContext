# Tài liệu cho thử thách nâng cao

| | |
|---|---|
| Tên | Việt Nam 2045: Nâng cao vị thế thương mại trong một thế giới đang thay đổi – Con đường dẫn đến tương lai thu nhập cao của Việt Nam (bản tiếng Việt của *Viet Nam 2045: Trading up in a Changing World – Pathways to a High-Income Future*) |
| Tác giả | Ngân hàng Thế giới (World Bank), tài trợ Australian Aid |
| Phiên bản | Phát hành 2024-12-06, số báo cáo 194764, file `P17878411dab3c0b918424157e400ddf99c.pdf` (mục "Vietnamese") |
| Trang gốc | https://openknowledge.worldbank.org/entities/publication/c3d2e6ed-e439-4091-be7e-237c99f864ed/full |
| Link tải | https://openknowledge.worldbank.org/bitstreams/3f60c067-695a-42ea-a965-c1335820305a/download |
| Kiểm tra file | 6.697.349 byte, SHA-256 `72d9cacb925acf0d05a1b3b7df27120081212bfee947e8bf646ee58c476182b7` |
| Giấy phép | CC BY-NC 3.0 IGO (https://creativecommons.org/licenses/by-nc/3.0/igo), chủ sở hữu World Bank. Dùng phi thương mại, ghi nguồn. |
| Số trang | 117 trang PDF; số trang in ở chân trang trùng số trang PDF |

PDF **không được commit** (`data/advanced/raw/` nằm trong `.gitignore`). Tải lại:

```bash
mkdir -p data/advanced/raw && curl -L -o data/advanced/raw/wb_vn2045_trading_up_vi.pdf "https://openknowledge.worldbank.org/bitstreams/3f60c067-695a-42ea-a965-c1335820305a/download" && shasum -a 256 data/advanced/raw/wb_vn2045_trading_up_vi.pdf
```

## Kiểm kê nội dung (2026-11-16)

- **Text:** PDF có lớp text (Adobe PDF Library), 67.721 từ; trích bằng PyMuPDF giữ đúng dấu cách và số. Trang gần như không có chữ: 2, 3, 114, 115, 117 (bìa, trang trống).
- **Biểu đồ:** khoảng 43 "Hình", phần lớn là biểu đồ vector có ghi số liệu (ví dụ Hình 20 trang 53: đóng góp tăng trưởng 4,4 / 0,5 / 0,8; GDP đầu người 4.500 / 11.500 / 15.000).
- **Sơ đồ, ảnh có thông tin:** Hình 23 trang 59 (sơ đồ Venn các FTA của Việt Nam, có cờ quốc gia), Hình B1.1 trang 38 (bản đồ thế giới + quy trình sản xuất vi mạch 7 bước), Hình 21 trang 56 (sơ đồ minh họa lộ trình nâng cao vị thế GVC trên nền biểu đồ).
- **Bảng:** 0.1 (tr. 18), B3.1 (tr. 63), 1 (tr. 86), 2 (tr. 91), A2.1 (tr. 105), A2.2 (tr. 106), 3 (tr. 112).
- **Hộp:** 1 Bán dẫn (tr. 37), 2 Phân tích cơ hội xuất khẩu (tr. 56), 3 Biện pháp phi thuế quan (tr. 61).
- **Ảnh trang trí, không dùng làm câu hỏi:** bìa (tr. 1), dải trang trí đầu trang, 20 ảnh raster ở tr. 53 là vệt mờ của dải bất định trong Hình 20.

Vấn đề đã biết: bộ dò bảng tự động của PyMuPDF làm dính chữ và tách sai số trong ô (`Tỷlệchiếtkhấu`, `0054 ,`); text thường của bảng đúng nhưng mất liên kết hàng–cột, nên bảng cần dựng lại theo tọa độ từ hoặc bằng vision và đối chiếu với bản gốc.
