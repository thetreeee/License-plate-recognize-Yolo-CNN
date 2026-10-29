# OCR v4 — Làm sạch dữ liệu + Tách 2 dòng theo khe nghiêng

Notebook: [`VN_ALPR_LPRNet_v4_CleanData_SlopedSplit.ipynb`](VN_ALPR_LPRNet_v4_CleanData_SlopedSplit.ipynb) · Kết quả: [`lprnet_v4/`](lprnet_v4/)

## Mục tiêu và phương pháp
Sau v3, phần lớn lỗi còn lại đến từ **dữ liệu** và **cách tách dòng**, không phải từ mô hình. v4 tập trung vào ba việc:
1. Làm sạch nhãn 1/2 dòng và chống rò rỉ dữ liệu.
2. Tách biển 2 dòng theo khe thật giữa hai dòng chữ, kể cả khi biển nghiêng.
3. Đánh giá trung thực hơn: giữ mọi mẫu khó trong Test và gắn cờ chất lượng.

## Chức năng đã triển khai
- **Sửa nhãn layout**: áp dụng 21 ảnh sai nhãn 1/2 dòng và 1 bbox sai đã xem tay. Notebook tự tìm ảnh **nghi sai layout** theo tỉ lệ rộng/cao, in ảnh để xem, ghi [`layout_suspects.csv`](lprnet_v4/layout_suspects.csv). Có thể sửa thêm qua `layout_corrections.csv`.
- **Tách dòng theo khe nghiêng** (`find_row_gap`): tìm đường thẳng có thể nghiêng, cắt qua ít nét mực nhất giữa hai dòng. Từng dòng được **nắn thẳng** bằng `cv2.remap` rồi mới resize về 94×24. Mẫu nào tách không chắc chắn thì gắn cờ để xem lại.
- **`WidthBatchNorm2d`**: dùng chung gamma/beta nhưng giữ thống kê chạy **riêng** cho ảnh 94 và ảnh 188. Lớp này sửa lỗi biển 1 dòng bị giảm độ chính xác ở v3.
- **Chia tập theo nhóm liên thông**: hai ảnh cùng nhóm nếu cùng biển số **hoặc** cùng sha256 (union-find), nên ảnh trùng byte không thể lọt vào cả train và test.
- **Val/Test giữ mọi mẫu**: ảnh không đọc được hoặc bbox sai cho dự đoán rỗng và tính là sai.
- **Cờ chất lượng** (mờ, crop nhỏ, tách không chắc, nhãn lạ) và bảng kết quả Test theo từng cờ.
- **Sanity check**: mô hình phải học thuộc 32 ảnh trước khi train thật, dùng `CTCLoss(zero_infinity=False)`. Checkpoint lưu **dấu vân tay dữ liệu**, nên không thể resume khi dữ liệu đã đổi.

## Kết quả (tập Test, 5.649 ảnh, giữ cả mẫu khó)
| Nhóm | Acc v3 | **Acc v4** | CER v4 |
| :--- | ---: | ---: | ---: |
| 1 dòng | 91.11% | **95.15%** | 0.0090 |
| 2 dòng ô tô | 92.84% | 91.81% | 0.0141 |
| xe máy | 81.59% | **85.35%** | 0.0269 |
| **Tổng** | 90.31% | **92.58%** | 0.0136 |

Best Val Acc: 92.18% (epoch 81). Theo cờ chất lượng:

| Cờ | Số ảnh | Acc |
| :--- | ---: | ---: |
| Ảnh sạch (không cờ) | 5.541 | 94.26% |
| Tách 2 dòng không chắc | 11 | 63.64% |
| Nhãn lạ | 97 | **0.00%** |

Trên ảnh sạch, biển 2 dòng ô tô đạt 95.75%. Con số tổng 91.81% bị kéo xuống chủ yếu bởi 76 ảnh "nhãn lạ".

## Hạn chế và lỗi kỹ thuật
- **Ảnh lỗi / nhãn sai trong train phạt sai mô hình.** Ví dụ biển ghi 98-M2 nhưng nhãn là 98N26985.
- **Bảng ký tự không có chữ R** (biển rơ-moóc). 446 ảnh có ký tự ngoài bảng bị loại khỏi train, nên mô hình không bao giờ đoán được R.
- **Regex định dạng không nhận seri 2 chữ + 6 số** (vd. 29-MD1 524.62). Các ảnh này bị gắn "nhãn lạ" và loại khỏi train; trên Test chúng đạt 0%.
- **Giải mã greedy thêm ký tự thừa** (`890B199639`, `29H1174111`, `8B1H106876`) và nhầm chữ ↔ số. `fix_vn_plate` chỉ đổi được chữ ↔ số, không bỏ được ký tự thừa.
- **Chữ hiếm học kém**: A có 11.111 mẫu, trong khi Z chỉ có 114, X 224, K 247, G 343. Mô hình hay nhầm K, X, G.

Các lỗi này được sửa ở [v5](../v5/README.md).
