# OCR v2 — Transfer 100% trọng số LPRNet gốc

Notebook: [`VN_ALPR_LPRNet_v2_FullTransfer.ipynb`](VN_ALPR_LPRNet_v2_FullTransfer.ipynb) · Kết quả: [`lprnet_v2/`](lprnet_v2/)

## Mục tiêu và phương pháp
- Dùng **đúng kiến trúc LPRNet của sirius-ai/LPRNet_Pytorch** (Zherzdev & Gruzdev, 2018): Global Context và `small_basic_block` không residual. Nhờ vậy nạp được **100%** trọng số pre-trained, thay vì 8/47 tensor như v1.
- 3 tầng phụ thuộc số lớp được **cắt theo ký tự chung** (0–9, chữ Latinh) chứ không khởi tạo ngẫu nhiên.
- Bảng ký tự: 30 ký tự biển VN + blank (31 lớp). Đầu vào 94×24, CTC 18 bước.

## Chức năng đã triển khai
- Parser VNLP và kiểm tra chất lượng dữ liệu (bbox, EXIF, nhãn). Nhãn được chuẩn hóa `O→0`, `I→1`; mẫu có ký tự lạ bị bỏ, và nhãn được kiểm tra theo định dạng biển VN.
- Chia Train 70% / Val 15% / Test 15% **theo chuỗi biển số** để tránh data leakage.
- Tiền xử lý: luôn crop theo bbox có đệm, đọc ảnh theo pixel gốc (bỏ qua EXIF). Chỉ cắt đôi khi nhãn là biển 2 dòng, cắt **chồng lấn 5%**.
- Augmentation: lệch bbox (mô phỏng YOLO), xoay/nghiêng, phối cảnh, mờ, nhiễu, JPEG, độ phân giải thấp. Không lật ảnh.
- Lấy mẫu lặp xe máy ×2.5 (`WeightedRandomSampler`).
- Train 2 pha: 3 epoch đóng băng, rồi 80 epoch AdamW + OneCycleLR (max 1e-3). Checkpoint được lưu lên Drive mỗi epoch và resume được.
- Giải mã greedy CTC + `fix_vn_plate` (sửa chữ ↔ số theo vị trí).
- Đánh giá zero-shot, đánh giá Test theo nhóm, in ảnh lỗi, hàm suy luận cho pipeline YOLO → LPRNet, đo FPS có `cuda.synchronize`.

## Kết quả (tập Test, 5.524 ảnh)
| Nhóm | Số ảnh | Acc (greedy) | CER |
| :--- | ---: | ---: | ---: |
| 1 dòng | 2.901 | 94.93% | 0.0087 |
| 2 dòng ô tô | 1.830 | 87.92% | 0.0198 |
| xe máy | 793 | **64.94%** | 0.0521 |
| **Tổng** | 5.524 | **88.31%** | 0.0192 |

Best Val Acc: 87.77% (epoch 83).

## Hạn chế và lỗi kỹ thuật
- **Biển 2 dòng bị nén vào chung 94×24.** Mỗi dòng chỉ còn 47 px, tương đương khoảng 9 bước CTC. Dòng dưới có 5 chữ số, dấu chấm, và CTC cần blank chen giữa các số lặp, nên mô hình **mất ký tự** (`99→9`, `555→55`). Đây là nguyên nhân xe máy kẹt ở ~65%.
- Global Context dùng avgpool kích thước cố định, chỉ chạy được với ảnh rộng 94.
- Augmentation chồng nhiều phép làm xấu ảnh (tối + mờ + phân giải thấp + JPEG), sinh ra mẫu mắt người cũng không đọc được, thành nhiễu nhãn.

Các lỗi này được sửa ở [v3](../v3/README.md).
