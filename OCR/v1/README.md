# OCR v1 — LPRNet bản đầu tiên (baseline)

> Thư mục này chỉ có tài liệu. Notebook v1 không có trong thư mục nguồn cục bộ (`OCR/` chỉ có `v2`–`v5`), nên nội dung dưới đây tổng hợp lại từ phần "Vấn đề ở bản cũ" trong notebook v2, nơi các lỗi của v1 được phân tích và sửa.

## Mục tiêu và phương pháp
- Nhận dạng ký tự biển số Việt Nam (dữ liệu VNLP) sau bước phát hiện biển bằng YOLOv8n.
- Mô hình: một biến thể LPRNet tự viết (CNN + CTC), nạp trọng số pre-trained LPRNet của sirius-ai.
- Biển 2 dòng: cắt đôi ảnh theo chiều cao rồi ghép thành một ảnh 94×24.

## Chức năng đã triển khai
- Đọc dữ liệu VNLP, crop biển theo bbox, resize về 94×24.
- Train với CTC loss, LR 1e-4, Cosine 40 epoch.
- Giải mã greedy CTC và đo FPS.

## Hạn chế và lỗi kỹ thuật
| Vấn đề | Hậu quả |
| :--- | :--- |
| Kiến trúc tự viết khác bản gốc, chỉ nạp được **8/47** tensor | Thực chất là train từ đầu, mất lợi ích transfer learning |
| Không crop khi bbox rộng gần bằng ảnh, rồi heuristic `h/w > 0.65` cắt đôi cả ảnh toàn cảnh | Đầu vào thành lưới tản nhiệt và mặt đường |
| Cắt biển 2 dòng đúng ở `h/2` | Mất hoặc lặp ký tự ở chỗ nối hai dòng |
| `cv2.imread` tự xoay ảnh theo EXIF | Ảnh lệch với bbox |
| Ký tự ngoài bảng bị xóa khỏi nhãn | Nhãn ngắn đi, mô hình học sai |
| Augmentation chỉ chỉnh sáng/tối | Không bền với bbox lệch, nghiêng, mờ |
| Xe máy chỉ chiếm ~15% dữ liệu | Mô hình thiên về biển ô tô |
| LR 1e-4 + Cosine về 0 sau 40 epoch | Loss vẫn còn giảm, chưa hội tụ |
| Đo FPS không có `torch.cuda.synchronize()` | Số FPS không chính xác |

Các lỗi này được sửa ở [v2](../v2/README.md).
