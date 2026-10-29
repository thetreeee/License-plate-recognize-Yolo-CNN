# OCR v3 — Biển 2 dòng đầu vào 188×24

Notebook: [`VN_ALPR_LPRNet_v3_TwoRow188.ipynb`](VN_ALPR_LPRNet_v3_TwoRow188.ipynb) · Kết quả: [`lprnet_v3/`](lprnet_v3/)

## Mục tiêu và phương pháp
Sửa lỗi mất ký tự ở biển 2 dòng của v2. Hai dòng không còn bị nén chung vào 94×24: **mỗi dòng được resize riêng về 94×24** như một biển 1 dòng, rồi ghép ngang thành **188×24**. Số bước CTC tăng từ 18 lên **42** (khoảng 21 bước mỗi dòng). Biển 1 dòng vẫn giữ 94×24.

## Chức năng đã triển khai
- **Global Context theo độ rộng**: giữ nguyên avgpool gốc (cùng tỉ lệ cửa sổ), rồi nội suy về đúng độ rộng đầu ra khi ảnh rộng 188. Với ảnh 94 kết quả giống hệt bản gốc, nên vẫn transfer 100% trọng số.
- **Batch theo chiều rộng**: mỗi batch chỉ chứa ảnh 94 hoặc chỉ ảnh 188, thứ tự các batch được xáo. Vẫn lấy mẫu lặp xe máy ×2.5.
- **Augmentation nhẹ hơn**: thu hẹp biên độ sáng/tối, mỗi ảnh có tối đa 1 phép làm suy giảm (mờ / phân giải thấp / JPEG). Nếu ảnh sau augmentation mất quá nửa độ tương phản thì dùng lại ảnh chưa làm xấu.
- Giữ toàn bộ cải tiến của v2: transfer 100% trọng số, crop có đệm, train 2 pha, resume.

## Kết quả (tập Test, 5.524 ảnh)
| Nhóm | Acc v2 | **Acc v3** | CER v3 |
| :--- | ---: | ---: | ---: |
| 1 dòng | 94.93% | 91.11% | 0.0142 |
| 2 dòng ô tô | 87.92% | **92.84%** | 0.0128 |
| xe máy | 64.94% | **81.59%** | 0.0292 |
| **Tổng** | 88.31% | **90.31%** | 0.0161 |

Best Val Acc: 90.32% (epoch 83). Xe máy tăng khoảng 17 điểm, nhưng biển 1 dòng **giảm gần 4 điểm**.

## Hạn chế và lỗi kỹ thuật
- **Biển 1 dòng bị giảm độ chính xác.** Ảnh 94 và ảnh 188 dùng chung thống kê BatchNorm (running mean/var). Mỗi batch chỉ có một loại chiều rộng nên bộ thống kê bị kéo qua lại, lúc đánh giá cả hai loại đều bị chuẩn hóa lệch. Sửa ở v4 bằng `WidthBatchNorm2d`.
- **Tin hoàn toàn nhãn 1/2 dòng trong tên file.** Biển 1 dòng bị gán nhãn 2 dòng sẽ bị cắt đôi thành ảnh rác, và ngược lại.
- **Cắt ngang cố định ở `h/2`.** Biển xe máy nghiêng thì đường cắt xuyên qua chữ.
- Chia tập chỉ theo chuỗi biển số. Ảnh giống hệt từng byte nhưng khác nhãn có thể rơi vào cả train và test.
- Ảnh lỗi (bbox sai, nhãn lạ) bị loại khỏi cả tập Test, nên số liệu Test hơi lạc quan.
- Không biết lỗi tập trung ở loại ảnh khó nào; lỗi pipeline chỉ lộ ra sau nhiều giờ train.

Các lỗi này được sửa ở [v4](../v4/README.md).
