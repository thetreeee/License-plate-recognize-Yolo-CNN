# OCR v5 — LPRNet hoàn thiện cho biển số Việt Nam

Notebook: [`VN_ALPR_LPRNet_v5_TrainReview.ipynb`](VN_ALPR_LPRNet_v5_TrainReview.ipynb) · Script suy luận: [`lprnet_v5_infer.py`](lprnet_v5_infer.py) · Kết quả và trọng số: [`lprnet_v5/`](lprnet_v5/)

Bước OCR trong pipeline **YOLOv8n (phát hiện biển) → LPRNet (đọc ký tự)**. Mô hình có 307K tham số, chạy 2.4 ms/biển trên GPU (batch 1, khoảng 400 FPS).

---

## 1. Tổng quan v5

### Kết quả trên tập Test (5.649 ảnh, giữ cả ảnh khó và ảnh lỗi)
| Nhóm | v2 | v3 | v4 | **v5 (greedy)** | **v5 (ràng buộc)** | CER v5 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 dòng | 94.93% | 91.11% | 95.15% | 93.67% | **94.66%** | 0.0096 |
| 2 dòng ô tô | 87.92% | 92.84% | 91.81% | 95.57% | **96.56%** | 0.0065 |
| xe máy | 64.94% | 81.59% | 85.35% | 88.01% | **90.56%** | 0.0183 |
| **Tổng** | 88.31% | 90.31% | 92.58% | 93.49% | **94.71%** | 0.0100 |

- Trên ảnh sạch (không cờ chất lượng, 5.625 ảnh): **94.97%**, trong đó xe máy đạt 91.65%.
- Giải mã ràng buộc so với greedy: **sửa đúng 74 ảnh, làm sai 5 ảnh** vốn đúng. Có 12 ảnh không tìm được chuỗi hợp lệ và dùng lại kết quả greedy.
- Best Val Acc (greedy): 93.59% ở epoch 80.
- v2 và v3 được đánh giá trên 5.524 ảnh Test, do còn loại ảnh lỗi khỏi Test.

### Cải tiến chính so với v1–v4
| Vấn đề | Xử lý | Có từ |
| :--- | :--- | :--- |
| Kiến trúc tự viết chỉ nạp được 8/47 tensor pre-trained | Dùng đúng LPRNet sirius-ai, transfer **100%** trọng số, tầng phân lớp được ghép theo ký tự | v2 |
| Biển 2 dòng bị nén vào 94×24 (~9 bước CTC mỗi dòng), mất số lặp | Mỗi dòng resize riêng về 94×24, ghép ngang thành **188×24** (42 bước CTC) | v3 |
| Ảnh 94 và 188 dùng chung thống kê BatchNorm | `WidthBatchNorm2d`: thống kê chạy riêng cho từng độ rộng | v4 |
| Cắt ngang cố định ở `h/2` xuyên qua chữ khi biển nghiêng | **Tìm khe nghiêng** giữa 2 dòng, nắn thẳng từng dòng bằng `cv2.remap` | v4 |
| Nhãn 1/2 dòng sai, ảnh trùng byte rơi vào cả train và test | Sửa layout, chia tập theo nhóm liên thông (biển số hoặc sha256) | v4 |
| Ảnh lỗi / nhãn sai trong train phạt sai mô hình | **Rà soát train bằng checkpoint v4** (Bước 10.1), sửa qua `train_fixes.csv`: 112 ảnh đổi nhãn, 32 ảnh bị loại | **v5** |
| Không có chữ **R** (biển rơ-moóc), 446 ảnh bị loại | Bảng ký tự 31 ký tự + blank (32 lớp); trọng số lớp R lấy từ trọng số tác giả | **v5** |
| Regex không nhận seri 2 chữ + 6 số (29-MD1 524.62) | Regex `^\d{2}[A-Z]{1,2}\d{4,6}$` | **v5** |
| Greedy thêm ký tự thừa (`890B199639`, `29H1174111`) và nhầm chữ ↔ số | **Giải mã CTC có ràng buộc định dạng** (beam search, ngữ pháp học từ nhãn train) | **v5** |
| Chữ hiếm (Z 114, X 224, K 247, G 343 mẫu so với A 11.111) học kém | **Lấy mẫu lặp ảnh có chữ hiếm** ×√(1000 / số lần), tối đa ×3; bảng độ chính xác theo từng chữ | **v5** |

### Giải mã có ràng buộc định dạng
Greedy chọn ký tự xác suất cao nhất ở từng bước nên có thể ra chuỗi không thể là biển số. v5 dùng prefix beam search CTC (beam 10, xét 6 ký tự cao nhất mỗi bước) và **chỉ giữ chuỗi hợp lệ** theo ngữ pháp học từ nhãn train:
- 2 số đầu là mã tỉnh có trong train (65 mã).
- Seri 1–2 chữ có trong train (35 seri, 14 seri 2 chữ).
- Số chữ số phía sau là 4, 5 hoặc 6, tính riêng cho biển 1 dòng và 2 dòng.

Nếu không có chuỗi hợp lệ nào, kết quả quay về greedy + `fix_vn_plate`. Giải mã tốn 0.6 ms (1 dòng) và 1.2 ms (2 dòng) mỗi biển trên CPU.

---

## 2. Bài học và rút kinh nghiệm

1. **Kiểm tra trọng số thực sự được nạp.** v1 tưởng là transfer learning nhưng chỉ 8/47 tensor khớp, thực chất là train từ đầu. Từ v2 notebook in số tensor nạp được và đánh giá zero-shot trước khi train.
2. **Độ phân giải theo thời gian của CTC là giới hạn cứng.** Ở v2, mỗi dòng của biển 2 dòng chỉ có khoảng 9 bước CTC, không đủ cho 5 chữ số cộng blank giữa các số lặp. Tăng augmentation hay số epoch không sửa được; phải đổi cách đưa ảnh vào (188×24 ở v3). Xe máy tăng từ 65% lên 82% chỉ nhờ thay đổi này.
3. **Batch khác độ rộng thì BatchNorm cũng phải tách.** v3 tăng mạnh biển 2 dòng nhưng làm biển 1 dòng giảm gần 4 điểm, vì hai loại ảnh dùng chung running mean/var. `WidthBatchNorm2d` ở v4 lấy lại phần mất này.
4. **Cắt ngang cố định không chịu được biển nghiêng.** Đường `h/2` xuyên qua chữ ở biển xe máy chụp chéo. Tìm khe ít nét mực nhất (cho phép nghiêng) rồi nắn thẳng từng dòng là cách sửa tận gốc. Cắt **chồng lấn 3%** tránh mất chân hoặc đầu chữ.
5. **Bias và nhiễu của tập dữ liệu ảnh hưởng nhiều hơn kiến trúc.**
   - Nhãn sai (biển ghi 98-M2 nhưng nhãn 98N26985) dạy mô hình đọc sai. Rà soát train theo CTC loss giảm dần làm lộ các ảnh này nhanh nhất.
   - Lọc nhãn quá chặt (thiếu chữ R, regex không nhận seri 2 chữ) âm thầm loại cả một loại biển khỏi train. Mô hình không bao giờ học được, và trên Test các ảnh đó đạt 0%.
   - Phân bố chữ cái lệch nặng (A gấp ~100 lần Z) cần lấy mẫu lặp có giới hạn. Lặp quá mạnh dễ overfit vài ảnh hiếm, nên hệ số dùng √ và tối đa ×3.
6. **Đánh giá phải trung thực trước khi tối ưu.** v2–v3 loại ảnh lỗi khỏi Test nên số liệu hơi lạc quan. Từ v4, Test giữ mọi mẫu, ảnh hỏng tính là sai, và có bảng theo cờ chất lượng để biết lỗi nằm ở đâu.
7. **Overfitting và loss.** Lịch LR 1e-4 + Cosine 40 epoch ở v1 dừng khi loss còn giảm. Từ v2 dùng 3 epoch đóng băng rồi 80 epoch AdamW + OneCycleLR (max 1e-3), Dropout 0.3, weight decay 1e-4, augmentation mỗi ảnh tối đa 1 phép làm suy giảm. Val Acc v5 ổn định quanh 93.5% ở các epoch cuối, không giảm lại. Checkpoint được chọn theo Val, không theo Test.
8. **Phát hiện lỗi pipeline sớm.** Sanity check (học thuộc 32 ảnh), `CTCLoss(zero_infinity=False)` và dấu vân tay dữ liệu trong checkpoint chặn được lỗi ngay từ đầu thay vì sau nhiều giờ train.
9. **Ràng buộc định dạng đáng giá khi định dạng thật sự cố định.** Biển VN có cấu trúc rõ ràng, nên ràng buộc lúc giải mã tăng 1.2 điểm mà không phải train lại. Cái giá là biển có mã tỉnh hoặc seri chưa từng gặp trong train sẽ bị ép thành chuỗi khác; ngữ pháp cần cập nhật khi có seri mới.

### Hạn chế còn lại
- **Phân loại 1/2 dòng khi chạy thật.** LPRNet không tự phân loại; nó cần biết trước để chọn đầu vào 94 hay 188. Đoán theo tỉ lệ bbox (< 1.50 là 2 dòng) đúng 95.97% mẫu, nhưng trên 1000 ảnh Test, Acc giảm từ **93.70%** (dùng nhãn thật) xuống **90.70%** (đoán từ bbox). Hướng sửa: YOLO detect 2 lớp `plate_1row` / `plate_2row`, hoặc thêm bộ phân loại nhỏ trên ảnh crop.
- CPU batch 1 khá chậm (107 ms biển 1 dòng, 224 ms biển 2 dòng). Muốn ≥ 20 FPS cần GPU, gom batch, hoặc xuất ONNX/OpenVINO.
- Val trong lúc train dùng greedy; giải mã ràng buộc chỉ dùng cho Test và suy luận.

---

## 3. Hướng dẫn sử dụng trọng số v5 (Inference Guide)

### 3.1 File checkpoint
| File | Nội dung | Dùng khi |
| :--- | :--- | :--- |
| [`lprnet_v5/best_lprnet_vn_v5.pth`](lprnet_v5/best_lprnet_vn_v5.pth) (1.3 MB) | `state_dict` của epoch có Val Acc cao nhất | **Suy luận** |
| [`lprnet_v5/last_lprnet_vn_v5.pth`](lprnet_v5/last_lprnet_vn_v5.pth) (3.7 MB) | dict `{"epoch", "phase", "model", "optimizer", "scheduler", "history", "fingerprint", ...}` | Train tiếp (`WEIGHTS_MODE = "resume"`) |

Tải trực tiếp:
```bash
curl -L -o best_lprnet_vn_v5.pth https://github.com/thetreeee/License-plate-recognize-Yolo-CNN/raw/HoangAnh_Feature/OCR/v5/lprnet_v5/best_lprnet_vn_v5.pth
```

Checkpoint có **32 lớp** (31 ký tự + blank). Checkpoint v2–v4 có 31 lớp (chưa có R) và không nạp trực tiếp được vào kiến trúc v5; notebook v5 có hàm `load_any_checkpoint` để ghép lại.

### 3.2 Chạy nhanh bằng script
Cài đặt: `pip install torch opencv-python numpy`.

```bash
# khung hình gốc + bbox từ YOLO (xmin ymin xmax ymax), tự đoán 1/2 dòng
python lprnet_v5_infer.py --image frame.jpg --bbox 120 340 260 420

# ảnh đã crop sẵn, chỉ định biển 2 dòng
python lprnet_v5_infer.py --image plate.jpg --layout two_rows
```

### 3.3 Dùng trong code Python
```python
import cv2
from lprnet_v5_infer import load_model, recognize_plate, PlateGrammar

net = load_model("lprnet_v5/best_lprnet_vn_v5.pth", device="cuda")   # hoặc "cpu"
grammar = PlateGrammar("lprnet_v5/plate_grammar.json")   # không có file -> ngữ pháp tổng quát

frame = cv2.imread("frame.jpg", cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
for xmin, ymin, xmax, ymax in yolo_boxes:                 # bbox biển số từ YOLOv8n
    plate, two_rows = recognize_plate(net, frame, (xmin, ymin, xmax, ymax),
                                      is_two_rows=None,   # None: đoán theo tỉ lệ bbox; nên truyền True/False nếu YOLO có lớp 1/2 dòng
                                      grammar=grammar, device="cuda")
    print(plate, "2 dòng" if two_rows else "1 dòng")
```

Các bước bên trong `recognize_plate`:
```python
crop = crop_plate(frame, bbox)                          # 1. crop có đệm 3% ngang, 6% dọc
two = (xmax - xmin) / (ymax - ymin) < 1.50              # 2. đoán 1/2 dòng (nếu không truyền is_two_rows)
x = preprocess(crop, two)                               # 3. 1 dòng: resize 94x24 | 2 dòng: tách khe nghiêng, nắn thẳng,
                                                        #    mỗi dòng 94x24, ghép ngang 188x24; chuẩn hóa (x - 127.5) / 128
logits = net(x)                                         # 4. [T, 1, 32], T = 18 (94) hoặc 42 (188)
plate = decode_constrained(logits.log_softmax(2)[:, 0].numpy(), two, grammar)   # 5. beam search có ràng buộc
if plate is None:
    plate = fix_vn_plate(decode_greedy(logits))         # 6. không có chuỗi hợp lệ -> greedy + sửa chữ <-> số
```

### 3.4 Xuất ngữ pháp biển số từ notebook
Giải mã ràng buộc đạt đúng số liệu ở trên khi dùng ngữ pháp học từ nhãn train (65 mã tỉnh, 35 seri). Chạy ô này trong notebook v5 sau Bước 10, rồi đặt file vào `lprnet_v5/`:
```python
json.dump({"provinces": sorted(PROVINCES), "series": sorted(SERIES),
           "digit_lens": {"one_row": sorted(DIGIT_LENS[False]), "two_rows": sorted(DIGIT_LENS[True])}},
          open(f"{SAVE_DIR}/plate_grammar.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
```
Nếu không có file, `PlateGrammar()` dùng ngữ pháp tổng quát (2 số + 1–2 chữ + 4–6 số). Cách này vẫn chặn được ký tự thừa và nhầm chữ ↔ số, nhưng không loại được mã tỉnh hoặc seri không tồn tại.

### 3.5 Lưu ý cấu hình tiền xử lý
| Tham số | Giá trị | Ghi chú |
| :--- | :--- | :--- |
| Kích thước một dòng | **94×24** (W×H) | Đúng đầu vào LPRNet gốc |
| Biển 2 dòng | **188×24** = dòng trên \| dòng dưới | Mỗi dòng resize riêng về 94×24, không nén chung 2 dòng vào 94×24 |
| Đệm quanh bbox | `PAD_X = 0.03`, `PAD_Y = 0.06` | Tỉ lệ theo kích thước bbox |
| Chồng lấn khi tách dòng | `SPLIT_OVERLAP = 0.03` | Mỗi dòng lấn qua khe 3% chiều cao |
| Ngưỡng 1/2 dòng | `rộng / cao < 1.50` là 2 dòng | Dò trên dữ liệu ở Bước 4.3; tính trên bbox **trước khi** đệm |
| Resize | `INTER_AREA` khi thu nhỏ, `INTER_LINEAR` khi phóng to | |
| Chuẩn hóa | `(pixel - 127.5) × 0.0078125`, ảnh **BGR**, tensor `[B, 3, 24, W]` | Giống LPRNet gốc |
| Đọc ảnh | `cv2.IMREAD_IGNORE_ORIENTATION` | Bbox VNLP theo pixel gốc, không xoay theo EXIF |
| Batch | Mỗi batch chỉ chứa **một** loại độ rộng | `WidthBatchNorm2d` chọn thống kê theo độ rộng của cả batch |

---

## 4. Huấn luyện lại
1. Mở notebook trên Colab (GPU), chỉnh đường dẫn Drive ở Bước 1, rồi *Run all*.
2. Lần chạy đầu với `REVIEW_MODE = True`, notebook dừng ở **Bước 10.1** sau khi checkpoint v4 đọc lại toàn bộ train và ghi `train_review.csv` cùng các ảnh `train_review_pageNNN.png` (không đẩy lên GitHub, notebook tự sinh lại).
3. Điền `drop` / `relabel` / `one_row` / `two_rows` vào [`train_fixes.csv`](lprnet_v5/train_fixes.csv) (bản hiện tại: 112 relabel, 32 drop), đặt `REVIEW_MODE = False`, rồi *Run all* để train.
4. Colab bị ngắt: đặt `WEIGHTS_MODE = "resume"`. Checkpoint chặn resume nếu dữ liệu đã thay đổi.

## 5. Cấu trúc thư mục
```
v5/
├── VN_ALPR_LPRNet_v5_TrainReview.ipynb   # notebook train + đánh giá (có output)
├── lprnet_v5_infer.py                    # suy luận độc lập
└── lprnet_v5/
    ├── best_lprnet_vn_v5.pth             # trọng số suy luận
    ├── last_lprnet_vn_v5.pth             # checkpoint train tiếp
    ├── test_results.json                 # số liệu Test (greedy / fix / ràng buộc / theo cờ)
    ├── test_predictions.csv              # dự đoán từng ảnh Test
    ├── train_history.json                # loss, Val Acc từng epoch
    ├── train_fixes.csv                   # sửa nhãn tập train
    ├── train_review.csv                  # kết quả rà soát train bằng checkpoint v4
    ├── layout_suspects.csv               # ảnh nghi sai nhãn 1/2 dòng
    ├── lprnet_v5_training_curves.png
    └── test_errors_{1_dong,2_dong_oto,xe_may}.png
```
