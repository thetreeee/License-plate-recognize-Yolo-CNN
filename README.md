# 🚗 License Plate Recognition — YOLOv8n + OCR

Nhận diện biển số xe sử dụng **YOLOv8n** để detect biển số, kết hợp **OCR** để đọc ký tự.

## 📁 Cấu trúc project

```
├── weights/          # Trọng số model đã train
│   └── best.pt
├── data/             # Dataset (images + labels)
│   └── data.rar
├── notebooks/        # Jupyter Notebooks (training pipeline)
│   └── YOLOv8n_License_Plate_Detection_Pipeline.ipynb
├── scripts/          # Scripts chạy inference
│   └── realtime_lp_detect.py
├── .gitignore
├── .gitattributes    # Git LFS config
└── README.md
```

## ⚙️ Cài đặt

```bash
# 1. Clone repo
git clone https://github.com/thetreeee/License-plate-recognize-Yolo-CNN.git
cd License-plate-recognize-Yolo-CNN

# 2. Pull file lớn (weights + data) qua Git LFS
git lfs pull

# 3. Cài dependencies
pip install -r requirements.txt
```

## 🚀 Sử dụng

### Detect biển số real-time (Webcam)

```bash
python scripts/realtime_lp_detect.py --weights weights/best.pt
```

**Tùy chọn:**
| Argument | Default | Mô tả |
|----------|---------|-------|
| `--weights` | `weights/best.pt` | Đường dẫn trọng số |
| `--source` | `0` | Index webcam (0, 1, ...) |
| `--conf` | `0.1` | Ngưỡng confidence |
| `--iou` | `0.45` | Ngưỡng NMS IoU |
| `--imgsz` | `1280` | Kích thước ảnh inference |

> Nhấn **`q`** để thoát.

### Training model

Mở notebook `notebooks/YOLOv8n_License_Plate_Detection_Pipeline.ipynb` trên **Google Colab** (GPU T4).

## 📝 Thêm data / model mới

- **Thêm trọng số**: Bỏ file `.pt` vào folder `weights/`
- **Thêm dataset**: Bỏ file vào folder `data/`
- **Thêm script**: Bỏ file `.py` vào folder `scripts/`

## Test toàn pipeline YOLO → MobileNetV3-Small → CRNN-CTC

Script mới `scripts/recognize_license_plate.py` dùng các file sau (không thay script detection cũ):

```text
weights/best.pt          # YOLO detection
weights/layout_best.pt   # MobileNet: one_row / two_rows
weights/crnn_best.pt     # CRNN đọc chuỗi ký tự
```

Cài dependencies bằng `python -m pip install -r requirements.txt`; nếu checkpoint chỉ là Git LFS pointer, chạy `git lfs pull`.

```powershell
# Ảnh: không mở cửa sổ mặc định, tự lưu ảnh có kết quả + CSV/JSON
python scripts/recognize_license_plate.py --source "D:\Downloads\test.jpg" --device cpu --save-crops

# Video: auto chọn CUDA nếu có, bấm q/Esc để dừng preview
python scripts/recognize_license_plate.py --source "D:\Downloads\test.mp4" --device auto --show

# Webcam, giới hạn 300 frame nếu chỉ cần thử nhanh
python scripts/recognize_license_plate.py --source 0 --device auto --show --max-frames 300

# Xem tất cả tùy chọn
python scripts/recognize_license_plate.py --help
```

Mỗi lần chạy tạo một thư mục mới trong `outputs/lpr/`, không ghi đè ảnh/video gốc:

- `annotated.png` hoặc `annotated.mp4`: bbox, chuỗi OCR, layout và cảnh báo cần kiểm tra.
- `detections.csv`: mỗi bbox một dòng, gồm confidence YOLO/layout, OCR, `split_ok` và lỗi.
- `frames.csv`: cả frame không có detection, thời gian YOLO/crop/layout/xử lý hàng/CRNN.
- `summary.json`: thông tin checkpoint, thiết bị, latency trung bình/P95 và processing FPS.
- `crops/` khi bật `--save-crops`: crop RGB, ảnh một dòng và input CRNN; mặc định tối đa 100 biển (`--debug-limit`).

Script lấy layout từ MobileNet, tách/ghép khi cần, sau đó dùng đúng input/charset của CRNN checkpoint. Không tự sửa O/0 hoặc dùng filename làm nhãn. OCR các biển trong một frame được gom batch. Mặc định FP32; chưa có tracking, bỏ frame hoặc đồng thuận nhiều frame. `--conf` là ngưỡng YOLO, không phải confidence OCR.

`Infer FPS` trên ảnh chỉ tính pipeline; `processing_fps` trong JSON tính cả đọc frame/vẽ/lưu (không gồm nạp model và warmup). Video đầu ra giữ FPS nguồn, không giữ âm thanh. Máy CPU có thể xử lý video chậm hơn tốc độ phát nhưng vẫn lưu đủ frame đã xử lý. Chưa có ground truth nên script không tự tính accuracy; text không rỗng không đồng nghĩa đọc đúng. Kết quả thư mục `outputs/` đã được gitignore.

## 🛠️ Tech Stack

- **YOLOv8n** (Ultralytics) — Object Detection
- **OpenCV** — Video capture & display
- **Python 3.x**
