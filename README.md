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

## 🛠️ Tech Stack

- **YOLOv8n** (Ultralytics) — Object Detection
- **OpenCV** — Video capture & display
- **Python 3.x**
