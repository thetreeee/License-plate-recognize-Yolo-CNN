# Detect biển số bằng YOLOv8n

Chạy các lệnh sau trong PowerShell tại thư mục gốc project.

## 1. Kích hoạt môi trường ảo

```powershell
.\venv\Scripts\Activate.ps1
```

Nếu môi trường của bạn tên `.venv`, dùng `.\.venv\Scripts\Activate.ps1`.

## 2. Chạy detect

Video test:

```powershell
python scripts/realtime_lp_detect.py --mode detect --source "data/test/test2.mp4" --imgsz 640
```

Webcam:

```powershell
python scripts/realtime_lp_detect.py --mode detect --source 0 --imgsz 640
```

Trọng số mặc định: `weights/best.pt`. Nhấn **q** để thoát demo; chạy `deactivate` để thoát môi trường ảo.
