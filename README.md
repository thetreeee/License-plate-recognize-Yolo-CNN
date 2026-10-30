# Detect và tracking biển số — YOLOv8n + ByteTrack

Tiến độ hiện tại: detect bbox và gán ID qua các frame; chưa tích hợp OCR vào bản chạy này. YOLO chạy mỗi frame, ByteTrack liên kết ID. Bản CSRT cũ được giữ để đối chiếu.

Chạy trong PowerShell tại thư mục gốc project:

```powershell
# Chỉ cần tạo môi trường khi chưa có
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

# Video cục bộ
python scripts/realtime_lp_bytetrack.py --source "data/test/test2.mp4"

# Webcam
python scripts/realtime_lp_bytetrack.py --source 0
```

Nếu đã có môi trường thì chỉ cần activate; nếu tên `.venv`, dùng `.\.venv\Scripts\Activate.ps1`. Máy dùng GPU NVIDIA cần PyTorch có hỗ trợ CUDA; mặc định script tự chọn GPU/CPU.

Trọng số: `weights/best.pt` (chạy `git lfs pull` nếu mới clone). Nhấn **q** để thoát, `deactivate` để thoát môi trường ảo. Video trong `data/test` là dữ liệu cục bộ, không đi cùng repository.

Mặc định `imgsz=640`, confidence YOLO `0.1`; ngưỡng tạo ID `0.6` trong `configs/bytetrack.yaml`. Thêm `--mode detect` để chạy YOLO thuần, hoặc `--output "data/test/result.mp4"` để lưu video.

[Chi tiết cách chạy](docs/bytetrack_demo.md) · [Benchmark YOLO / CSRT / ByteTrack](docs/benchmark_yolo_csrt_bytetrack.md)
