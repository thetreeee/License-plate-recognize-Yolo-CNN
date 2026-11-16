# YOLO + ByteTrack

Runner hiện tại: `scripts/realtime_lp_bytetrack.py`. Giữ bản CSRT để đối chiếu và dùng lại trọng số `weights/best.pt`; không cần train lại. Chưa tích hợp OCR hoặc chia vùng frame để chạy lại YOLO.

## Luồng xử lý

```text
Video/webcam → YOLO trên từng frame → ByteTrack ghép ID → vẽ và hiển thị
```

Model được load và warmup một lần trước khi phát. Gọi `model.track(..., persist=True)` để giữ trạng thái giữa các frame liên tiếp. Không dùng CSRT, worker YOLO nền hay buffer bù chuyển động trong bản mới. YOLO vẫn được gọi trên mọi frame.

## Chạy trong PowerShell, tại thư mục gốc project

```powershell
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/realtime_lp_bytetrack.py --source "data/test/test2.mp4"
```

Nếu môi trường tên `.venv`, thay đường dẫn activate tương ứng. Nhấn `q` để thoát. Đổi `--source 0` để dùng webcam.

Video file mặc định phát theo FPS nguồn. `--unpaced` bỏ thời gian chờ để đo throughput; tùy chọn này không làm model chạy nhanh hơn. FPS trên cửa sổ là tốc độ xử lý/phát quan sát được; `processing_fps` trong JSON loại thời gian chờ và model warmup. CPU dùng 4 thread PyTorch và 1 thread OpenCV, tương tự bản CSRT hiện tại.

## Confidence và tạo ID

`--conf` là ngưỡng YOLO lọc detection **trước** tracker; mặc định `0.1`. Cấu hình `configs/bytetrack.yaml` dùng:

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `track_low_thresh` | `0.1` | Cho phép detection yếu tham gia bước ghép thứ hai |
| `track_high_thresh` | `0.5` | Ngưỡng bước ghép confidence cao |
| `new_track_thresh` | `0.6` | Ngưỡng tối thiểu khởi tạo ID mới |
| `track_buffer` | `30` | Giữ thông tin track mất dấu trong thời gian ngắn; không vẽ box mất dấu |
| `match_thresh` | `0.8` | Ngưỡng chi phí ghép của ByteTrack, không phải IoU trực tiếp như bản CSRT |

Detection yếu có thể duy trì ID đã có nhưng không tự khởi tạo ID mới. Script chỉ vẽ box có ID tracker trả về, tránh việc bản Ultralytics cũ trả lại proposal chưa được xác nhận khi không có track.

Đặt `--conf` cao hơn `track_low_thresh` sẽ lọc bớt detection trước khi ByteTrack nhận được, làm hạn chế bước cứu track confidence thấp. Ví dụ nhận nhầm logo không được giải quyết bằng cách hạ mọi ngưỡng; cần chỉnh phối hợp ngưỡng tạo ID và kiểm tra dữ liệu.

## Lưu kết quả

```powershell
python scripts/realtime_lp_bytetrack.py --source "data/test/test2.mp4" --output "data/test/bytetrack_result.mp4" --stats-json "data/test/bytetrack_stats.json" --tracks-jsonl "data/test/bytetrack_tracks.jsonl"
```

JSONL chứa số frame, timestamp theo video, ID, box xyxy và confidence. Có thể dùng để kiểm tra hoặc bổ sung quản lý crop/OCR theo ID. `IDs seen` đếm số ID đã xuất hiện, **không phải** số xe duy nhất: ID có thể đổi hoặc bị ghép nhầm. Không dùng số box/ID làm độ chính xác.

## Đối chiếu hiệu năng

Giữ cùng video, weights, `imgsz`, confidence, device, số frame và chế độ hiển thị/ghi video. Có thể dùng mức confidence chung `0.5` để đo chi phí; lưu ý phép đo này hạn chế bước ghép confidence thấp của ByteTrack. Sau đó kiểm tra chất lượng bằng cấu hình vận hành `--conf 0.1` riêng.

```powershell
python scripts/realtime_lp_bytetrack.py --source "data/test/test2.mp4" --mode detect --conf 0.5 --no-display --unpaced --stats-json "data/test/yolo_stats.json"
python scripts/realtime_lp_bytetrack.py --source "data/test/test2.mp4" --conf 0.5 --no-display --unpaced --stats-json "data/test/bytetrack_stats.json"
python scripts/realtime_lp_detect.py --source "data/test/test2.mp4" --conf 0.5 --imgsz 640 --no-display --unpaced --stats-json "data/test/csrt_stats.json"
```

Chế độ async CSRT có thể loại nhiều kết quả khi đọc file không chờ; cần chạy thêm cả ba ở chế độ phát theo FPS nguồn (bỏ `--unpaced`) để xem độ trễ và chất lượng box. `p95`/`max` là thời gian vòng xử lý; FPS cao không bảo đảm ID chính xác. Bộ nhớ CUDA trong JSON là peak allocated/reserved của tiến trình PyTorch, không phải toàn bộ bộ nhớ GPU của máy.

ByteTrack không có re-identification theo ngoại hình hay bù chuyển động camera trong cấu hình này. Các video cầm tay lia nhanh có thể làm mất hoặc đổi ID. Cần dữ liệu gán ID/bounding box để đo chất lượng định lượng; chưa thể kết luận từ FPS và số ID.

Kết quả 9 lượt so sánh đã lưu trong [báo cáo benchmark](benchmark_yolo_csrt_bytetrack.md), kèm JSON và ảnh đối chiếu tại `benchmarks/2026-10-29/`. Video nguồn/đầu ra vẫn nằm cục bộ trong `data/`.
