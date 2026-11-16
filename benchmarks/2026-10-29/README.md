# Benchmark YOLO / CSRT / ByteTrack — 29/10/2026

Kết quả lịch sử của 9 lượt chạy trên RTX 3050 Laptop, không có OCR. Xem [báo cáo và giới hạn phép đo](../../docs/benchmark_yolo_csrt_bytetrack.md).

- `comparison.json`: môi trường, cấu hình, SHA256 video/trọng số, lệnh chạy và số liệu tổng hợp.
- `test*_yolo.json`, `test*_csrt.json`, `test*_bytetrack.json`: thống kê gốc từng lượt.
- `*_tracks.jsonl`: bbox/ID ByteTrack theo frame.
- `*_comparison.jpg`: ảnh đối chiếu ở các frame mẫu.

JSON gốc được giữ nguyên, gồm đường dẫn tuyệt đối của máy đã đo. Video nguồn/đầu ra không được đưa lên Git. Khi tái chạy, cần tự đặt ba video gốc vào `data/test/` và dùng đúng `weights/best.pt`; SHA256 trong manifest giúp kiểm tra đầu vào có trùng không.

```powershell
python benchmarks/2026-10-29/run_comparison.py
```

Script tái chạy cần CUDA và lưu lượt đo mới vào `data/test/diagnostics/comparison_<timestamp>/`, không ghi đè số liệu lịch sử trong thư mục này. Kết quả mới có thể khác do phần cứng, thư viện hoặc đầu vào. Video nguồn khoảng 30 FPS và lượt đo cũ có pacing; không diễn giải FPS phát là FPS tối đa hoặc số box/ID là accuracy.
