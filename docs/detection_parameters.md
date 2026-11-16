# Bảng tham số YOLO + CSRT

Nguồn cấu hình: `scripts/realtime_lp_detect.py`, hàm `parse_args()`.

Các giá trị dưới đây là mặc định tại thời điểm xuất tài liệu. Khi truyền tham số lúc chạy, giá trị được truyền sẽ thay thế giá trị mặc định cho lượt chạy đó.

| Tham số | Mặc định | Tác dụng |
|---|---|---|
| `--weights` | `weights/best.pt` | Đường dẫn trọng số YOLO đã train. |
| `--source` | `0` | Nguồn đầu vào: `0` là webcam mặc định; có thể truyền đường dẫn video hoặc URL. |
| `--conf` | `0.6` | Ngưỡng confidence tối thiểu để giữ detection. Tăng thường giảm nhận nhầm nhưng có thể tăng bỏ sót. Confidence không phải tỷ lệ chính xác tổng thể của model. |
| `--iou` | `0.45` | Ngưỡng IoU của NMS, dùng loại các detection chồng lấp. Tăng ngưỡng cho phép giữ nhiều box chồng lấp hơn. |
| `--imgsz` | `640` | Kích thước ảnh đưa vào YOLO. Tăng giúp giữ chi tiết biển số nhỏ nhưng tốn tính toán hơn. Không đổi kích thước video hiển thị. |
| `--device` | `auto` | Thiết bị chạy YOLO: `auto` tự chọn, `cpu` dùng CPU, `0` dùng GPU CUDA đầu tiên nếu khả dụng. |
| `--mode` | `track` | `track`: YOLO kết hợp CSRT. `detect`: YOLO chạy trên mọi frame, không tracking. |
| `--sync` | Tắt | Bật để chạy YOLO ngay trên luồng chính nhằm đối chiếu với chạy nền. Chế độ `detect` luôn chạy đồng bộ. |
| `--detect-interval` | `10` | Với async, khoảng cách gửi yêu cầu YOLO khi đang tracking, tính theo frame; không bảo đảm mọi yêu cầu đều được xử lý do worker có thể bận. Với sync, là lịch detect định kỳ và có thể detect sớm khi mất dấu. Giảm giá trị để kiểm tra thường xuyên hơn, nhưng tăng tải. |
| `--search-interval` | `5` | Trong async, nhịp gửi yêu cầu tìm lại khi không có track hiển thị; cũng giới hạn tần suất yêu cầu khi vừa mất dấu. |
| `--match-iou` | `0.2` | Ngưỡng chồng lấp để ghép detection với track cũ và giữ ID. Tăng khiến việc ghép khắt khe hơn, có thể tạo ID mới nhiều hơn. Giảm có thể ghép nhầm đối tượng. |
| `--max-age` | `300` | Số frame giữ thông tin một track kể từ lần YOLO xác nhận gần nhất. Không có nghĩa box luôn hiển thị suốt thời gian này; track mất dấu có thể bị ẩn trước đó. |
| `--appearance-threshold` | `0.35` | Ngưỡng tương đồng với mẫu ảnh biển số để kiểm tra bám nhầm. Tăng sẽ loại track khác mẫu mạnh hơn, nhưng dễ mất dấu khi ảnh mờ hoặc đổi góc. |
| `--buffer-size` | `32` | Số frame gần nhất được lưu để bù chuyển động cho kết quả YOLO đến trễ. Tăng sẽ tốn thêm RAM, phụ thuộc độ phân giải video. |
| `--max-result-age` | `20` | Loại kết quả async nếu frame nguồn cũ hơn frame hiện tại quá số frame này. |
| `--max-result-seconds` | `1.0` | Loại kết quả async nếu đã quá số giây này kể từ lúc đọc frame nguồn. Giảm chỉ làm loại kết quả chậm nhiều hơn, không tăng tốc inference. |
| `--cpu-threads` | `4` | Số thread CPU cho PyTorch. Tăng không chắc nhanh hơn vì có thể tranh tài nguyên với tracking. OpenCV được cấu hình riêng cố định một thread, không đổi theo tham số này. |
| `--csrt-quality` | `fast` | `fast`: cấu hình CSRT nhẹ. `full`: cấu hình OpenCV đầy đủ, tốn CPU hơn và có thể bám tốt hơn trong một số cảnh. OpenCV legacy không hỗ trợ Params sẽ dùng cấu hình đầy đủ. |
| `--unpaced` | Tắt | Bật để đọc video file nhanh nhất có thể thay vì theo FPS nguồn. Có thể khiến kết quả async không kịp trả về trước khi frame đã đi qua. |
| `--no-display` | Tắt | Bật để không mở cửa sổ hiển thị. Vẫn có thể ghi video đầu ra; video file vẫn được phát theo FPS nguồn trừ khi có `--unpaced`. |
| `--max-frames` | `0` | Giới hạn số frame xử lý. `0` nghĩa là không giới hạn, chạy đến hết nguồn hoặc khi thoát. |
| `--output` | Không có | Đường dẫn lưu video MP4 có bounding box, ID và thông số hiển thị. Phải khác đường dẫn video đầu vào. |
| `--stats-json` | Không có | Đường dẫn lưu thống kê hiệu năng và worker dưới dạng JSON. |
| `-h`, `--help` | — | Hiển thị hướng dẫn các tham số rồi thoát. |

## Điều kiện và lưu ý

- `1 <= detect-interval <= max-age`.
- `1 <= max-result-age < buffer-size`.
- `0 < match-iou <= 1` và `0 < appearance-threshold <= 1`.
- `search-interval`, `cpu-threads` và `max-result-seconds` phải lớn hơn 0; `max-frames` không được âm.
- Nhịp tìm lại thực tế trong async lấy giá trị nhỏ hơn giữa `search-interval` và `detect-interval`.
- Kết quả async bị loại nếu vượt một trong hai giới hạn tuổi: số frame hoặc số giây.
- Các cờ `--sync`, `--no-display`, `--unpaced` chỉ cần ghi tên để bật, không cần thêm `True`.
- `--iou` loại box trùng trong kết quả YOLO; `--match-iou` ghép box với đối tượng đã tracking để giữ ID. Hai tham số này có nhiệm vụ khác nhau.
- Giảm `detect-interval` không làm một lần inference nhanh hơn. Giảm `max-result-age` hoặc `max-result-seconds` cũng không tăng tốc model; chỉ loại kết quả cũ mạnh hơn.
- Các đường dẫn tương đối được tính từ thư mục terminal hiện tại.

## Ví dụ chạy từ thư mục gốc project

Khi đã kích hoạt môi trường `venv`, chạy video với cấu hình mặc định:

```powershell
python scripts/realtime_lp_detect.py --source "data/test/test2.mp4"
```

Chỉ định rõ các tham số thường thay đổi:

```powershell
python scripts/realtime_lp_detect.py --source "data/test/test2.mp4" --imgsz 640 --conf 0.6 --detect-interval 10 --search-interval 5
```

Chạy đồng bộ để đối chiếu:

```powershell
python scripts/realtime_lp_detect.py --source "data/test/test2.mp4" --sync
```

Lưu video và thống kê:

```powershell
python scripts/realtime_lp_detect.py --source "data/test/test2.mp4" --output "data/test/result.mp4" --stats-json "data/test/result_stats.json"
```
