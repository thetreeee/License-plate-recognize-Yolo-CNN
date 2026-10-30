# So sánh YOLO, YOLO + CSRT và YOLO + ByteTrack

Ngày chạy: 29/10/2026. Đã chạy đủ **9 lượt**, toàn bộ 3 video gốc, và đọc lại đủ frame của cả 9 video kết quả để kiểm tra file đầu ra.

## Điều kiện đo

- NVIDIA GeForce RTX 3050 Laptop GPU; PyTorch `2.1.0+cu121`, Ultralytics `8.4.34`.
- Cùng `weights/best.pt`, `imgsz=640`, `conf=0.5`, NMS IoU `0.45`, device `0`, FP16.
- PyTorch 4 thread CPU, OpenCV 1 thread; không mở cửa sổ, có ghi video MP4 cho cả ba phương án.
- ByteTrack được ghi thêm JSONL để kiểm tra ID; chi phí ghi này cũng nằm trong thời gian vòng xử lý của nó. Vì vậy chênh lệch nhỏ không được xem là phép đo cô lập duy nhất thuật toán ghép ID.
- Mỗi video được phát theo FPS nguồn, không dùng `--unpaced`; đo sau khi load/warmup model. Mỗi trường hợp chạy một lần, nối tiếp, thay đổi thứ tự giữa các video để giảm thiên lệch do thứ tự. Chưa phải benchmark lặp nhiều lần có khoảng tin cậy.
- YOLO thuần và ByteTrack gọi YOLO ở mọi frame. CSRT dùng pipeline async hiện tại, detect interval 10, search interval 5, cấu hình CSRT fast.
- ByteTrack dùng cấu hình trong `configs/bytetrack.yaml`: high `0.5`, low `0.1`, new ID `0.6`, buffer 30, fuse_score false. Vì đặt confidence chung `0.5` để so chi phí, phép đo này **hạn chế bước ghép confidence thấp**. Script ByteTrack vận hành mặc định ở `--conf 0.1`, khác phép đo chung này.
- Không xoay, cắt, chỉnh sáng hay sửa video nguồn. Không có model OCR trong phép đo.

## Kết quả hiệu năng

`FPS phát` = số frame / thời gian vòng phát, có thời gian chờ để phát theo FPS nguồn. Các video gốc đều khoảng 30 FPS nên chỉ số này không cho biết FPS tối đa khi bỏ giới hạn.

`p95` nghĩa là 95% vòng xử lý không chậm hơn giá trị đó. Thời gian vòng xử lý gồm đọc, vẽ, ghi video và phần xử lý trên luồng chính; không gồm thời gian chờ phát và model warmup. Với CSRT async, inference đang chạy song song nên p95 luồng chính không phải độ trễ từ frame nguồn đến kết quả.

| Video | Phương án | FPS phát | p95 vòng chính (ms) | Chậm nhất (ms) | Lần gọi YOLO / tổng frame |
|---|---|---:|---:|---:|---:|
| test.mp4 | YOLO mỗi frame | 29.55 | 28.35 | 55.32 | 232 / 232 |
| test.mp4 | YOLO + CSRT | 29.64 | 12.64 | 19.58 | 46 / 232 |
| test.mp4 | YOLO + ByteTrack | 29.52 | 26.98 | 63.27 | 232 / 232 |
| test2.mp4 | YOLO mỗi frame | 29.64 | 25.41 | 31.03 | 250 / 250 |
| test2.mp4 | YOLO + CSRT | 29.63 | 26.05 | 34.64 | 33 / 250 |
| test2.mp4 | YOLO + ByteTrack | 29.65 | 27.63 | 31.54 | 250 / 250 |
| test3.mp4 | YOLO mỗi frame | 29.55 | 25.86 | 33.22 | 349 / 349 |
| test3.mp4 | YOLO + CSRT | 29.30 | 32.60 | 62.77 | 48 / 349 |
| test3.mp4 | YOLO + ByteTrack | 29.55 | 27.99 | 38.86 | 349 / 349 |

Ở hai video đúng chiều, thời gian xử lý chủ động trung bình tương ứng với khoảng 51.6/51.5 FPS của YOLO và 45.2/48.3 FPS của ByteTrack, **khi loại thời gian chờ phát**. Đây không phải FPS quan sát của lượt phát đang bị giới hạn gần 30. Không so trực tiếp con số này với `active_loop_fps` CSRT để kết luận chi phí tổng: phần inference/bù của CSRT chạy ngoài luồng chính.

## Box và ID: số liệu quan sát, không phải accuracy

| Video | Frame có box: YOLO | Frame có box: CSRT | Frame có box: ByteTrack | ID ByteTrack đã hiển thị |
|---|---:|---:|---:|---:|
| test.mp4 (232 frame) | 4 | 10 | 0 | 0 |
| test2.mp4 (250 frame) | 207 | 182 | 203 | 5 |
| test3.mp4 (349 frame) | 221 | 219 | 214 | 1 |

Không có nhãn bounding box/ID chuẩn cho video nên chưa tính precision, recall, mAP, IDF1 hay số lần đổi ID chính xác. Frame có nhiều box chưa chắc tốt hơn: có thể là box nhận nhầm hoặc bám nhầm. ByteTrack chỉ vẽ box có ID được xác nhận, còn YOLO thuần vẽ mọi detection vượt `conf=0.5`, nên hai cột cũng khác tiêu chí chấp nhận box. Mốc tạo ID ByteTrack là `0.6`.

### test.mp4

Video chứa biển số bị xoay ngang khoảng 90 độ. Với cấu hình chung 640/0.5, model gần như không bắt được biển số; ByteTrack không tạo được track xác nhận. Ở frame 205, CSRT vẽ box trên nền gạch trong khi biển số nằm phía dưới ảnh. Vì vậy 10 frame có box CSRT không chứng minh nhận diện tốt hơn 4 frame YOLO hay 0 frame ByteTrack.

Trường hợp này cho thấy lỗi detector/đầu vào không được giải quyết chỉ bằng thay tracker. Không dùng kết quả này làm bằng chứng ByteTrack tăng độ chính xác hoặc CSRT hiệu quả hơn.

### test2.mp4

Ở các frame kiểm tra 22, 67 và 203, cả ba phương án bám vào vùng biển số; YOLO và ByteTrack cập nhật box theo frame hiện tại, CSRT dùng box tracking đã được hiệu chỉnh định kỳ. ByteTrack có 203 frame hiển thị box, gần 207 frame YOLO; CSRT có 182 frame.

ByteTrack hiển thị 5 ID, không có nghĩa có 5 xe: biển số quay lại sau khi rời khung có thể nhận ID mới. Ví dụ biển `59-M1 / 204.24` ở frame 67 mang ID 2 và ở frame 203 mang ID 4. Đây là giới hạn tracking ngắn hạn, không có re-identification theo ngoại hình.

### test3.mp4

Ở frame 70 không có biển số trong ảnh và cả ba không vẽ box. Ở frame 150 và 280, cả ba đều vẽ box vào biển số đang thấy. ByteTrack duy trì một ID được hiển thị trong lượt chạy, nhưng cần xem toàn bộ và gán nhãn để đánh giá định lượng.

YOLO có p95 thấp nhất trong ba phương án trên video này. CSRT có p95 gần ngân sách 33.3 ms/frame của 30 FPS và một vòng chậm khoảng 62.8 ms. Không xem chênh lệch FPS phát vài phần mười là kết luận thống kê chắc chắn từ một lượt chạy.

## Kết luận sử dụng

- Với ba video ngắn hiện có, cả ba gần đạt tốc độ phát nguồn. **Chưa có bằng chứng ByteTrack tăng FPS so với YOLO thuần**; trên hai video đúng chiều nó có thêm chi phí nhỏ nhưng cung cấp ID để quản lý OCR sau này.
- CSRT giảm khoảng 80–87% số lần YOLO trong lượt đo, nhưng lợi ích đó không đồng nghĩa tăng chất lượng hay FPS. Cần tính cả độ trễ, lỗi bám nền và chi phí theo số đối tượng.
- Nếu chỉ cần bounding box, YOLO mỗi frame vẫn là baseline đơn giản và mạnh trên GPU này. Nếu cần ID để gom crop và hạn chế đọc OCR lặp, ByteTrack là bản thử nghiệm phù hợp để tiếp tục đánh giá.
- Ba video hiện tại có ít biển số; không suy rộng thành kết luận cho cảnh đông xe, camera cố định hay model OCR chưa được tích hợp. Chưa chứng minh hiệu năng toàn hệ thống hai model.

## Kết quả đã lưu trong repository

Thư mục [`benchmarks/2026-10-29/`](../benchmarks/2026-10-29/) chứa số liệu JSON gốc, JSONL của ByteTrack, ảnh đối chiếu và script tái chạy. Các đường dẫn tuyệt đối trong JSON là thông tin lịch sử của máy đo; chỉ phần xác định thư mục project/đầu ra trong script tái chạy được điều chỉnh để dùng từ repository.

- [Đối chiếu test.mp4](../benchmarks/2026-10-29/test_comparison.jpg)
- [Đối chiếu test2.mp4](../benchmarks/2026-10-29/test2_comparison.jpg)
- [Đối chiếu test3.mp4](../benchmarks/2026-10-29/test3_comparison.jpg)
- [Số liệu và lệnh đầy đủ](../benchmarks/2026-10-29/comparison.json)

Video nguồn và 9 MP4 đầu ra vẫn ở `data/test/diagnostics/comparison_20261029/` hoặc `data/test/` tại máy hiện tại, được Git ignore và không đi cùng repository khi clone. Báo cáo này chỉ so sánh detection/tracking; kết quả thử OCR nằm ngoài phạm vi công bố lần này.

Ví dụ video kết quả ByteTrack cục bộ: `data/test/diagnostics/comparison_20261029/test2_bytetrack.mp4`. Muốn tái chạy cần có các video nguồn tương ứng; xem [hướng dẫn](../benchmarks/2026-10-29/README.md).
