# Nhận diện biển số — YOLOv8n, ByteTrack và LPRNet v5

Chạy `scripts/realtime_lp_ocr.py` để phát hiện, theo dõi và đọc biển số. MobileNet phân loại biển một hoặc hai dòng, sau đó LPRNet v5 đọc ký tự. Chế độ mặc định bật quét bổ sung, nối lại ID, cửa sổ hiển thị và phát theo FPS nguồn.

## Cách chạy

Chạy trong PowerShell tại thư mục gốc project. Dùng Git LFS để tải trọng số YOLO, MobileNet và CRNN. Checkpoint LPRNet v5 có sẵn tại `weights/lprnet_v5.pth`. Tạo môi trường ảo một lần, ưu tiên Python 3.11.

```powershell
git lfs install
git lfs pull
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

```powershell
# Webcam mặc định, đổi 0 thành 1 nếu dùng camera thứ hai
python scripts/realtime_lp_ocr.py --source 0

# Video cục bộ, thay bằng đường dẫn video của bạn
python scripts/realtime_lp_ocr.py --source "data/test/test2.mp4"
```

## So sánh với CRNN

```powershell
# LPRNet v5 mặc định
python scripts/realtime_lp_ocr.py --source "data/test/test2.mp4" --no-show --no-paced

# CRNN đối chứng trên cùng video và cài đặt
python scripts/realtime_lp_ocr.py --source "data/test/test2.mp4" --ocr-model crnn --no-show --no-paced
```

Dùng `--ocr-weights` để nạp checkpoint khác tương thích với backend đã chọn. LPRNet v5 và checkpoint đi kèm lấy từ nhánh `HoangAnh_Feature` tại commit `2bae356`. `summary.json` ghi backend đã chạy trong `ocr_backend`.

Nhấn **Q** hoặc **Esc** để dừng. Dùng `--no-show --no-paced` để chạy benchmark. Dùng `--no-adaptive` để chỉ quét toàn cảnh mỗi frame rồi chạy ByteTrack và OCR. Mặc định tự chọn CPU hoặc GPU. Muốn dùng GPU NVIDIA, môi trường cần PyTorch và torchvision có hỗ trợ CUDA. Mỗi lượt chạy lưu video, log từng frame và thống kê ở `runs/plate_recognition/`. Video trong `data/` do bạn tự cung cấp.

Số ID trên bbox và trong log `CONSENSUS` được cấp lần lượt từ 1 khi mỗi track xuất hiện lần đầu. Số đó được giữ nếu cùng track xuất hiện lại. Trong `frames.jsonl`, `tracks[].display_id` là số hiển thị và `tracks[].id` là ID ByteTrack dùng để đối chiếu `readings.json` và các lần `id_relinks`.

## Pipeline adaptive

```mermaid
flowchart TD
    A["Frame gốc"] --> B["YOLO 640 toàn cảnh mỗi frame"]
    B --> G["Gộp box trong tọa độ ảnh gốc"]
    B --> H{"Đến lịch quét 1280?"}
    H -->|Có| Y["YOLO 1280 toàn cảnh"]
    H -->|Chưa| F["Ưu tiên tìm lại track vừa mất bbox quanh vị trí dự đoán"]
    F --> G
    F --> R{"Còn lượt quét và đến lịch ROI cho box nhỏ hoặc yếu?"}
    R -->|Có| C["Crop rộng từ ảnh gốc, chạy YOLO 640 và kiểm tra box"]
    R -->|Chưa| G
    Y --> G
    C --> G
    G --> N["Loại box trùng bằng NMS"]
    N --> T["ByteTrack ghép bbox và quản lý ID"]
    T --> K["Nếu bật relink, thử nối lại track vừa hụt"]
    K --> O{"ID đến lịch, crop đạt chất lượng và được chọn vào batch?"}
    O -->|Có| M["MobileNet phân loại số dòng → xử lý hàng → LPRNet v5 + CTC"]
    M --> V["Tổng hợp kết quả chữ theo ID"]
    O -->|Chưa| U["Dùng chữ đã lưu nếu có"]
    V --> D["Hiển thị và ghi log khi đồng thuận thay đổi"]
    U --> D
    D -->|Frame tiếp theo| A
```

Mặc định quét 1280 mỗi **30 frame**. Các frame còn lại có tối đa **2 vùng ROI** mỗi frame. Ưu tiên tìm lại track đã xác nhận trong tối đa **3 frame** kể từ lần cuối có bbox khớp. Vị trí dự đoán chỉ dùng để chọn vùng tìm. Bbox đưa vào ByteTrack phải đến từ ảnh hiện tại. Nếu còn lượt, kiểm tra thêm box nhỏ hoặc yếu theo lịch **mỗi 5 frame**. Box yếu vẫn có thể giúp ByteTrack nối ID đang hoạt động.

Chế độ adaptive mặc định ghép lại ID sau tối đa **3 frame bị hụt**. ByteTrack ưu tiên các cặp đã ghép theo cách cũ, rồi xét tâm và kích thước từ các bbox YOLO gần nhất của track chưa ghép. Chỉ nối khi một bbox mới khớp rõ một-một với track đã xác nhận. `frames.jsonl` ghi `id_relinks` ở frame có lượt ghép này. Dùng `--no-relink` để đối chứng.

YOLO mặc định dùng ảnh xám ba kênh. Thêm `--no-grayscale` để đối chứng với đầu vào màu. OCR và video hiển thị dùng frame màu gốc.

Nếu biết vùng thường có biển bị bỏ sót, dùng `--search-roi X1 Y1 X2 Y2` để ưu tiên quét vùng đó tìm biển mới. Tọa độ tính theo tỷ lệ từ 0 đến 1 của chiều rộng và chiều cao frame. Có thể lặp cờ để quét luân phiên nhiều vùng trong ngân sách ROI hiện tại.

```powershell
python scripts/realtime_lp_ocr.py --source "data/test/external_ddkhoa/Download.mp4" --search-roi .547 .483 .825 .674 --search-roi .738 .454 1 .645
```

```powershell
python scripts/realtime_lp_ocr.py --source "data/test/test6.mp4"
```

Ở chế độ OCR theo lịch, ID mới phải xuất hiện **5 frame liên tiếp** trước lần đọc đầu. Sau đó, chờ ít nhất **10 frame** giữa hai lần đọc hoặc **30 frame** khi đã đạt `CONSENSUS`. Mỗi frame đọc tối đa **4 crop**. `CONSENSUS` phản ánh sự đồng thuận giữa các lần đọc và vẫn có thể sai.
