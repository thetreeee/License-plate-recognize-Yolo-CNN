# Nhận diện biển số — YOLOv8n, ByteTrack và OCR

Chạy `scripts/realtime_lp_ocr.py` để phát hiện và theo dõi biển số qua video, rồi đọc ký tự theo lịch riêng cho từng ID. Bật `--adaptive` để chạy bản có quét bổ sung và tìm lại biển vừa mất dấu.

## Cách chạy

Chạy trong PowerShell tại thư mục gốc project. Cần cài Git LFS để tải ba checkpoint trong `weights/`. Tạo môi trường ảo một lần, ưu tiên Python 3.11.

```powershell
git lfs install
git lfs pull
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

```powershell
# Webcam mặc định, đổi 0 thành 1 nếu dùng camera thứ hai
python scripts/realtime_lp_ocr.py --source 0 --adaptive --show

# Video cục bộ, thay bằng đường dẫn video của bạn
python scripts/realtime_lp_ocr.py --source "data/test/test2.mp4" --adaptive --show --paced
```

Nhấn **Q** hoặc **Esc** để dừng. Bỏ `--show` khi đo tốc độ không cần cửa sổ, bỏ `--adaptive` để chạy YOLO toàn cảnh mỗi frame + ByteTrack + OCR. Mặc định tự chọn CPU hoặc GPU. Muốn dùng GPU NVIDIA, môi trường cần PyTorch và torchvision có hỗ trợ CUDA. Mỗi lượt chạy lưu video, log từng frame và thống kê ở `runs/plate_recognition/`. Video trong `data/` do bạn tự cung cấp.

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
    T --> O{"ID đến lịch, crop đạt chất lượng và được chọn vào batch?"}
    O -->|Có| M["MobileNet phân loại số dòng → xử lý hàng → CRNN + CTC"]
    M --> V["Tổng hợp kết quả chữ theo ID"]
    O -->|Chưa| U["Dùng chữ đã lưu nếu có"]
    V --> D["Hiển thị và ghi log khi đồng thuận thay đổi"]
    U --> D
    D -->|Frame tiếp theo| A
```

Mặc định quét 1280 mỗi **30 frame**. Các frame còn lại có tối đa **2 vùng ROI** mỗi frame. Ưu tiên tìm lại track đã xác nhận trong tối đa **3 frame** kể từ lần cuối có bbox khớp. Vị trí dự đoán chỉ dùng để chọn vùng tìm. Bbox đưa vào ByteTrack phải đến từ ảnh hiện tại. Nếu còn lượt, kiểm tra thêm box nhỏ hoặc yếu theo lịch **mỗi 5 frame**. Box yếu vẫn có thể giúp ByteTrack nối ID đang hoạt động.

Ở chế độ OCR theo lịch, ID mới phải xuất hiện **5 frame liên tiếp** trước lần đọc đầu. Sau đó, chờ ít nhất **10 frame** giữa hai lần đọc hoặc **30 frame** khi đã đạt `CONSENSUS`. Mỗi frame đọc tối đa **4 crop**. `CONSENSUS` phản ánh sự đồng thuận giữa các lần đọc và vẫn có thể sai.
