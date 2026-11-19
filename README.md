# Web nhận diện biển số bằng Docker CUDA

Nhánh `PhungVu_dockerDeloy` cung cấp web tải video và camera laptop, chạy YOLO, ByteTrack, MobileNet và LPRNet v5 trên GPU NVIDIA. Trình duyệt hiển thị video dự đoán trong lúc xử lý và các biển số từng đạt `CONSENSUS` kèm ảnh crop.

## Chuẩn bị máy

- GPU NVIDIA và driver hoạt động. Kiểm tra bằng `nvidia-smi`.
- Git và Git LFS để lấy đủ checkpoint.
- Docker và Docker Compose hỗ trợ `gpus: all`.
- Trên Windows, dùng Docker Desktop với Linux containers và backend WSL 2. Bật **Use the WSL 2 based engine** trong Settings → General.
- Trên Linux, cấu hình NVIDIA Container Toolkit để Docker truy cập GPU.
- Có Internet và dung lượng trống cho lần build đầu. Image CUDA trong lần triển khai trước chiếm khoảng 10 GB.

Image cài Python 3.11, PyTorch 2.6.0, torchvision 0.21.0 và CUDA runtime 12.4. Dockerfile cài các thư viện Python và FFmpeg trong image. Máy chủ cung cấp driver NVIDIA.

Điều kiện GPU trên Windows được mô tả trong [tài liệu Docker Desktop](https://docs.docker.com/desktop/features/gpu/). Cặp phiên bản PyTorch và torchvision dùng theo [hướng dẫn PyTorch](https://docs.pytorch.org/get-started/previous-versions/).

## Cài đặt lần đầu

Các lệnh dưới đây chạy từ PowerShell hoặc terminal.

```powershell
git lfs install
git clone --branch PhungVu_dockerDeloy --single-branch https://github.com/thetreeee/License-plate-recognize-Yolo-CNN.git
cd License-plate-recognize-Yolo-CNN
git lfs pull
```

Ứng dụng cần ba file trọng số ở gốc repo.

| File | Vai trò |
|---|---|
| `weights/best.pt` | YOLO phát hiện biển số |
| `weights/layout_best.pt` | MobileNet phân loại số dòng |
| `weights/lprnet_v5.pth` | LPRNet đọc ký tự |

Nếu log báo `Git LFS pointer`, chạy lại `git lfs pull` trong repo.

Build và chạy từ thư mục chứa `compose.yaml`.

```powershell
docker compose config --quiet
docker compose up --build -d --wait --wait-timeout 120
docker compose ps
```

Mở [http://localhost:8000](http://localhost:8000). Khi container chuyển sang `healthy`, backend đã nạp model và hoàn tất lượt suy luận khởi động.

Kiểm tra GPU bên trong container bằng phép tính CUDA.

```powershell
docker compose exec web python -c "import torch; assert torch.cuda.is_available(); x=torch.ones(1,device='cuda:0'); torch.cuda.synchronize(); print(torch.version.cuda, torch.cuda.get_device_name(0), x.device)"
```

Kết quả cần có tên GPU NVIDIA và `cuda:0`.

## Sử dụng

**Video tải lên**

1. Chọn video và bấm **Chạy nhận diện**.
2. Backend đọc file tuần tự, chạy pipeline và gửi khung hình dự đoán về trình duyệt qua MJPEG.
3. Danh sách biển số cập nhật khi một track đạt `CONSENSUS`. Mỗi mục có ảnh crop, ID và thời điểm.
4. Khi xử lý xong, giao diện phát MP4 H.264. Bấm **Tua đến vị trí** ở một biển số để xem thời điểm ghi nhận.

MP4 kết quả chứa phần hình đã vẽ bbox. Tiến độ có thể tiếp tục ở giai đoạn xuất MP4 sau khi đã đọc hết frame.

**Camera laptop**

1. Bấm **Mở camera** và cấp quyền camera trong trình duyệt.
2. Trình duyệt lấy hình từ camera trên máy người dùng, gửi JPEG qua WebSocket và hiển thị kết quả trả về.
3. Bấm **Dừng camera** để kết thúc.

Camera trong trình duyệt cần `localhost` hoặc HTTPS. Nếu truy cập từ máy khác, cấu hình HTTPS và proxy hỗ trợ WebSocket.

**Giới hạn đang áp dụng**

- Một lượt suy luận đang hoạt động tại một thời điểm.
- Video tải lên tối đa 100 MiB, 18.000 frame và 2.073.600 pixel mỗi frame, tương đương 1920 × 1080.
- Video hỗ trợ MP4, MOV, MKV, WebM, AVI, MPEG và FLV tùy codec giải mã.
- Kết quả được lưu tạm khoảng một giờ sau khi xử lý xong. Khởi động lại ứng dụng sẽ xóa các kết quả tạm.
- Phiên camera tối đa một giờ. Nếu ngừng gửi hình 30 giây, máy chủ đóng kết nối.

## Pipeline và cấu trúc code

```text
Video → FastAPI → OpenCV đọc frame
                      ↓
YOLO → ByteTrack → crop chất lượng tốt nhất → MobileNet → LPRNet
                      ↓
               TemporalOCR consensus
                      ↓
       Vẽ bbox → xem trước MJPEG + crop
                      ↓
          Hết video → FFmpeg → MP4 H.264
```

YOLO dùng kích thước đầu vào mặc định 960, quét bổ sung 1280 mỗi 30 frame và quét ROI theo lịch. Crop được chọn dựa trên kích thước và độ nét vùng chữ, giữ tối đa khoảng một giây theo FPS video. Camera dùng cửa sổ 30 frame. Mỗi crop chỉ đóng góp một lượt đọc OCR.

`CONSENSUS` cần ít nhất ba phiếu đồng ý và tỷ lệ tối thiểu 60% trong cửa sổ tối đa năm lần đọc. Chất lượng crop và lỗi OCR vẫn ảnh hưởng đến độ chính xác của kết quả.

```text
app/
  main.py                 API upload, preview, MP4, WebSocket
  pipeline.py             Nạp model và xử lý từng frame
  static/
    index.html            Giao diện
    app.js                Upload, camera và cập nhật kết quả
    style.css             Định dạng giao diện
scripts/                  Detection, tracking, crop và OCR dùng chung
configs/bytetrack.yaml    Cấu hình tracker
weights/                  Checkpoint
Dockerfile                Cài môi trường và khởi động Uvicorn
compose.yaml              Cấp GPU, mở cổng và mount weights chỉ đọc
requirements.docker.txt   Dependency riêng cho web Docker
requirements.txt          Dependency của pipeline CLI
```

Docker build copy `app/`, `scripts/` và `configs/` vào `/app`. Compose mount `weights/` từ máy chủ vào `/app/weights` ở chế độ chỉ đọc. Khi sửa code, build lại image để áp dụng.

## Các lệnh thường dùng

Chạy lại image đã build.

```powershell
docker compose up -d --no-build --wait --wait-timeout 120
```

Dừng web.

```powershell
docker compose stop
```

Xem log để tìm nguyên nhân khi giao diện báo lỗi.

```powershell
docker compose logs -f --tail 100 web
```

Cập nhật nhánh deploy rồi build lại.

```powershell
git pull --ff-only origin PhungVu_dockerDeloy
git lfs pull
docker compose up --build -d --wait --wait-timeout 120
```

Gỡ container và network của ứng dụng.

```powershell
docker compose down
```

## Xử lý lỗi

**WSL environment detected but no adapters were found**

Docker chưa truy cập được GPU trong WSL. Kiểm tra `nvidia-smi` trên Windows. Với Docker Desktop có lệnh `docker desktop`, có thể thử khởi động lại môi trường như sau.

```powershell
docker desktop stop
wsl --shutdown
docker desktop start
docker compose up -d --no-build --wait --wait-timeout 120
```

Thao tác này dừng toàn bộ Docker Desktop và các bản phân phối WSL đang chạy. Nếu Docker Desktop chưa hỗ trợ CLI trên máy, thoát Docker Desktop từ khay hệ thống, chạy `wsl --shutdown`, rồi mở Docker Desktop lại. Nếu lỗi còn lặp lại, kiểm tra driver NVIDIA, WSL và chế độ GPU của laptop.

**CUDA error: unknown error**

Đọc log để xác định lỗi xảy ra trong suy luận. Thử `docker compose restart web`, sau đó chạy lại phép tính CUDA bên trên. Nếu lỗi vẫn lặp lại, thử quy trình khởi động lại WSL và Docker Desktop. Cách này đã phục hồi được GPU trên máy thử, nguyên nhân lỗi CUDA ban đầu còn cần chẩn đoán trên từng máy.

**Xử lý video thất bại**

Chạy `docker compose logs --tail 100 web` để lấy stack trace. Giao diện hiện thông báo chung khi phát sinh lỗi nội bộ. Kiểm tra lỗi CUDA, codec đầu vào và bước FFmpeg trong log.

**Cổng 8000 đang được sử dụng**

Dùng `docker ps` để tìm container chiếm cổng. Dừng container đó hoặc đổi phần cổng máy chủ trong Compose, ví dụ `127.0.0.1:8001:8000`, rồi mở `http://localhost:8001`.

**Video vượt giới hạn độ phân giải**

Tạo bản 1080p trước khi upload. Lệnh sau cần FFmpeg trên máy thực hiện chuyển đổi.

```powershell
ffmpeg -i input.mp4 -vf "scale=1920:1080:force_original_aspect_ratio=decrease:force_divisible_by=2" -c:v libx264 -crf 20 -pix_fmt yuv420p -movflags +faststart output_web.mp4
```

## Phạm vi triển khai

Compose mặc định bind cổng vào loopback của máy chạy Docker. Backend có quyền truy cập GPU NVIDIA để suy luận. OpenCV giải mã và FFmpeg xuất video trên CPU.

Nếu đưa web lên máy chủ cho nhiều người sử dụng, cấu hình tên miền, HTTPS, reverse proxy và xác thực phù hợp. Bản hiện tại phục vụ một lượt suy luận tại một thời điểm.
