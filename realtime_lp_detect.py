# ============================================
# realtime_lp_detect.py
# License Plate Detection — Webcam / Video file
# ============================================
# Webcam: python scripts/realtime_lp_detect.py --weights weights/best.pt
# Video : python scripts/realtime_lp_detect.py --weights weights/best.pt --source "D:/videos/test.mp4"
# Lưu kết quả: thêm --save output.mp4

import cv2
import time
import argparse
import torch
from ultralytics import YOLO


def parse_source(src: str):
    """'0', '1' -> webcam index (int); còn lại -> đường dẫn video (str)."""
    return int(src) if src.isdigit() else src


def main():
    parser = argparse.ArgumentParser(description="License Plate Detection (webcam / video)")
    parser.add_argument("--weights", type=str, default="weights/best.pt", help="Path to YOLOv8 weights")
    parser.add_argument("--source", type=str, default="0", help="Webcam index (0,1,...) hoặc đường dẫn file video")
    parser.add_argument("--conf", type=float, default=0.1, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference image size")
    parser.add_argument("--device", type=str, default="auto", help="'auto', 'cpu', hoặc '0' cho GPU")
    parser.add_argument("--save", type=str, default="", help="Đường dẫn lưu video kết quả (vd: output.mp4)")
    parser.add_argument("--no-show", action="store_true", help="Không mở cửa sổ hiển thị (chỉ lưu file)")
    args = parser.parse_args()

    # Auto-detect device
    if args.device == "auto":
        device = 0 if torch.cuda.is_available() else "cpu"
    else:
        device = int(args.device) if args.device.isdigit() else args.device

    use_half = (device != "cpu")  # FP16 chỉ hỗ trợ trên GPU

    # Load model
    model = YOLO(args.weights)
    print(f"✅ Model loaded: {args.weights}")
    print(f"🖥️  Device: {'GPU (' + torch.cuda.get_device_name(0) + ')' if device != 'cpu' else 'CPU'}")

    # Open source
    source = parse_source(args.source)
    is_webcam = isinstance(source, int)
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"❌ Cannot open source: {args.source}")
        return

    if is_webcam:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_webcam else 0
    print(f"🎥 Source: {width}x{height} @ {src_fps:.1f} FPS"
          + (f", {total_frames} frames" if total_frames > 0 else ""))

    # Video writer (tuỳ chọn)
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, src_fps, (width, height))
        print(f"💾 Saving to: {args.save}")

    print("Phím tắt: 'q' = thoát | SPACE = tạm dừng/tiếp tục")

    fps_counter = 0
    fps_start = time.time()
    fps_display = 0
    frame_idx = 0
    paused = False
    annotated = None

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("📼 Hết video." if not is_webcam else "❌ Mất tín hiệu webcam.")
                break
            frame_idx += 1

            # Inference
            results = model.predict(
                source=frame,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                device=device,
                verbose=False,
                half=use_half,
            )

            # Draw results
            annotated = results[0].plot(conf=True, line_width=2)

            # FPS calculation
            fps_counter += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                fps_display = fps_counter / elapsed
                fps_counter = 0
                fps_start = time.time()

            cv2.putText(annotated, f"FPS: {fps_display:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            n_det = len(results[0].boxes)
            cv2.putText(annotated, f"Plates: {n_det}", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            if total_frames > 0:
                cv2.putText(annotated, f"Frame: {frame_idx}/{total_frames}", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            if writer is not None:
                writer.write(annotated)

        if not args.no_show and annotated is not None:
            cv2.imshow("License Plate Detection - YOLOv8n", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                paused = not paused
        elif args.no_show and frame_idx % 50 == 0 and total_frames > 0:
            print(f"  ... {frame_idx}/{total_frames} frames")

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print("👋 Done!")


if __name__ == "__main__":
    main()