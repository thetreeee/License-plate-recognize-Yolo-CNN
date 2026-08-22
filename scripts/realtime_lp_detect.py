
# ============================================
# realtime_lp_detect.py
# License Plate Detection — Real-time Webcam
# ============================================
# Usage: python scripts/realtime_lp_detect.py --weights weights/best.pt

import cv2
import time
import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Real-time License Plate Detection")
    parser.add_argument("--weights", type=str, default="weights/best.pt", help="Path to YOLOv8 weights")
    parser.add_argument("--source", type=int, default=0, help="Webcam index (0, 1, ...)")
    parser.add_argument("--conf", type=float, default=0.1, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference image size")
    args = parser.parse_args()

    # Load model
    model = YOLO(args.weights)
    print(f"✅ Model loaded: {args.weights}")

    # Open webcam
    cap = cv2.VideoCapture(args.source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("❌ Cannot open webcam!")
        return

    print("🎥 Press 'q' to quit")

    fps_counter = 0
    fps_start = time.time()
    fps_display = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Inference
        results = model.predict(
            source=frame,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=0,       # GPU. Đổi thành 'cpu' nếu không có GPU
            verbose=False,
            half=True,      # FP16 cho tốc độ nhanh hơn trên GPU
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

        # Draw FPS
        cv2.putText(annotated, f"FPS: {fps_display:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Draw detection count
        n_det = len(results[0].boxes)
        cv2.putText(annotated, f"Plates: {n_det}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("License Plate Detection - YOLOv8n", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("👋 Done!")

if __name__ == "__main__":
    main()
