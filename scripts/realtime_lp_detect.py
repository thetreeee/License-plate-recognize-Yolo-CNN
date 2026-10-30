"""YOLO plate detection with asynchronous inference and CSRT tracking."""

import argparse
from collections import deque
import json
from pathlib import Path
import time

import cv2

try:
    from .plate_tracking import PlateAppearance, PlateTracker, clip_box
    from .async_detector import AsyncDetector, DetectionSchedule, FrameHistory, propagate_detections
except ImportError:
    from plate_tracking import PlateAppearance, PlateTracker, clip_box
    from async_detector import AsyncDetector, DetectionSchedule, FrameHistory, propagate_detections


def csrt_factory(quality="fast"):
    factory = getattr(cv2, "TrackerCSRT_create", None)
    if factory is not None and quality == "fast" and hasattr(cv2, "TrackerCSRT_Params"):
        # Full CSRT can take longer than one 30-FPS frame on the CPU. Reduce
        # template/scale search work while retaining HOG features for plates.
        params = cv2.TrackerCSRT_Params()
        params.template_size = 128
        params.number_of_scales = 9
        params.admm_iterations = 2
        params.use_color_names = False
        params.use_segmentation = False
        return factory(params)
    if factory is None:
        factory = getattr(getattr(cv2, "legacy", None), "TrackerCSRT_create", None)
    if factory is None:
        raise RuntimeError(
            "CSRT is unavailable. Install opencv-contrib-python in your project "
            "environment (see README.md), or use --mode detect."
        )
    return factory()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="weights/best.pt")
    parser.add_argument("--source", default="0", help="Webcam index or video path/URL")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=0.45, help="YOLO NMS IoU")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="auto", help="auto, cpu, or CUDA index")
    parser.add_argument("--mode", choices=("track", "detect"), default="track")
    parser.add_argument("--sync", action="store_true", help="Run YOLO synchronously for comparison")
    parser.add_argument("--detect-interval", type=int, default=10)
    parser.add_argument("--search-interval", type=int, default=5,
                        help="Async search interval when no tracks are visible")
    parser.add_argument("--match-iou", type=float, default=0.2)
    parser.add_argument("--max-age", type=int, default=300,
                        help="Frames to retain an identity without YOLO confirmation")
    parser.add_argument("--appearance-threshold", type=float, default=0.35)
    parser.add_argument("--buffer-size", type=int, default=32)
    parser.add_argument("--max-result-age", type=int, default=20,
                        help="Reject async detections older than this many source frames")
    parser.add_argument("--max-result-seconds", type=float, default=1.0)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--csrt-quality", choices=("fast", "full"), default="fast")
    parser.add_argument("--unpaced", action="store_true",
                        help="Process files as fast as possible instead of source FPS")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--output", help="Annotated output video (.mp4)")
    parser.add_argument("--stats-json", help="Write timing and worker counters as JSON")
    args = parser.parse_args()
    if args.detect_interval < 1 or args.max_age < args.detect_interval:
        parser.error("Require 1 <= --detect-interval <= --max-age")
    if not 0 < args.match_iou <= 1:
        parser.error("--match-iou must be in (0, 1]")
    if args.max_frames < 0:
        parser.error("--max-frames must be >= 0")
    if not 0 < args.appearance_threshold <= 1:
        parser.error("--appearance-threshold must be in (0, 1]")
    if args.search_interval < 1 or args.cpu_threads < 1:
        parser.error("--search-interval and --cpu-threads must be positive")
    if args.max_result_age < 1 or args.buffer_size <= args.max_result_age:
        parser.error("Require 1 <= --max-result-age < --buffer-size")
    if args.max_result_seconds <= 0:
        parser.error("--max-result-seconds must be positive")
    if args.output and Path(args.source).resolve() == Path(args.output).resolve():
        parser.error("--output must differ from --source")
    return args


def detections_from_result(result):
    # This project's detector has one class: license_plate.
    for row in result.boxes.data.cpu().tolist():
        x1, y1, x2, y2, confidence, _ = row
        yield (x1, y1, x2 - x1, y2 - y1), confidence


def make_predictor(args, device):
    """Called on the model's owning thread; warm up before reading the video."""
    import numpy as np
    from ultralytics import YOLO

    model = YOLO(args.weights)
    use_half = str(device).isdigit() or str(device).startswith("cuda")

    def predict(frame):
        result = model.predict(
            source=frame, conf=args.conf, iou=args.iou, imgsz=args.imgsz,
            device=device, verbose=False, half=use_half,
        )[0]
        return list(detections_from_result(result))

    predict(np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8))
    return predict


def draw_box(frame, box, label, color):
    x, y, width, height = box
    cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
    cv2.putText(frame, label, (x, max(18, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def main():
    args = parse_args()
    import torch

    # Prevent YOLO/OpenCV thread pools from exhausting the CPU together.
    torch.set_num_threads(args.cpu_threads)
    cv2.setNumThreads(1)
    device = (0 if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    use_async = args.mode == "track" and not args.sync
    pipeline = "async" if use_async else "sync"
    appearance_factory = lambda frame, box: PlateAppearance(frame, box, args.appearance_threshold)
    tracker_factory = lambda: csrt_factory(args.csrt_quality)
    manager = None
    if args.mode == "track":
        tracker_factory()
        manager = PlateTracker(tracker_factory, args.detect_interval, args.match_iou, args.max_age,
                               appearance_factory=appearance_factory)
    history = FrameHistory(args.buffer_size)
    schedule = DetectionSchedule(args.detect_interval, args.search_interval)
    worker = None
    cap = writer = None
    frame_index = yolo_calls = 0
    detector_seconds = processing_seconds = 0.0
    loop_times = deque(maxlen=10000)
    visible_frames = 0
    stats = {}
    print(f"Model: {args.weights} | Device: {device} | Mode: {args.mode} | {pipeline}", flush=True)
    print("Loading and warming up YOLO before playback...", flush=True)
    try:
        if use_async:
            worker = AsyncDetector(
                lambda: make_predictor(args, device), tracker_factory, history,
                appearance_factory=appearance_factory, max_age_frames=args.max_result_age,
                max_age_seconds=args.max_result_seconds,
                propagate=propagate_detections,
            )
            worker.start()
        else:
            predict = make_predictor(args, device)

        source = int(args.source) if args.source.isdecimal() else args.source
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {args.source}")
        if isinstance(source, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        if not 0 < source_fps < 1000:
            source_fps = 30.0
        pace_file = not isinstance(source, int) and not args.unpaced
        if not args.no_display:
            print("Press q to quit", flush=True)
        playback_start = fps_start = time.perf_counter()
        fps_frames, fps_display = 0, 0.0

        while args.max_frames == 0 or frame_index < args.max_frames:
            loop_start = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                break
            current = history.append(frame_index, frame)
            old_visible = {track.track_id for track in manager.visible_tracks} if manager else set()
            need_detection = manager is None or manager.update(frame, frame_index)
            lost_now = bool(old_visible - {track.track_id for track in manager.visible_tracks}) if manager else False
            detections = []
            if use_async:
                result = worker.poll()
                if result is not None:
                    worker.apply(result, manager, current)
                if schedule.due(frame_index, bool(manager.visible_tracks), lost_now):
                    if worker.submit(current):
                        schedule.requested(frame_index)
                stats = worker.stats()
                yolo_calls = stats["calls"]
                detector_seconds = stats["inference_seconds"]
                status = "YOLO BG" if stats["busy"] else "TRACK"
                if not manager.visible_tracks:
                    status = "SEARCH / " + status
            else:
                status = "DETECT" if need_detection else "TRACK"
                if need_detection:
                    detect_start = time.perf_counter()
                    detections = predict(frame)
                    detector_seconds += time.perf_counter() - detect_start
                    yolo_calls += 1
                    if manager is not None:
                        manager.reconcile(frame, detections, frame_index)

            annotated = frame.copy()
            if manager is not None:
                tracks = manager.visible_tracks
                count = len(tracks)
                for track in tracks:
                    color = (0, 220, 0) if track.source == "YOLO" else (0, 200, 255)
                    draw_box(annotated, track.box,
                             f"ID {track.track_id} {track.source} last_det={track.confidence:.2f}", color)
            else:
                count = 0
                for box, confidence in detections:
                    box = clip_box(box, frame.shape)
                    if box is not None:
                        count += 1
                        draw_box(annotated, box, f"YOLO {confidence:.2f}", (0, 220, 0))
            visible_frames += bool(count)
            frame_index += 1
            fps_frames += 1
            elapsed = time.perf_counter() - fps_start
            if elapsed >= 1:
                fps_display = fps_frames / elapsed
                fps_frames, fps_start = 0, time.perf_counter()
            labels = [f"FPS: {fps_display:.1f} | {status}",
                      f"Plates: {count} | YOLO: {yolo_calls}/{frame_index}"]
            if use_async:
                labels.append(f"Delay: {stats['latency_ms']:.0f}ms | Applied: {stats['applied']} | Stale: {stats['stale']}")
            for line, label in enumerate(labels):
                cv2.putText(annotated, label, (10, 30 + line * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if args.output:
                if writer is None:
                    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"),
                                             source_fps, (frame.shape[1], frame.shape[0]))
                    if not writer.isOpened():
                        raise RuntimeError(f"Cannot write video: {args.output}")
                writer.write(annotated)
            quit_requested = False
            if not args.no_display:
                cv2.imshow("License Plate - YOLO + CSRT", annotated)
                quit_requested = cv2.waitKey(1) & 0xFF == ord("q")
            duration = time.perf_counter() - loop_start
            processing_seconds += duration
            loop_times.append(duration * 1000)
            if quit_requested:
                break
            if pace_file:
                # Do not rush to EOF while background inference is still working.
                time.sleep(max(0, 1 / source_fps - duration))

        playback_seconds = time.perf_counter() - playback_start
    finally:
        if worker is not None:
            if not worker.close():
                print("Worker still finishing inference; shutdown timeout reached.", flush=True)
            stats = worker.stats()
            yolo_calls = stats["calls"]
            detector_seconds = stats["inference_seconds"]
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    fps = frame_index / processing_seconds if processing_seconds else 0
    skipped = 100 * (1 - yolo_calls / frame_index) if frame_index else 0
    ordered = sorted(loop_times)
    percentile = lambda q: ordered[int(q * (len(ordered) - 1))] if ordered else 0.0
    summary = dict(
        pipeline=pipeline, frames=frame_index, yolo_calls=yolo_calls,
        skipped_percent=skipped, active_loop_fps=fps, playback_seconds=playback_seconds,
        loop_p50_ms=percentile(.5), loop_p95_ms=percentile(.95),
        loop_max_ms=max(ordered, default=0), inference_seconds=detector_seconds,
        frames_with_visible_tracks=visible_frames, worker=stats if use_async else None,
    )
    print(f"Frames: {frame_index} | YOLO calls: {yolo_calls} | Skipped: {skipped:.1f}%")
    print(f"Active loop FPS: {fps:.1f} | YOLO total: {detector_seconds:.3f}s")
    print(f"Loop ms: p50={summary['loop_p50_ms']:.1f} p95={summary['loop_p95_ms']:.1f} max={summary['loop_max_ms']:.1f}")
    if use_async:
        print(f"Async applied: {stats['applied']} | Stale discarded: {stats['stale']} | Latest delay: {stats['latency_ms']:.0f}ms")
    print("Loop timing excludes model warmup and file pacing; it includes tracking/drawing/output.")
    if args.stats_json:
        path = Path(args.stats_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
