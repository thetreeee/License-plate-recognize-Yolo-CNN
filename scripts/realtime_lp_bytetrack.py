"""Run license-plate YOLO on every frame, then associate IDs with ByteTrack."""

import argparse
from collections import deque
import json
from pathlib import Path
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default=str(PROJECT_ROOT / "weights/best.pt"))
    parser.add_argument("--source", default="0", help="Webcam index, video path, or URL")
    parser.add_argument("--mode", choices=("track", "detect"), default="track",
                        help="detect provides a YOLO-only timing baseline")
    parser.add_argument("--tracker", default=str(PROJECT_ROOT / "configs/bytetrack.yaml"))
    parser.add_argument("--conf", type=float, default=0.1,
                        help="YOLO prefilter; new IDs require new_track_thresh in tracker YAML")
    parser.add_argument("--iou", type=float, default=0.45, help="YOLO NMS IoU")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="auto", help="auto, cpu, or CUDA index")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--no-display", "--no-show", action="store_true")
    parser.add_argument("--output", "--save", help="Annotated MP4 output")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--unpaced", action="store_true", help="Read files as fast as processing allows")
    parser.add_argument("--stats-json", help="Write timing and ID counts")
    parser.add_argument("--tracks-jsonl", help="Optional per-frame boxes/IDs for inspection or future OCR")
    args = parser.parse_args(argv)
    if not 0 < args.conf <= 1 or not 0 < args.iou <= 1:
        parser.error("--conf and --iou must be in (0, 1]")
    if args.imgsz < 32 or args.cpu_threads < 1 or args.max_frames < 0:
        parser.error("Require imgsz >= 32, cpu-threads >= 1, max-frames >= 0")
    if not Path(args.weights).is_file():
        parser.error(f"Weights not found: {args.weights}")
    if args.mode == "track" and not Path(args.tracker).is_file():
        parser.error(f"Tracker config not found: {args.tracker}")
    outputs = [Path(path).resolve() for path in (args.output, args.stats_json, args.tracks_jsonl) if path]
    if len(set(outputs)) != len(outputs):
        parser.error("Output files must use distinct paths")
    protected = {Path(args.weights).resolve(), Path(args.tracker).resolve(), Path(args.source).resolve()}
    if any(path in protected for path in outputs):
        parser.error("Output files must differ from source, weights and tracker config")
    return args


def read_tracker_config(path):
    import yaml

    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("tracker_type") != "bytetrack":
        raise ValueError("This demo requires a ByteTrack tracker config")
    low, high, new = (float(config[key]) for key in
                      ("track_low_thresh", "track_high_thresh", "new_track_thresh"))
    if not 0 <= low < high <= 1 or not 0 < new <= 1:
        raise ValueError("Require 0 <= track_low_thresh < track_high_thresh <= 1 and 0 < new_track_thresh <= 1")
    return config


def observations(result, tracking=True):
    """Do not display raw low-score proposals when ByteTrack returned no IDs.

    Some Ultralytics versions retain the original detections when the tracker
    returns zero tracks. Those boxes must not masquerade as confirmed tracks.
    """
    boxes = result.boxes
    if boxes is None or (tracking and not boxes.is_track):
        return []
    coords = boxes.xyxy.cpu().tolist()
    scores = boxes.conf.cpu().tolist()
    ids = boxes.id.cpu().tolist() if boxes.is_track else [None] * len(coords)
    return [dict(id=int(identity) if identity is not None else None,
                 xyxy=[float(value) for value in box], confidence=float(score))
            for box, score, identity in zip(coords, scores, ids)]


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[int(fraction * (len(ordered) - 1))] if ordered else 0.0


def main(argv=None):
    args = parse_args(argv)
    import cv2
    import torch
    from ultralytics import YOLO

    torch.set_num_threads(args.cpu_threads)
    cv2.setNumThreads(1)
    device = (0 if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    use_half = str(device).isdigit() or str(device).startswith("cuda")
    if use_half and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this Python environment; use --device cpu or auto")
    gpu_device = (int(device) if str(device).isdigit() else device) if use_half else None
    if gpu_device is not None:
        torch.cuda.set_device(gpu_device)
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(gpu_device)
    tracking = args.mode == "track"
    if tracking:
        config = read_tracker_config(args.tracker)
        # Fail with setup instructions rather than triggering an implicit pip install.
        import importlib.util
        if importlib.util.find_spec("lap") is None:
            raise RuntimeError('ByteTrack requires lap. Run: python -m pip install "lap>=0.5.12"')
        if args.conf > config["track_low_thresh"]:
            print("Note: --conf filters low-score observations before ByteTrack; low-score rescue is limited.", flush=True)

    source = int(args.source) if args.source.isdecimal() else args.source
    cap = cv2.VideoCapture(source)
    writer = trace = None
    frame_count = visible_frames = 0
    ids_seen = set()
    active_seconds = predict_seconds = 0.0
    loops, predict_times = deque(maxlen=10000), deque(maxlen=10000)
    print(f"Model: {args.weights} | Device: {device} | Mode: {'ByteTrack' if tracking else 'YOLO only'}", flush=True)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open source: {args.source}")
        if isinstance(source, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        if not 0 < source_fps < 1000:
            source_fps = 30.0
        paced = (not isinstance(source, int) and cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0
                 and not args.unpaced)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Source contains no readable frames")
        model = YOLO(args.weights)
        kwargs = dict(conf=args.conf, iou=args.iou, imgsz=args.imgsz,
                      device=device, half=use_half, verbose=False, classes=[0])
        print("Warming up YOLO before playback...", flush=True)
        # Warm the actual source shape; ByteTrack is initialized only in the loop.
        for _ in range(2):
            model.predict(frame, **kwargs)[0].boxes.cpu()
        if tracking:
            # Load tracker callbacks/lap and initialize before playback, then
            # reset so warmup neither consumes frame 0 nor leaves phantom IDs.
            model.track(frame, persist=True, tracker=args.tracker, **kwargs)[0].boxes.cpu()
            for tracker in model.predictor.trackers:
                tracker.reset()
        else:
            model.predict(frame, **kwargs)[0].boxes.cpu()
        if args.output:
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), source_fps,
                                     (frame.shape[1], frame.shape[0]))
            if not writer.isOpened():
                raise RuntimeError(f"Cannot write video: {args.output}")
        if args.tracks_jsonl:
            path = Path(args.tracks_jsonl)
            path.parent.mkdir(parents=True, exist_ok=True)
            trace = path.open("w", encoding="utf-8")
        if not args.no_display:
            print("Press q to quit", flush=True)
        playback_start = fps_start = time.perf_counter()
        fps_count, fps_display = 0, 0.0
        first_frame = frame
        while args.max_frames == 0 or frame_count < args.max_frames:
            started = time.perf_counter()
            if first_frame is not None:
                frame, first_frame = first_frame, None
            else:
                ok, frame = cap.read()
                if not ok:
                    break
            predict_start = time.perf_counter()
            if tracking:
                result = model.track(frame, persist=True, tracker=args.tracker, **kwargs)[0]
            else:
                result = model.predict(frame, **kwargs)[0]
            tracks = observations(result, tracking)
            predict_duration = time.perf_counter() - predict_start
            predict_seconds += predict_duration
            predict_times.append(predict_duration * 1000)
            if trace:
                trace.write(json.dumps(dict(frame=frame_count, seconds=frame_count / source_fps,
                                            tracks=tracks)) + "\n")
            annotated = frame.copy()
            for track in tracks:
                x1, y1, x2, y2 = (round(value) for value in track["xyxy"])
                identity = track["id"]
                if identity is not None:
                    ids_seen.add(identity)
                label = (f"ID {identity}" if identity is not None else "plate")
                label += f" {track['confidence']:.2f}"
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 0), 2)
                cv2.putText(annotated, label, (x1, max(18, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 220, 0), 2)
            frame_count += 1
            visible_frames += bool(tracks)
            fps_count += 1
            elapsed = time.perf_counter() - fps_start
            if elapsed >= 1:
                fps_display, fps_count, fps_start = fps_count / elapsed, 0, time.perf_counter()
            labels = [f"FPS: {fps_display:.1f} | {'ByteTrack' if tracking else 'YOLO only'}",
                      f"Plates: {len(tracks)} | IDs seen: {len(ids_seen)}",
                      f"YOLO: {frame_count}/{frame_count}"]
            for line, label in enumerate(labels):
                cv2.putText(annotated, label, (10, 28 + line * 27),
                            cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 0), 2)
            if writer:
                writer.write(annotated)
            quit_requested = False
            if not args.no_display:
                cv2.imshow("License Plate - YOLO + ByteTrack", annotated)
                quit_requested = cv2.waitKey(1) & 0xFF == ord("q")
            duration = time.perf_counter() - started
            active_seconds += duration
            loops.append(duration * 1000)
            if quit_requested or (args.max_frames and frame_count >= args.max_frames):
                break
            if paced:
                time.sleep(max(0, 1 / source_fps - duration))
        playback_seconds = time.perf_counter() - playback_start
    finally:
        cap.release()
        if writer:
            writer.release()
        if trace:
            trace.close()
        if not args.no_display:
            cv2.destroyAllWindows()

    summary = dict(
        pipeline="bytetrack" if tracking else "detect", frames=frame_count, yolo_calls=frame_count,
        imgsz=args.imgsz, conf=args.conf, device=str(device), source=str(args.source),
        cpu_threads=args.cpu_threads, unique_ids=len(ids_seen), frames_with_visible_tracks=visible_frames,
        processing_fps=frame_count / active_seconds if active_seconds else 0,
        playback_seconds=playback_seconds, loop_p50_ms=percentile(loops, .5),
        loop_p95_ms=percentile(loops, .95), loop_max_ms=max(loops, default=0),
        predict_track_p50_ms=percentile(predict_times, .5),
        predict_track_p95_ms=percentile(predict_times, .95), inference_tracking_seconds=predict_seconds,
        timing_samples=len(loops),
        peak_cuda_allocated_mb=(torch.cuda.max_memory_allocated(gpu_device) / 2**20 if use_half else None),
        peak_cuda_reserved_mb=(torch.cuda.max_memory_reserved(gpu_device) / 2**20 if use_half else None),
    )
    print(f"Frames: {frame_count} | YOLO calls: {frame_count} | IDs seen: {len(ids_seen)}")
    print(f"Active loop FPS: {summary['processing_fps']:.1f} | p95: {summary['loop_p95_ms']:.1f}ms")
    print("Timing excludes model warmup/file pacing; includes drawing/output. ID count is not vehicle count or accuracy.")
    if args.stats_json:
        path = Path(args.stats_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    main()
