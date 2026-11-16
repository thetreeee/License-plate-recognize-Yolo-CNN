"""Run YOLO + ByteTrack with optional adaptive scans and scheduled OCR."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from scripts.temporal_ocr import TemporalOCR


def resolve_device(value):
    import torch

    value = value.lower()
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if value == "cpu":
        return torch.device("cpu")
    if value == "cuda":
        value = "cuda:0"
    elif value.isdigit():
        value = "cuda:" + value
    try:
        device = torch.device(value)
    except (RuntimeError,ValueError) as exc:
        raise ValueError("Use --device auto, cpu, 0, or cuda:0") from exc
    if device.type != "cuda":
        raise ValueError("This script supports CPU or NVIDIA CUDA")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use --device cpu")
    index = device.index if device.index is not None else 0
    if index >= torch.cuda.device_count():
        raise ValueError(f"CUDA device {index} does not exist")
    return torch.device(f"cuda:{index}")


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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="0", help="Video path or webcam index")
    parser.add_argument("--weights", type=Path, default=ROOT / "weights/best.pt")
    parser.add_argument("--layout-weights", type=Path, default=ROOT / "weights/layout_best.pt")
    parser.add_argument("--ocr-weights", type=Path, default=ROOT / "weights/crnn_best.pt")
    parser.add_argument("--tracker", type=Path, default=ROOT / "configs/bytetrack.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--adaptive", action="store_true", help="Scheduled high-resolution and ROI checks before ByteTrack")
    parser.add_argument("--relink", action="store_true", help="Try a short-gap motion match before creating a new ID (requires --adaptive)")
    parser.add_argument("--high-imgsz", type=int, default=1280)
    parser.add_argument("--high-interval", type=int, default=30)
    parser.add_argument("--roi-interval", type=int, default=5)
    parser.add_argument("--max-rois", type=int, default=2)
    parser.add_argument("--conf", type=float, default=0.2)
    parser.add_argument("--iou", type=float, default=.45)
    parser.add_argument("--ocr-conf", type=float, default=.6, help="Minimum YOLO score for an OCR crop")
    parser.add_argument("--ocr-interval", type=int, default=10)
    parser.add_argument("--stable-interval", type=int, default=30)
    parser.add_argument("--ocr-batch", type=int, default=4)
    parser.add_argument("--min-blur", type=float, default=20., help="Minimum grayscale Laplacian variance; heuristic")
    parser.add_argument("--ocr-every-frame", action="store_true", help="Baseline: read all visible tracks, bypass quality gates")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--paced", action="store_true", help="Limit playback to source FPS; default measures processing speed")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/plate_recognition")
    args = parser.parse_args(argv)
    if min(args.imgsz, args.ocr_interval, args.stable_interval, args.ocr_batch) < 1 or args.max_frames < 0:
        parser.error("Sizes/intervals/batch must be positive; max-frames must be >= 0")
    if not all(0 < value <= 1 for value in (args.conf, args.iou, args.ocr_conf)) or args.min_blur < 0:
        parser.error("Invalid confidence/IoU/blur threshold")
    if args.adaptive and (args.high_imgsz <= args.imgsz or
                         min(args.high_interval, args.roi_interval, args.max_rois) < 1):
        parser.error("Adaptive mode requires high-imgsz > imgsz and positive intervals/ROI budget")
    if args.relink and not args.adaptive:
        parser.error("--relink requires --adaptive")
    return args


def main(reader_factory=None):
    args = parse_args()
    import cv2
    import numpy as np
    import torch
    from ultralytics import YOLO
    from scripts.lpr_models import PlateReader, crop_from_box, check_weight_file, synchronize

    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device)
    read_tracker_config(args.tracker)
    detector = YOLO(str(check_weight_file(args.weights)), task="detect")
    raw_detections = []

    def capture_raw(predictor):
        # Register before model.track installs its callback: those results are
        # still YOLO proposals, before ByteTrack replaces/filters their boxes.
        raw_detections.clear()
        for result in predictor.results:
            boxes = result.boxes
            if boxes is not None:
                raw_detections.extend(dict(xyxy=box, confidence=score) for box, score in
                                      zip(boxes.xyxy.detach().cpu().tolist(),
                                          boxes.conf.detach().cpu().tolist()))

    if not args.adaptive:
        detector.add_callback("on_predict_postprocess_end", capture_raw)
    reader = (reader_factory or PlateReader)(args.ocr_weights, args.layout_weights, device)
    ocr_name = getattr(reader, "name", "CRNN")
    detector_device = str(device.index) if device.type == "cuda" else "cpu"
    kwargs = dict(device=detector_device, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                  classes=[0], half=False, verbose=False)
    adaptive = None
    if args.adaptive:
        from scripts.adaptive_detection import AdaptiveDetector
        adaptive = AdaptiveDetector(detector, kwargs, read_tracker_config(args.tracker),
                                    args.high_imgsz, args.high_interval, args.roi_interval, args.max_rois,
                                    relink=args.relink)
    mode = "every_frame" if args.ocr_every_frame else "scheduled"
    detection_mode = "adaptive_relink" if args.relink else ("adaptive" if args.adaptive else "full_frame")
    name = Path(args.source).stem if not args.source.isdigit() else "webcam"
    folder = args.output_dir / (name + "_" + detection_mode + "_" + mode + "_" + time.strftime("%Y%m%d_%H%M%S"))
    folder.mkdir(parents=True, exist_ok=False)
    source = int(args.source) if args.source.isdigit() else args.source
    capture = cv2.VideoCapture(source)
    writer = trace = None
    frames = ocr_batches = ocr_crops = eligible_crops = 0
    visible_frames = consensus_frames = 0
    consensus_events = 0
    raw_frames = raw_boxes = strong_boxes = track_observations = max_tracks = 0
    ids_seen, readings_by_id = set(), {}
    track_times, ocr_times, loop_times = [], [], []
    stages = Counter()
    scheduler = TemporalOCR(args.ocr_interval, args.stable_interval)
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    if not 0 < source_fps < 1000:
        source_fps = 30.
    print(f"Device {device} | FP32, imgsz {args.imgsz} | {detection_mode} | {ocr_name} {mode}", flush=True)
    try:
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Cannot read {args.source}")
        # Warm up all three networks even if the first source frame has no plate.
        detector.predict(frame, **kwargs)
        reader.read_many([np.full((64, 128, 3), 127, np.uint8)])
        if adaptive is not None:
            adaptive.predict(frame, args.high_imgsz)
        else:
            detector.track(frame, persist=True, tracker=str(args.tracker), **kwargs)
            for tracker in detector.predictor.trackers:
                tracker.reset()
        writer = cv2.VideoWriter(str(folder / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                 source_fps, (frame.shape[1], frame.shape[0]))
        if not writer.isOpened():
            raise RuntimeError("Cannot create output video")
        trace = (folder / "frames.jsonl").open("w", encoding="utf-8")
        wall_start = time.perf_counter()
        while ok and (not args.max_frames or frames < args.max_frames):
            loop_start = time.perf_counter()
            synchronize(device)
            start = time.perf_counter()
            detection_info = None
            if adaptive is not None:
                result, raw_detections, detection_info = adaptive.step(frame, frames)
            else:
                result = detector.track(frame, persist=True, tracker=str(args.tracker), **kwargs)[0]
            tracks = observations(result)
            raw_frames += bool(raw_detections)
            raw_boxes += len(raw_detections)
            strong_boxes += sum(box["confidence"] >= .6 for box in raw_detections)
            track_observations += len(tracks)
            max_tracks = max(max_tracks, len(tracks))
            synchronize(device)
            track_ms = (time.perf_counter() - start) * 1000
            track_times.append(track_ms)
            scheduler.observe([track["id"] for track in tracks], frames)
            ids_seen.update(track["id"] for track in tracks)
            eligible_crops += len(tracks)
            visible_frames += bool(tracks)
            candidates = []
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if tracks else None
            h, w = frame.shape[:2]
            for track in tracks:
                identity = track["id"]
                if not scheduler.due(identity, frames, args.ocr_every_frame):
                    continue
                x1, y1, x2, y2 = track["xyxy"]
                box = (int(np.clip(np.floor(x1), 0, w)), int(np.clip(np.floor(y1), 0, h)),
                       int(np.clip(np.ceil(x2), 0, w)), int(np.clip(np.ceil(y2), 0, h)))
                try:
                    crop, _ = crop_from_box(rgb, box, reader.config.margin)
                except ValueError:
                    continue
                blur = float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())
                if not args.ocr_every_frame and (track["confidence"] < args.ocr_conf or
                        crop.shape[0] < 16 or crop.shape[1] < 24 or blur < args.min_blur):
                    continue
                candidates.append((scheduler.states[identity].last_attempt, identity, crop, blur))
            candidates.sort(key=lambda item: (item[0], -item[3]))
            if not args.ocr_every_frame:
                candidates = candidates[:args.ocr_batch]
            batch_ms = 0.
            if candidates:
                start = time.perf_counter()
                readings, timing = reader.read_many([item[2] for item in candidates])
                synchronize(device)
                batch_ms = (time.perf_counter() - start) * 1000
                ocr_times.append(batch_ms)
                stages.update(timing)
                ocr_batches += 1
                ocr_crops += len(candidates)
                for (_, identity, _, _), reading in zip(candidates, readings):
                    state = scheduler.states[identity]
                    previous_consensus = (state.consensus, state.text)
                    scheduler.attempted(identity, frames)
                    scheduler.accept(identity, reading)
                    if state.consensus and previous_consensus != (True, state.text):
                        consensus_events += 1
                        agreeing = sum(text == state.text for text in state.votes)
                        print(
                            f"[CONSENSUS] frame={frames} | time={frames/source_fps:.2f}s | "
                            f"ID={identity} | plate={state.text} | "
                            f"votes={agreeing}/{len(state.votes)} | layout={state.layout}",
                            flush=True,
                        )
                    history = readings_by_id.setdefault(str(identity), [])
                    history.append(dict(frame=frames, **reading.as_dict()))
            annotated = frame.copy()
            rows = []
            any_consensus = False
            for track in tracks:
                state = scheduler.states[track["id"]]
                status = "CONSENSUS" if state.consensus else "READING"
                any_consensus |= state.consensus
                text = state.text or state.latest or "UNREAD"
                color = (40, 220, 40) if state.consensus else (0, 190, 255)
                x1, y1, x2, y2 = [round(value) for value in track["xyxy"]]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label = f"ID {track['id']} {text} {status}"
                cv2.putText(annotated, label, (max(0,x1), max(90,y1-8)),
                            cv2.FONT_HERSHEY_SIMPLEX, .44, color, 1, cv2.LINE_AA)
                rows.append(dict(**track, text=state.text, latest=state.latest, layout=state.layout,
                                 consensus=state.consensus, attempts=state.attempts, error=state.error))
            consensus_frames += any_consensus
            elapsed_ms = (time.perf_counter() - loop_start) * 1000
            for line, label in enumerate([
                f"YOLO + ByteTrack + MobileNet + {ocr_name} | {detection_mode}",
                f"Process: {1000/max(elapsed_ms,.001):.1f} FPS | dets: {len(raw_detections)} | tracks: {len(tracks)}",
                f"YOLO calls: {sum(adaptive.counts[k] for k in ('base_calls','high_calls','roi_calls')) if adaptive else frames+1} | OCR crops: {ocr_crops} | batch: {batch_ms:.0f}ms",
            ]):
                cv2.putText(annotated, label, (7, 19+line*23), cv2.FONT_HERSHEY_SIMPLEX,
                            .43, (0,255,0), 1, cv2.LINE_AA)
            writer.write(annotated)
            trace.write(json.dumps(dict(frame=frames, timestamp_s=frames/source_fps, tracks=rows,
                                        detections=list(raw_detections), track_ms=track_ms,
                                        ocr_ms=batch_ms, adaptive=detection_info), ensure_ascii=False)+"\n")
            quit_requested = False
            if args.show:
                cv2.imshow(f"YOLO + ByteTrack + {ocr_name}", annotated)
                quit_requested = cv2.waitKey(1) & 0xFF in (ord("q"), 27)
            frames += 1
            # Include decoding the next frame in loop timing; exclude deliberate pacing.
            ok, frame = capture.read() if not quit_requested else (False, None)
            loop_ms = (time.perf_counter()-loop_start)*1000
            loop_times.append(loop_ms)
            if args.paced:
                time.sleep(max(0, 1/source_fps-loop_ms/1000))
            if frames % 60 == 0:
                print(f"Frames {frames} | OCR crops {ocr_crops}", flush=True)
        wall_seconds = time.perf_counter()-wall_start
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if trace is not None:
            trace.close()
        if args.show:
            cv2.destroyAllWindows()
    def p95(values):
        return float(np.percentile(values, 95)) if values else 0.
    summary = dict(
        mode=mode, source=str(Path(args.source).resolve()) if not isinstance(source,int) else args.source,
        output=str(folder.resolve()), frames=frames,
        yolo_calls=sum(adaptive.counts[k] for k in ('base_calls','high_calls','roi_calls')) if adaptive else frames,
        detection_mode=detection_mode, adaptive_counts=dict(adaptive.counts) if adaptive else {},
        adaptive_config=dict(high_imgsz=args.high_imgsz, high_interval=args.high_interval,
                             roi_interval=args.roi_interval, max_rois=args.max_rois) if adaptive else None,
        ocr_batches=ocr_batches,
        ocr_crops=ocr_crops, visible_track_observations=eligible_crops,
        skipped_ocr_percent=100*(1-ocr_crops/eligible_crops) if eligible_crops else None,
        unique_ids=len(ids_seen), frames_with_tracks=visible_frames, frames_with_consensus=consensus_frames,
        consensus_events=consensus_events,
        raw_detection_frames=raw_frames, raw_detection_boxes=raw_boxes, raw_boxes_ge_06=strong_boxes,
        track_observations=track_observations, max_simultaneous_tracks=max_tracks,
        processing_fps=frames/(sum(loop_times)/1000) if loop_times else 0.,
        wall_fps=frames/wall_seconds if wall_seconds else 0., loop_p95_ms=p95(loop_times),
        loop_max_ms=max(loop_times,default=0.),
        track_mean_ms=float(np.mean(track_times)) if track_times else 0.,
        ocr_batch_mean_ms=float(np.mean(ocr_times)) if ocr_times else 0.,
        ocr_batch_p95_ms=p95(ocr_times), ocr_stage_total_ms=dict(stages),
        device=str(device), precision="FP32", imgsz=args.imgsz, conf=args.conf,
        ocr_conf=args.ocr_conf, ocr_interval=args.ocr_interval, stable_interval=args.stable_interval,
        min_blur=args.min_blur, ocr_batch=args.ocr_batch, iou=args.iou, source_fps=source_fps, paced=args.paced,
        tracker_config=read_tracker_config(args.tracker),
        peak_cuda_allocated_mb=torch.cuda.max_memory_allocated(device)/2**20 if device.type=="cuda" else None,
        reader_config=vars(reader.config),
        ocr_backend=ocr_name,
        provenance=(reader.provenance if reader_factory else
                    json.loads((ROOT / "weights/ocr_provenance.json").read_text(encoding="utf-8"))),
        notes=["No ground truth: repeated agreement and nonempty OCR are not accuracy.",
               "Every-frame mode uses ByteTrack too; it bypasses crop quality gates and batch cap.",
               "Scheduled mode combines intervals and quality gates; this is not an interval-only comparison.",
               "All three networks run synchronously in FP32; OCR frames may have latency spikes.",
               "IDs are local tracks; re-entry/ID switches can change the ID or attach stale text.",
               "Saved video uses source FPS, not processing FPS. Peak allocated is PyTorch memory, not total VRAM."])
    (folder/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    (folder/"readings.json").write_text(json.dumps(readings_by_id,indent=2,ensure_ascii=False),encoding="utf-8")
    print(f"Saved {folder}\n{frames} frames | {summary['processing_fps']:.1f} FPS | OCR crops {ocr_crops}",flush=True)
    return summary


if __name__ == "__main__":
    main()
