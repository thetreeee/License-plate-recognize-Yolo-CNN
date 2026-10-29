"""Test image/video/webcam with YOLO -> layout MobileNet -> CRNN-CTC.

Examples (run from project root):
  python scripts/recognize_license_plate.py --source path/to/photo.jpg --device cpu
  python scripts/recognize_license_plate.py --source path/to/video.mp4 --device auto --show
  python scripts/recognize_license_plate.py --source 0 --show --max-frames 300

Outputs are saved in a new run directory; source images/videos are not modified.
No tracking or temporal voting is applied: every detected plate is read per frame.
"""
import argparse
import csv
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import re
import sys
import time

import cv2
import numpy as np
import torch

from lpr_models import PlateReader, check_weight_file, crop_from_box, synchronize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"}
DETECTION_FIELDS = [
    "frame_index","timestamp_s","detection_index","class_id","class_name","det_confidence",
    "x1","y1","x2","y2","crop_x1","crop_y1","crop_x2","crop_y2",
    "text","layout","layout_confidence","split_ok","split_method","error",
]
FRAME_FIELDS = [
    "frame_index","timestamp_s","plates","nonempty_texts","errors",
    "yolo_ms","crop_ms","layout_ms","preprocess_ms","crnn_decode_ms","inference_ms",
]


def resolve_device(value):
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


def read_image(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data,cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def write_image(path,image):
    ok,encoded = cv2.imencode(Path(path).suffix,image)
    if not ok:
        raise RuntimeError(f"Cannot encode image: {path}")
    encoded.tofile(str(path))


class LicensePlatePipeline:
    def __init__(self,args):
        from ultralytics import YOLO
        self.device = resolve_device(args.device)
        if self.device.type == "cpu":
            torch.set_num_threads(args.cpu_threads)
        self.detector = YOLO(str(check_weight_file(args.det_weights)),task="detect")
        if self.detector.task != "detect":
            raise ValueError("YOLO checkpoint must be an axis-aligned detection model")
        self.detector.model.float()
        self.reader = PlateReader(args.ocr_weights,args.layout_weights,self.device)
        self.conf,self.iou,self.imgsz = args.conf,args.iou,args.imgsz
        self.classes,self.max_det = args.classes,args.max_det
        self.detector_device = str(self.device.index) if self.device.type=="cuda" else "cpu"

    @torch.inference_mode()
    def process(self,frame,debug=False):
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Expected uint8 BGR frame")
        synchronize(self.device)
        start = time.perf_counter()
        result = self.detector.predict(
            source=frame,conf=self.conf,iou=self.iou,imgsz=self.imgsz,
            device=self.detector_device,classes=self.classes,
            max_det=self.max_det,verbose=False,save=False,
        )[0]
        boxes = result.boxes.data.detach().cpu().numpy() if result.boxes is not None else np.empty((0,6))
        synchronize(self.device)
        after_yolo = time.perf_counter()
        rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        h,w = frame.shape[:2]
        records,crops,owners = [],[],[]
        for detection_index,box in enumerate(boxes):
            x1,y1,x2,y2 = box[:4]
            clipped = (
                int(np.clip(np.floor(x1),0,w)),int(np.clip(np.floor(y1),0,h)),
                int(np.clip(np.ceil(x2),0,w)),int(np.clip(np.ceil(y2),0,h)),
            )
            cls = int(box[-1])
            record = dict(
                detection_index=detection_index,class_id=cls,class_name=str(result.names[cls]),
                det_confidence=float(box[-2]),**dict(zip(["x1","y1","x2","y2"],clipped)),
                text="",layout="",layout_confidence=None,split_ok=False,split_method="",error="",
            )
            records.append(record)
            try:
                crop,crop_box = crop_from_box(rgb,clipped,self.reader.config.margin)
                record.update(dict(zip(["crop_x1","crop_y1","crop_x2","crop_y2"],crop_box)))
                owners.append(len(records)-1)
                crops.append(crop)
            except ValueError as exc:
                record["error"] = str(exc)
        after_crop = time.perf_counter()
        readings,ocr_times = self.reader.read_many(crops,debug=debug)
        debug_images = {}
        for owner,reading in zip(owners,readings):
            records[owner].update(reading.as_dict())
            if reading.debug_images:
                debug_images[owner] = reading.debug_images
        synchronize(self.device)
        end = time.perf_counter()
        stats = dict(
            plates=len(records),nonempty_texts=sum(bool(r["text"]) for r in records),
            errors=sum(bool(r["error"]) for r in records),
            yolo_ms=(after_yolo-start)*1000,crop_ms=(after_crop-after_yolo)*1000,
            **ocr_times,inference_ms=(end-start)*1000,
        )
        return records,stats,debug_images


def annotate(frame,records,stats):
    canvas = frame.copy()
    h,w = canvas.shape[:2]
    for row in records:
        warning = bool(row["error"]) or not row["text"] or not row["split_ok"]
        color = (0,165,255) if warning else (50,220,50)
        x1,y1,x2,y2 = (row[k] for k in ["x1","y1","x2","y2"])
        cv2.rectangle(canvas,(x1,y1),(x2,y2),color,2)
        layout = {"one_row":"1-row","two_rows":"2-row"}.get(row["layout"],"?-row")
        text = row["text"] or "UNREAD"
        suffix = " CHECK" if warning else ""
        label = f"{text} | {layout} | det {row['det_confidence']:.2f}{suffix}"
        (tw,th),base = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,.55,1)
        tx = max(0,min(x1,max(0,w-tw-6)))
        ty = y1-7 if y1>th+10 else min(h-base-2,y2+th+5)
        cv2.rectangle(canvas,(tx,max(0,ty-th-4)),(min(w,tx+tw+6),min(h,ty+base+3)),(20,20,20),-1)
        cv2.putText(canvas,label,(tx+3,ty),cv2.FONT_HERSHEY_SIMPLEX,.55,color,1,cv2.LINE_AA)
    elapsed = stats["inference_ms"]
    hud = f"Plates: {stats['plates']} | Infer: {elapsed:.1f} ms | {1000/max(elapsed,1e-9):.1f} FPS"
    cv2.rectangle(canvas,(0,0),(min(w,780),30),(20,20,20),-1)
    cv2.putText(canvas,hud,(8,21),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA)
    return canvas


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source",required=True,help="Image/video path, webcam index, or RTSP URL")
    parser.add_argument("--det-weights",type=Path,default=PROJECT_ROOT/"weights/best.pt")
    parser.add_argument("--layout-weights",type=Path,default=PROJECT_ROOT/"weights/layout_best.pt")
    parser.add_argument("--ocr-weights",type=Path,default=PROJECT_ROOT/"weights/crnn_best.pt")
    parser.add_argument("--device",default="auto",help="auto, cpu, 0, or cuda:0")
    parser.add_argument("--imgsz",type=int,default=1280)
    parser.add_argument("--conf",type=float,default=.25,help="YOLO confidence threshold, not OCR confidence")
    parser.add_argument("--iou",type=float,default=.45)
    parser.add_argument("--classes",type=int,nargs="+",default=None,help="Optional YOLO plate class IDs")
    parser.add_argument("--max-det",type=int,default=20)
    parser.add_argument("--cpu-threads",type=int,default=4)
    parser.add_argument("--output-dir",type=Path,default=PROJECT_ROOT/"outputs/lpr")
    parser.add_argument("--show",action="store_true",help="Open preview; press q/Esc to stop")
    parser.add_argument("--save-crops",action="store_true",help="Save RGB crop/line/input diagnostics as PNG")
    parser.add_argument("--debug-limit",type=int,default=100,help="Maximum plate diagnostics to save per run")
    parser.add_argument("--max-frames",type=int,default=0,help="0=all frames")
    parser.add_argument("--warmup",type=int,default=1,help="Untimed passes on first frame")
    parser.add_argument("--log-every",type=int,default=30)
    parser.add_argument("--codec",default="mp4v",help="Four-character VideoWriter codec")
    args = parser.parse_args(argv)
    if not (0 < args.conf <= 1 and 0 < args.iou <= 1):
        parser.error("--conf and --iou must be in (0,1]")
    if min(args.imgsz,args.max_det,args.cpu_threads,args.log_every) < 1:
        parser.error("--imgsz, --max-det, --cpu-threads and --log-every must be positive")
    if min(args.max_frames,args.warmup,args.debug_limit) < 0 or len(args.codec)!=4:
        parser.error("Invalid frame/warmup/debug limit or codec")
    return args


def make_run_dir(root,source):
    slug = "camera_"+source if source.isdigit() else Path(source).stem
    slug = re.sub(r"[^A-Za-z0-9._-]","_",slug)[:80] or "source"
    result = root/(slug+"_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    result.mkdir(parents=True,exist_ok=False)
    return result


def save_debug_images(run_dir,frame_index,images,remaining):
    saved = 0
    for detection_index,stages in images.items():
        if saved >= remaining:
            break
        folder = run_dir/"crops"
        folder.mkdir(exist_ok=True)
        for name,rgb in stages.items():
            path = folder/f"frame_{frame_index:06d}_plate_{detection_index:02d}_{name}.png"
            write_image(path,cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
        saved += 1
    return saved


def run(args):
    source_path = Path(args.source)
    is_image = source_path.is_file() and source_path.suffix.lower() in IMAGE_EXTENSIONS
    is_camera = args.source.isdigit() and not source_path.is_file()
    if not is_camera and "://" not in args.source and not source_path.is_file():
        raise FileNotFoundError(f"Source does not exist: {args.source}")

    pipeline = LicensePlatePipeline(args)
    print(f"Device: {pipeline.device} | precision: FP32 | epochs: {pipeline.reader.epochs}")
    run_dir = make_run_dir(args.output_dir,args.source)
    capture,writer = None,None
    try:
        if is_image:
            frame = read_image(source_path)
            source_fps = None
        else:
            capture = cv2.VideoCapture(int(args.source) if is_camera else args.source)
            if not capture.isOpened():
                raise RuntimeError(f"Cannot open video/camera: {args.source}")
            ok,frame = capture.read()
            if not ok:
                raise RuntimeError("Source opened but returned no frame")
            source_fps = float(capture.get(cv2.CAP_PROP_FPS))
            if not np.isfinite(source_fps) or source_fps<=0:
                source_fps = 30.
            height,width = frame.shape[:2]
            writer = cv2.VideoWriter(
                str(run_dir/"annotated.mp4"),cv2.VideoWriter_fourcc(*args.codec),
                source_fps,(width,height),
            )
            if not writer.isOpened():
                raise RuntimeError("Cannot create output video; try --codec mp4v")
        for _ in range(args.warmup):
            pipeline.process(frame)
        if args.show:
            cv2.namedWindow("YOLO + MobileNet + CRNN",cv2.WINDOW_NORMAL)

        frame_stats,debug_saved,frame_index = [],0,0
        stopped_by_user = False
        start = time.perf_counter()
        with (run_dir/"detections.csv").open("w",encoding="utf-8-sig",newline="") as dfh, \
             (run_dir/"frames.csv").open("w",encoding="utf-8-sig",newline="") as ffh:
            detection_csv = csv.DictWriter(dfh,fieldnames=DETECTION_FIELDS)
            frame_csv = csv.DictWriter(ffh,fieldnames=FRAME_FIELDS)
            detection_csv.writeheader(); frame_csv.writeheader()
            while True:
                timestamp = 0. if is_image else (
                    time.perf_counter()-start if is_camera else frame_index/source_fps
                )
                debug = args.save_crops and debug_saved<args.debug_limit
                records,stats,images = pipeline.process(frame,debug=debug)
                annotated = annotate(frame,records,stats)
                frame_row = dict(frame_index=frame_index,timestamp_s=timestamp,**stats)
                frame_csv.writerow(frame_row)
                for row in records:
                    detection_csv.writerow(dict(frame_index=frame_index,timestamp_s=timestamp,**row))
                frame_stats.append(frame_row)

                if is_image:
                    write_image(run_dir/"annotated.png",annotated)
                else:
                    if annotated.shape[:2] != (height,width):
                        raise RuntimeError("Video resolution changed during stream")
                    writer.write(annotated)
                if debug:
                    debug_saved += save_debug_images(
                        run_dir,frame_index,images,args.debug_limit-debug_saved
                    )
                if is_image or frame_index%args.log_every==0:
                    texts = ", ".join(r["text"] or "UNREAD" for r in records) or "(no plate detected)"
                    print(f"Frame {frame_index}: {texts} | {stats['inference_ms']:.1f} ms")
                frame_index += 1
                if args.show:
                    cv2.imshow("YOLO + MobileNet + CRNN",annotated)
                    if cv2.waitKey(1)&0xFF in (ord("q"),27):
                        stopped_by_user = True
                        break
                if is_image or (args.max_frames and frame_index>=args.max_frames):
                    break
                ok,frame = capture.read()
                if not ok:
                    break
            wall_seconds = time.perf_counter()-start

        if writer is not None:
            writer.release()
            writer = None
        times = np.array([r["inference_ms"] for r in frame_stats])
        summary = dict(
            source=args.source,device=str(pipeline.device),precision="FP32",
            frames=len(frame_stats),plates=sum(r["plates"] for r in frame_stats),
            nonempty_texts=sum(r["nonempty_texts"] for r in frame_stats),
            inference_errors=sum(r["errors"] for r in frame_stats),
            frames_without_detection=sum(r["plates"]==0 for r in frame_stats),
            inference_mean_ms=float(times.mean()),inference_p95_ms=float(np.percentile(times,95)),
            inference_fps=len(times)/(float(times.sum())/1000),
            processing_fps=len(frame_stats)/max(wall_seconds,1e-9),
            wall_seconds=wall_seconds,stopped_by_user=stopped_by_user,
            warmup_passes=args.warmup,source_video_fps=source_fps,
            checkpoints=dict(yolo=str(args.det_weights.resolve()),layout=str(args.layout_weights.resolve()),
                             crnn=str(args.ocr_weights.resolve())),
            checkpoint_epochs=pipeline.reader.epochs,ocr_config=asdict(pipeline.reader.config),
            detection_config=dict(imgsz=args.imgsz,conf=args.conf,iou=args.iou,classes=args.classes,max_det=args.max_det),
            stage_mean_ms={key:float(np.mean([r[key] for r in frame_stats]))
                           for key in ["yolo_ms","crop_ms","layout_ms","preprocess_ms","crnn_decode_ms"]},
            notes=[
                "No ground-truth labels are available: nonempty_texts is NOT accuracy.",
                "Layout confidence and split_ok are not calibrated OCR correctness probabilities.",
                "Inference FPS excludes capture/drawing/output; processing FPS includes loop I/O but excludes model loading/warmup.",
                "Video output keeps source FPS and has no audio; offline processing FPS is not its playback FPS.",
                "No tracking, frame skipping, or temporal voting is applied.",
            ],
        )
        (run_dir/"summary.json").write_text(
            json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8"
        )
        print(f"Saved: {run_dir}")
        print(f"Frames: {summary['frames']} | plates: {summary['plates']} | "
              f"infer mean/P95: {summary['inference_mean_ms']:.1f}/{summary['inference_p95_ms']:.1f} ms | "
              f"processing FPS: {summary['processing_fps']:.1f}")
        return run_dir
    finally:
        if capture is not None:
            capture.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()


def main(argv=None):
    args = parse_args(argv)
    try:
        run(args)
    except (RuntimeError,ValueError,FileNotFoundError,OSError) as exc:
        print(f"ERROR: {exc}",file=sys.stderr)
        return 1
    return 0


if __name__=="__main__":
    raise SystemExit(main())
