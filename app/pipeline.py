import base64
import os
from collections import deque

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from scripts.adaptive_detection import AdaptiveDetector
from scripts.lpr_models import check_weight_file, crop_from_box
from scripts.lprnet_reader import LPRReader
from scripts.ocr_crops import BestCrops, crop_quality
from scripts.realtime_lp_ocr import OCR_CROP_WIDTH_RATIO, ROOT, observations, parse_args, read_tracker_config, resolve_device
from scripts.temporal_ocr import TemporalOCR


def jpeg(image):
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("Không thể mã hóa ảnh.")
    return base64.b64encode(encoded).decode("ascii")


class Models:
    def __init__(self):
        torch.set_num_threads(4)
        cv2.setNumThreads(1)
        self.args = parse_args([])
        self.device = resolve_device(os.getenv("PLATE_DEVICE", "cuda:0"))
        self.detector = YOLO(str(check_weight_file(self.args.weights)), task="detect")
        self.reader = LPRReader(ROOT / "weights/lprnet_v5.pth", self.args.layout_weights, self.device)
        self.config = read_tracker_config(self.args.tracker)
        self.options = dict(device=str(self.device), imgsz=self.args.imgsz, conf=self.args.conf,
                            iou=self.args.iou, classes=[0], half=False, verbose=False)
        blank = np.full((64, 128, 3), 127, np.uint8)
        self.detector.predict(blank, **self.options)
        self.reader.read_many([blank])


class Session:
    def __init__(self, models, *, uploaded_video=False, source_fps=30):
        self.models = models
        args = models.args
        options = models.options
        if uploaded_video and models.device.type == "cuda":
            options = {**options, "imgsz": 960}
        self.detector = AdaptiveDetector(models.detector, options, models.config,
                                         args.high_imgsz, args.high_interval, args.roi_interval,
                                         args.max_rois, relink=args.relink, search_rois=args.search_roi)
        self.scheduler = TemporalOCR(args.ocr_interval, args.stable_interval)
        self.crops = {}
        self.crop_buffer = BestCrops(max_age=max(1, round(source_fps)))
        self.display_ids = {}
        self.frame = 0

    def step(self, frame, seconds):
        args = self.models.args
        gray = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        result, _, detection_info = self.detector.step(gray, self.frame)
        tracks = observations(result)
        self.scheduler.observe([track["id"] for track in tracks], self.frame)
        self.crops = {key: value for key, value in self.crops.items() if key in self.scheduler.states}
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if tracks else None
        height, width = frame.shape[:2]
        reader = self.models.reader
        candidates = []
        matched_boxes = detection_info.get("matched_boxes", {})
        self.crop_buffer.retain([track["id"] for track in tracks], self.frame)
        for track in tracks:
            identity = track["id"]
            self.display_ids.setdefault(identity, len(self.display_ids) + 1)
            x1, y1, x2, y2 = track["xyxy"]
            detection_box = matched_boxes.get(identity)
            # Use the fresh crop when tracking has added substantial background.
            if detection_box is not None and x2 - x1 >= OCR_CROP_WIDTH_RATIO * (
                    detection_box[2] - detection_box[0]):
                x1, y1, x2, y2 = detection_box
            box = (int(np.clip(np.floor(x1), 0, width)), int(np.clip(np.floor(y1), 0, height)),
                   int(np.clip(np.ceil(x2), 0, width)), int(np.clip(np.ceil(y2), 0, height)))
            try:
                crop, _ = crop_from_box(rgb, box, reader.config.margin)
                if (track["confidence"] >= args.ocr_conf and
                        0 < x1 < x2 < width and 0 < y1 < y2 < height and
                        crop.shape[0] >= 16 and crop.shape[1] >= 24):
                    blur, quality = crop_quality(crop)
                    if blur >= args.min_blur:
                        self.crop_buffer.offer(identity, self.frame, crop, quality)
            except ValueError:
                pass
            selected = self.crop_buffer.pending.get(identity)
            if selected is not None and self.scheduler.due(identity, self.frame):
                candidates.append((self.scheduler.states[identity].last_attempt, identity, selected))

        candidates.sort(key=lambda item: (item[0], -item[2].quality))
        candidates = candidates[:args.ocr_batch]
        events = []
        if candidates:
            readings, _ = reader.read_many([item[2].image for item in candidates])
            for (_, identity, selected), reading in zip(candidates, readings):
                self.crop_buffer.consume(identity)
                state = self.scheduler.states[identity]
                previous = state.consensus, state.text
                self.scheduler.attempted(identity, self.frame)
                self.scheduler.accept(identity, reading)
                samples = self.crops.setdefault(identity, deque(maxlen=state.votes.maxlen))
                samples.append((state.votes[-1], selected.quality,
                                selected.image.copy() if state.votes[-1] else None))
                if state.consensus and previous != (True, state.text):
                    _, _, best = max((sample for sample in samples if sample[0] == state.text),
                                     key=lambda sample: sample[1])
                    events.append(dict(id=self.display_ids[identity], plate=state.text,
                                       status="CONSENSUS", seconds=round(seconds, 2),
                                       crop=jpeg(cv2.cvtColor(best, cv2.COLOR_RGB2BGR))))

        for track in tracks:
            identity = track["id"]
            state = self.scheduler.states[identity]
            status = "CONSENSUS" if state.consensus else "READING"
            color = (40, 220, 40) if state.consensus else (0, 190, 255)
            x1, y1, x2, y2 = map(round, track["xyxy"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID {self.display_ids[identity]} {state.text or state.latest or 'UNREAD'} {status}"
            cv2.putText(frame, label, (max(0, x1), max(16, y1 - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1, cv2.LINE_AA)
        self.frame += 1
        return frame, events
