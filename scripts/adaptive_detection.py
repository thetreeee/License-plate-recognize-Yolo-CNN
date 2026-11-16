"""Budgeted full-frame/ROI detection, followed by one ByteTrack update per frame."""
from collections import Counter
from types import SimpleNamespace

import numpy as np
import torch
from torchvision.ops import batched_nms
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker


def overlap(box, boxes):
    """IoU in original-image coordinates."""
    intersection = np.maximum(0, np.minimum(box[2:4], boxes[:, 2:4]) -
                              np.maximum(box[:2], boxes[:, :2])).prod(axis=1)
    area = np.maximum(0, box[2:4] - box[:2]).prod()
    areas = np.maximum(0, boxes[:, 2:4] - boxes[:, :2]).prod(axis=1)
    return intersection / np.maximum(area + areas - intersection, 1e-6)


def crop_region(box, shape):
    """Keep context around a candidate, using original pixels (no crop resizing here)."""
    h, w = shape[:2]
    side = max(128, 3 * max(box[2] - box[0], box[3] - box[1]))
    cx, cy = (box[:2] + box[2:4]) / 2
    return (max(0, int(np.floor(cx-side/2))), max(0, int(np.floor(cy-side/2))),
            min(w, int(np.ceil(cx+side/2))), min(h, int(np.ceil(cy+side/2))))


def merge_boxes(groups, shape, threshold):
    rows = np.concatenate(groups).astype(np.float32, copy=True)
    h, w = shape[:2]
    rows[:, [0, 2]] = rows[:, [0, 2]].clip(0, w)
    rows[:, [1, 3]] = rows[:, [1, 3]].clip(0, h)
    rows = rows[np.isfinite(rows).all(axis=1) & (rows[:, 2] > rows[:, 0]) &
                (rows[:, 3] > rows[:, 1])]
    data = torch.from_numpy(rows)
    keep = batched_nms(data[:, :4], data[:, 4], data[:, 5], threshold)
    return rows[keep.numpy()]


class AdaptiveDetector:
    def __init__(self, model, kwargs, tracker_config, high_imgsz=1280,
                 high_interval=30, roi_interval=5, max_rois=2):
        if min(high_interval, roi_interval, max_rois) < 1 or high_imgsz <= kwargs["imgsz"]:
            raise ValueError("Positive intervals/ROI budget and high-imgsz > imgsz are required")
        self.model, self.kwargs = model, dict(kwargs)
        self.high_imgsz, self.high_interval = high_imgsz, high_interval
        self.roi_interval, self.max_rois = roi_interval, max_rois
        self.tracker = BYTETracker(SimpleNamespace(**tracker_config), frame_rate=30)
        self.strong = tracker_config["new_track_thresh"]
        self.counts = Counter()
        self.recent_regions = []

    def predict(self, image, size):
        return self.model.predict(image, **{**self.kwargs, "imgsz": size})[0]

    def step(self, frame, index):
        result = self.predict(frame, self.kwargs["imgsz"])
        base = result.boxes.data.detach().cpu().numpy().copy()
        groups = [base]
        info = dict(base_boxes=len(base), high_scan=False, roi_regions=[], roi_verified=0)
        self.counts["base_calls"] += 1
        self.recent_regions = [(i, box) for i, box in self.recent_regions
                               if index-i < 2*self.roi_interval]
        if index % self.high_interval == 0:
            # Do not stack high-resolution and crop work on the same frame.
            high = self.predict(frame, self.high_imgsz)
            groups.append(high.boxes.data.detach().cpu().numpy())
            info["high_scan"] = True
            self.counts["high_calls"] += 1
        elif index % self.roi_interval == 0:
            scale = self.kwargs["imgsz"] / max(frame.shape[:2])
            small = np.minimum(base[:, 2]-base[:, 0], base[:, 3]-base[:, 1]) * scale < 32
            candidates = base[small | (base[:, 4] < self.strong)]
            candidates = candidates[np.argsort(-candidates[:, 4], kind="stable")]
            for seed in candidates:
                region = np.array(crop_region(seed[:4], frame.shape))
                if any(overlap(region, old[None])[0] > .3 for _, old in self.recent_regions):
                    continue
                x1, y1, x2, y2 = region
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                refined = self.predict(crop, self.kwargs["imgsz"])
                rows = refined.boxes.data.detach().cpu().numpy().copy()
                rows[:, [0, 2]] += x1
                rows[:, [1, 3]] += y1
                # A crop must confirm its candidate geometrically; unrelated detections
                # and weaker crop predictions cannot promote a low-score proposal.
                rows = rows[(overlap(seed[:4], rows) >= .2) & (rows[:, 5] == seed[5]) &
                            (rows[:, 4] >= self.strong)]
                groups.append(rows)
                self.recent_regions.append((index, region))
                info["roi_regions"].append(region.tolist())
                info["roi_verified"] += len(rows)
                self.counts["roi_calls"] += 1
                if len(info["roi_regions"]) >= self.max_rois:
                    break
        # Keep weak original observations for ByteTrack's second association pass;
        # neither a small box nor an extra inference changes the new-ID threshold.
        merged = merge_boxes(groups, frame.shape, self.kwargs["iou"])
        raw = [dict(xyxy=row[:4].tolist(), confidence=float(row[4])) for row in merged]
        tracks = self.tracker.update(Boxes(merged, frame.shape[:2]), frame)
        result.update(boxes=torch.as_tensor(tracks[:, :-1]) if len(tracks)
                      else torch.empty((0, 7)))
        info["merged_boxes"] = len(merged)
        self.counts["roi_verified"] += info["roi_verified"]
        return result, raw, info
