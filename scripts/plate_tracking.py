"""Periodic detection and per-plate visual tracking (no OCR required)."""

from dataclasses import dataclass
from math import isfinite
from typing import Any, Callable


def clip_box(box, frame_shape):
    """Return an integer xywh box inside the image, or None if too small."""
    height, width = frame_shape[:2]
    x, y, w, h = box
    if not all(isfinite(value) for value in box) or w <= 0 or h <= 0:
        return None
    x1, y1 = max(0, round(x)), max(0, round(y))
    x2, y2 = min(width, round(x + w)), min(height, round(y + h))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return x1, y1, x2 - x1, y2 - y1


def box_iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    intersection = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(
        0, min(ay + ah, by + bh) - max(ay, by)
    )
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


class PlateAppearance:
    """Check current pixels against the last detector-confirmed plate crop.

    A small grayscale template tolerates small translations and brightness
    changes. It is a drift heuristic, not an OCR model or re-identification.
    Refresh only after YOLO confirmation so background drift cannot update it.
    """

    def __init__(self, frame, box, threshold=0.35):
        self.threshold = threshold
        crop = self._crop(frame, box)
        self.template = crop[4:-4, 4:-4].copy()

    @staticmethod
    def _crop(frame, box):
        import cv2

        x, y, w, h = box
        crop = frame[y:y + h, x:x + w]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 24), interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(gray, (3, 3), 0)

    def matches(self, frame, box):
        import cv2

        crop = self._crop(frame, box)
        # Constant patches give misleading correlation scores in OpenCV.
        if self.template.std() < 2 or crop.std() < 2:
            return False
        scores = cv2.matchTemplate(crop, self.template, cv2.TM_CCOEFF_NORMED)
        return cv2.minMaxLoc(scores)[1] >= self.threshold


@dataclass
class PlateTrack:
    track_id: int
    box: tuple
    confidence: float  # Last YOLO score, not a tracking confidence.
    tracker: Any
    last_detection: int
    visible: bool = True
    source: str = "YOLO"
    appearance: Any = None


class PlateTracker:
    """Track each frame; periodically reconcile detections using greedy IoU.

    Lost tracks retain their last box briefly for spatial reacquisition. This is
    short-term association, not appearance-based re-identification.
    """

    def __init__(self, tracker_factory: Callable, detect_interval=5,
                 match_iou=0.2, max_age=30, appearance_factory=None):
        if detect_interval < 1 or max_age < detect_interval:
            raise ValueError("Require 1 <= detect_interval <= max_age")
        if not 0 < match_iou <= 1:
            raise ValueError("match_iou must be in (0, 1]")
        self.tracker_factory = tracker_factory
        self.detect_interval = detect_interval
        self.match_iou = match_iou
        self.max_age = max_age
        self.appearance_factory = appearance_factory
        self.tracks = {}
        self.next_id = 1
        self.last_detection_frame = None

    def update(self, frame, frame_index):
        """Update existing trackers and return whether YOLO is needed now."""
        failed_now = False
        for track_id, track in list(self.tracks.items()):
            if frame_index - track.last_detection > self.max_age:
                del self.tracks[track_id]
                continue
            if not track.visible:
                continue
            ok, box = track.tracker.update(frame)
            raw_box = box
            box = clip_box(raw_box, frame.shape) if ok else None
            # Do not clip a mostly departed target into a tiny sticky edge box.
            if box is not None and box[2] * box[3] < 0.6 * raw_box[2] * raw_box[3]:
                box = None
            if box is not None and track.appearance is not None:
                if not track.appearance.matches(frame, box):
                    box = None
            if box is None:
                track.visible = False
                failed_now = True
            else:
                track.box = box
                track.source = "TRACK"
        return (
            self.last_detection_frame is None
            or frame_index - self.last_detection_frame >= self.detect_interval
            or failed_now
        )

    def reconcile(self, frame, detections, frame_index, *, prepared_tracks=None,
                  detected_at=None):
        """Correct matched tracks, create new IDs, and hide unmatched tracks.

        detections contains (xywh_box, YOLO_confidence) pairs. Async callers may
        transfer prepared trackers already advanced to frame_index. Ownership
        transfers to this manager; the worker must never update them again.
        detected_at identifies the actual source frame that YOLO observed.
        """
        detected_at = frame_index if detected_at is None else detected_at
        self.last_detection_frame = detected_at
        valid = []
        prepared = []
        for index, (box, confidence) in enumerate(detections):
            box = clip_box(box, frame.shape)
            if box is not None:
                valid.append((box, float(confidence)))
                if prepared_tracks is not None:
                    prepared.append(prepared_tracks[index])

        candidates = []
        for track_id, track in self.tracks.items():
            for index, (box, _) in enumerate(valid):
                overlap = box_iou(track.box, box)
                if overlap >= self.match_iou:
                    candidates.append((overlap, track_id, index))
        used_tracks, used_detections = set(), set()
        matches = {}
        for _, track_id, index in sorted(candidates, reverse=True):
            if track_id not in used_tracks and index not in used_detections:
                matches[index] = track_id
                used_tracks.add(track_id)
                used_detections.add(index)

        # YOLO did not confirm these boxes. Hide them now rather than showing
        # potentially stale CSRT results until max_age; retain IDs for recovery.
        for track_id, track in self.tracks.items():
            if track_id not in used_tracks:
                track.visible = False

        for index, (box, confidence) in enumerate(valid):
            track_id = matches.get(index)
            if track_id is None:
                track_id = self.next_id
                self.next_id += 1
            if prepared_tracks is None:
                tracker = self.tracker_factory()
                # OpenCV init returns None in some versions, bool in others.
                if tracker.init(frame, box) is False:
                    continue
                appearance = (self.appearance_factory(frame, box)
                              if self.appearance_factory is not None else None)
            else:
                tracker = prepared[index].tracker
                appearance = prepared[index].appearance
            self.tracks[track_id] = PlateTrack(
                track_id, box, confidence, tracker, detected_at,
                source="YOLO" if detected_at == frame_index else "TRACK",
                appearance=appearance,
            )

    @property
    def visible_tracks(self):
        return [track for track in self.tracks.values() if track.visible]
