"""A single YOLO worker with bounded history and delayed-tracker handoff.

Frames in FrameHistory are immutable after publication. Each CSRT instance has
one owner: the worker until poll(), then the foreground tracker manager.
"""

from collections import deque
from dataclasses import dataclass
from threading import Condition, Event, Lock, Thread
import time

try:
    from .plate_tracking import PlateTracker, clip_box
except ImportError:
    from plate_tracking import PlateTracker, clip_box


class StaleResult(Exception):
    """Missing frame history or a result too old to apply safely."""


@dataclass(frozen=True)
class VideoFrame:
    index: int
    image: object
    captured_at: float


class FrameHistory:
    def __init__(self, capacity=32):
        if capacity < 2:
            raise ValueError("History capacity must be >= 2")
        self._frames = deque(maxlen=capacity)
        self._lock = Lock()

    def append(self, index, image, captured_at=None):
        frame = VideoFrame(index, image, time.perf_counter() if captured_at is None else captured_at)
        with self._lock:
            if self._frames and index != self._frames[-1].index + 1:
                raise ValueError("Frame indices must be consecutive")
            self._frames.append(frame)
        return frame

    def span(self, first, last=None):
        with self._lock:
            frames = list(self._frames)
        if not frames:
            raise StaleResult("Frame history is empty")
        last = frames[-1].index if last is None else last
        if first < frames[0].index or last > frames[-1].index or first > last:
            raise StaleResult("Required frames have expired")
        return frames[first - frames[0].index:last - frames[0].index + 1]


@dataclass
class DetectionResult:
    source: VideoFrame
    aligned_frame: int
    tracks: list


class DetectionSchedule:
    """Throttle requests separately from completed detections."""

    def __init__(self, interval, search_interval=5):
        self.interval = interval
        self.search_interval = min(interval, search_interval)
        self.last_request = None

    def due(self, frame_index, has_tracks, lost_now=False):
        if self.last_request is None:
            return True
        gap = frame_index - self.last_request
        interval = self.interval if has_tracks else self.search_interval
        return gap >= interval or (lost_now and gap >= self.search_interval)

    def requested(self, frame_index):
        self.last_request = frame_index


def propagate_detections(frames, detections, appearance_factory=None):
    """Move boxes through buffered images with sparse LK flow and robust affine.

    This lightweight replay runs on the worker. CSRT is initialized only on the
    destination image. Poor feature support is rejected rather than guessed.
    """
    import cv2
    import numpy as np

    first = frames[0].image
    previous = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    states = []
    for box, confidence in detections:
        box = clip_box(box, first.shape)
        if box is None:
            continue
        x, y, w, h = box
        features = cv2.goodFeaturesToTrack(previous[y:y + h, x:x + w],
                                          maxCorners=60, qualityLevel=.01, minDistance=3)
        if features is None or len(features) < 4:
            if len(frames) > 1:
                continue
        else:
            features += np.array([x, y], dtype=np.float32)
        appearance = appearance_factory(first, box) if appearance_factory else None
        states.append([box, confidence, features, appearance])

    for frame in frames[1:]:
        gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
        surviving = []
        for box, confidence, points, appearance in states:
            if points is None:
                continue
            moved, status, _ = cv2.calcOpticalFlowPyrLK(
                previous, gray, points, None, winSize=(21, 21), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, .03),
            )
            if moved is None:
                continue
            # Forward/backward agreement rejects features lost during a pan.
            back, reverse_status, _ = cv2.calcOpticalFlowPyrLK(
                gray, previous, moved, None, winSize=(21, 21), maxLevel=3,
            )
            if back is None:
                continue
            good = ((status.ravel() == 1) & (reverse_status.ravel() == 1)
                    & (np.linalg.norm(back - points, axis=2).ravel() < 2))
            old, new = points[good], moved[good]
            if len(old) < 4:
                continue
            affine, inliers = cv2.estimateAffinePartial2D(
                old, new, method=cv2.RANSAC, ransacReprojThreshold=3,
            )
            if affine is None or inliers.sum() < 4:
                continue
            scale = np.linalg.norm(affine[:, 0])
            if not .7 <= scale <= 1.4:
                continue
            x, y, w, h = box
            corners = np.float32([[[x, y], [x + w, y], [x + w, y + h], [x, y + h]]])
            corners = cv2.transform(corners, affine)[0]
            x1, y1 = corners.min(axis=0)
            x2, y2 = corners.max(axis=0)
            raw_box = (float(x1), float(y1), float(x2 - x1), float(y2 - y1))
            box = clip_box(raw_box, frame.image.shape)
            if box is None or box[2] * box[3] < .6 * raw_box[2] * raw_box[3]:
                continue
            surviving.append([box, confidence, new[inliers.ravel() == 1], appearance])
        states = surviving
        previous = gray
        if not states:
            break
    return [(box, confidence, appearance) for box, confidence, _, appearance in states
            if appearance is None or appearance.matches(frames[-1].image, box)]


class AsyncDetector:
    """Load the detector and do inference/CSRT catch-up in one daemon worker.

    predictor_factory runs on the worker and returns callable(image)->detections.
    Only one pending request and one completed result are retained. Model warmup
    belongs in predictor_factory and finishes before start() returns.
    """

    def __init__(self, predictor_factory, tracker_factory, history, *,
                 appearance_factory=None, max_age_frames=20, max_age_seconds=1.0,
                 propagate=None):
        self.predictor_factory = predictor_factory
        self.tracker_factory = tracker_factory
        self.appearance_factory = appearance_factory
        self.history = history
        self.max_age_frames = max_age_frames
        self.max_age_seconds = max_age_seconds
        self.propagate = propagate
        self._condition = Condition()
        self._ready = Event()
        self._stop = False
        self._pending = self._result = self._error = None
        self._busy = False
        self._stats = dict(calls=0, inference_seconds=0.0, replaced=0,
                           stale=0, applied=0, busy=False, latency_ms=0.0,
                           detected_boxes=0, prepared_boxes=0, applied_boxes=0)
        self._thread = Thread(target=self._run, name="plate-yolo", daemon=True)

    def start(self):
        self._thread.start()
        self._ready.wait()
        self._raise_error()
        return self

    def _raise_error(self):
        with self._condition:
            error = self._error
        if error is not None:
            raise RuntimeError("YOLO background worker failed") from error

    def submit(self, frame):
        self._raise_error()
        with self._condition:
            if self._stop:
                return False
            if self._pending is not None:
                self._stats["replaced"] += 1
            self._pending = frame
            self._condition.notify()
        return True

    def poll(self):
        self._raise_error()
        with self._condition:
            result, self._result = self._result, None
        return result

    def stats(self):
        with self._condition:
            return dict(self._stats, busy=self._busy)

    def mark_applied(self, result):
        with self._condition:
            self._stats["applied"] += 1
            self._stats["applied_boxes"] += len(result.tracks)
            self._stats["latency_ms"] = (time.perf_counter() - result.source.captured_at) * 1000

    def mark_stale(self):
        with self._condition:
            self._stats["stale"] += 1

    def _check_age(self, source, target_index):
        if (target_index - source.index > self.max_age_frames
                or time.perf_counter() - source.captured_at > self.max_age_seconds):
            raise StaleResult("Detection result is too old")

    def _prepare(self, source, detections):
        # Do expensive CSRT initialization and history replay off the UI thread.
        frames = self.history.span(source.index)
        self._check_age(source, frames[-1].index)
        temporary = PlateTracker(
            self.tracker_factory, detect_interval=1,
            max_age=self.max_age_frames + 1,
            appearance_factory=self.appearance_factory,
        )
        if self.propagate is None:
            temporary.reconcile(source.image, detections, source.index)
            aligned = source.index
        else:
            transported = self.propagate(frames, detections, self.appearance_factory)
            target = frames[-1]
            temporary.reconcile(target.image, [(box, conf) for box, conf, _ in transported],
                                target.index)
            for track, (_, _, appearance) in zip(temporary.visible_tracks, transported):
                track.appearance = appearance
            aligned = target.index
        # Initialization may take time: use a fresh history snapshot afterwards.
        frames = self.history.span(aligned)
        self._check_age(source, frames[-1].index)
        for frame in frames[1:]:
            with self._condition:
                if self._stop:
                    return None
            temporary.update(frame.image, frame.index)
            self._check_age(source, frame.index)
        if detections and not temporary.visible_tracks:
            # Losing every box while transporting an old positive detection is
            # not evidence that the current frame contains no plates.
            raise StaleResult("All detected boxes were lost during catch-up")
        return DetectionResult(source, frames[-1].index, temporary.visible_tracks)

    def _run(self):
        try:
            predict = self.predictor_factory()
            self._ready.set()
            while True:
                with self._condition:
                    self._condition.wait_for(lambda: self._stop or self._pending is not None)
                    if self._stop:
                        return
                    source, self._pending = self._pending, None
                    self._busy = True
                try:
                    # A pending frame can expire before inference even starts.
                    latest = self.history.span(source.index)[-1]
                    self._check_age(source, latest.index)
                    started = time.perf_counter()
                    detections = list(predict(source.image))
                    duration = time.perf_counter() - started
                    with self._condition:
                        self._stats["calls"] += 1
                        self._stats["inference_seconds"] += duration
                        self._stats["detected_boxes"] += len(detections)
                    result = self._prepare(source, detections)
                    if result is not None:
                        with self._condition:
                            self._stats["prepared_boxes"] += len(result.tracks)
                            if self._result is not None:
                                self._stats["stale"] += 1
                            self._result = result
                except StaleResult:
                    self.mark_stale()
                finally:
                    with self._condition:
                        self._busy = False
        except Exception as error:
            with self._condition:
                self._error = error
        finally:
            self._ready.set()

    def apply(self, result, manager, current, max_tail_frames=2):
        """Bound foreground catch-up; never place old boxes on a new frame."""
        try:
            self._check_age(result.source, current.index)
            if current.index - result.aligned_frame > max_tail_frames:
                raise StaleResult("Too many frames remain for foreground catch-up")
            if (manager.last_detection_frame is not None
                    and result.source.index <= manager.last_detection_frame):
                raise StaleResult("A newer detection was already applied")
            frames = self.history.span(result.aligned_frame, current.index)
            temporary = PlateTracker(self.tracker_factory, 1, max_age=self.max_age_frames + 1)
            temporary.tracks = {track.track_id: track for track in result.tracks}
            for frame in frames[1:]:
                temporary.update(frame.image, frame.index)
            self._check_age(result.source, current.index)
            tracks = temporary.visible_tracks
            if result.tracks and not tracks:
                raise StaleResult("All prepared boxes were lost before handoff")
            result.tracks = tracks
            manager.reconcile(
                current.image, [(track.box, track.confidence) for track in tracks], current.index,
                prepared_tracks=tracks, detected_at=result.source.index,
            )
            self.mark_applied(result)
            return True
        except StaleResult:
            self.mark_stale()
            return False

    def close(self, timeout=5):
        with self._condition:
            self._stop = True
            self._pending = None
            self._condition.notify_all()
        if self._thread.ident is not None:
            self._thread.join(timeout)
        return not self._thread.is_alive()
