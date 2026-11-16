"""ByteTrack with a conservative extra match for very short gaps."""
from collections import deque

import numpy as np
from ultralytics.trackers.basetrack import TrackState
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.trackers.utils import matching


class ShortGapByteTracker(BYTETracker):
    def __init__(self, args, frame_rate=30):
        super().__init__(args, frame_rate)
        self.history = {}
        self.last_relinks = []

    def reset(self):
        super().reset()
        self.history.clear()
        self.last_relinks = []

    def update(self, results, img=None, feats=None):
        high = results.conf >= self.args.track_high_thresh
        low = (results.conf > self.args.track_low_thresh) & ~high
        self._high_rows, self._low_rows = results.xyxy[high], results.xyxy[low]
        self._low_results = results[low]
        self._first_pass = True
        self.last_relinks = []
        tracks = super().update(results, img, feats)
        for track in self.tracked_stracks:
            if track.end_frame == self.frame_id:
                rows = self._high_rows if track.score >= self.args.track_high_thresh else self._low_rows
                self.history.setdefault(track.track_id, deque(maxlen=3)).append(
                    (self.frame_id, rows[int(track.idx)].copy()))
        self.history = {key: seen for key, seen in self.history.items()
                        if self.frame_id - seen[-1][0] <= 3}
        return tracks

    def get_dists(self, tracks, detections):
        costs = super().get_dists(tracks, detections)
        first_pass, self._first_pass = self._first_pass, False
        if not first_pass or not costs.size or not self.history:
            return costs
        threshold = self.args.match_thresh
        matches, missing, unused = matching.linear_assignment(costs, thresh=threshold)
        if not len(missing) or not len(unused):
            return costs

        # Do not take over a detection that can confirm an existing tentative track.
        tentative = [track for track in self.tracked_stracks if not track.is_activated]
        candidates = [detections[j] for j in unused]
        reserved = (super().get_dists(tentative, candidates) <= .7).any(axis=0)
        low_detections = self.init_track(self._low_results)
        pairs = []
        for i in missing:
            track = tracks[i]
            seen = self.history.get(track.track_id, ())
            if not track.is_activated or len(seen) < 2:
                continue
            gap = self.frame_id - seen[-1][0]
            if not 1 <= gap <= 4:
                continue
            if track.state == TrackState.Tracked and low_detections:
                weak = matching.iou_distance([track], low_detections)
                if self.args.fuse_score:
                    weak = matching.fuse_score(weak, low_detections)
                if (weak <= .5).any():
                    continue
            first_frame, first = seen[0]
            last_frame, last = seen[-1]
            size = np.median([box[2:] - box[:2] for _, box in seen], axis=0)
            if (size <= 0).any():
                continue
            center = (last[:2] + last[2:]) / 2
            velocity = (center - (first[:2] + first[2:]) / 2) / (last_frame - first_frame)
            predicted = center + velocity * gap
            # shortcut: linear motion over a few frames; revisit if camera cuts or abrupt turns dominate.
            for col, j in enumerate(unused):
                det = detections[j]
                if reserved[col] or det.cls != track.cls or det.score < self.args.new_track_thresh:
                    continue
                box = det.xyxy
                scale = (box[2:] - box[:2]) / size
                offset = ((box[:2] + box[2:]) / 2 - predicted) / size
                if (scale >= .6).all() and (scale <= 1.7).all() and np.linalg.norm(offset) <= .65:
                    pairs.append((i, j))

        relinks = [(i, j) for i, j in pairs
                   if sum(a == i for a, _ in pairs) == 1 and sum(b == j for _, b in pairs) == 1]
        if not relinks:
            return costs
        # Keep the original strong matches fixed when ByteTrack solves the matrix again.
        fixed = np.full_like(costs, threshold + 1.)
        for i, j in matches:
            fixed[i, j] = costs[i, j]
        for i, j in relinks:
            fixed[i, j] = 0.
            self.last_relinks.append(dict(id=int(tracks[i].track_id),
                                          xyxy=detections[j].xyxy.tolist(),
                                          confidence=float(detections[j].score)))
        return fixed
