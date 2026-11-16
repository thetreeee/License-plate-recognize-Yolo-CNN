"""Verify short-gap association without changing normal ByteTrack matches."""
from types import SimpleNamespace
import unittest

import numpy as np
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker

from scripts.short_gap_tracker import ShortGapByteTracker


class ShortGapTrackerTests(unittest.TestCase):
    def tracker(self, cls=ShortGapByteTracker):
        config = SimpleNamespace(track_high_thresh=.5, track_low_thresh=.1,
                                 new_track_thresh=.6, track_buffer=30,
                                 match_thresh=.8, fuse_score=False)
        return cls(config)

    def update(self, tracker, *rows):
        boxes = np.asarray(rows, dtype=np.float32).reshape(-1, 6)
        return [int(x[4]) for x in tracker.update(Boxes(boxes, (640, 640)))]

    def seed(self, tracker):
        for row in ([614.9, 412.5, 639.9, 448.6, .63, 0],
                    [610.6, 395.7, 638.9, 430.8, .60, 0],
                    [606.8, 383.2, 634.4, 417.5, .58, 0],
                    [608.2, 396.7, 633.1, 416.0, .83, 0]):
            ids = self.update(tracker, row)
        return ids

    def test_uses_measured_history_before_new_id(self):
        target = [602.8, 372, 630.2, 404.3, .67, 0]
        baseline = self.tracker(BYTETracker)
        self.assertEqual(self.seed(baseline), [1])
        self.assertEqual(self.update(baseline, target), [])
        tracker = self.tracker()
        self.assertEqual(self.seed(tracker), [1])
        self.assertEqual(self.update(tracker, target), [1])
        self.assertEqual(tracker.last_relinks[0]["id"], 1)
        self.assertEqual(self.update(tracker), [])

    def test_two_plausible_detections_are_not_relinked(self):
        tracker = self.tracker()
        self.assertEqual(self.seed(tracker), [1])
        self.update(tracker, [602.8, 372, 630.2, 404.3, .67, 0],
                    [604, 371, 632, 403, .72, 0])
        self.assertEqual(tracker.last_relinks, [])

    def test_relink_expires_and_reset_clears_history(self):
        tracker = self.tracker()
        self.seed(tracker)
        for _ in range(4):
            self.assertEqual(self.update(tracker), [])
        self.update(tracker, [602.8, 372, 630.2, 404.3, .67, 0])
        self.assertEqual(tracker.last_relinks, [])
        tracker.reset()
        self.assertEqual(tracker.history, {})
