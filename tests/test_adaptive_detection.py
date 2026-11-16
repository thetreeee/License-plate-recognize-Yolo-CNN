"""Geometry, scheduling and actual ByteTrack regressions without model weights."""
import unittest

import numpy as np
import torch
from ultralytics.engine.results import Results

from scripts.adaptive_detection import AdaptiveDetector, crop_region, merge_boxes


CONFIG = dict(track_high_thresh=.5, track_low_thresh=.1, new_track_thresh=.6,
              track_buffer=30, match_thresh=.8, fuse_score=False)


class FakeModel:
    def __init__(self, predictions):
        self.predictions = iter(predictions)
        self.calls = []

    def predict(self, image, **kwargs):
        self.calls.append((image.shape, kwargs["imgsz"]))
        rows = np.asarray(next(self.predictions), np.float32).reshape(-1, 6)
        return [Results(image, "test", {0: "plate"}, boxes=torch.from_numpy(rows))]


class AdaptiveTests(unittest.TestCase):
    def detector(self, predictions, **kwargs):
        return AdaptiveDetector(FakeModel(predictions), dict(imgsz=640, iou=.45), CONFIG, **kwargs)

    def test_crop_context_and_clip(self):
        self.assertEqual(crop_region(np.array([0, 0, 20, 10]), (100, 100)), (0, 0, 74, 69))

    def test_merge_suppresses_duplicates_clips_and_rejects_invalid(self):
        rows = np.array([[-5, 0, 50, 20, .7, 0], [0, 0, 50, 20, .9, 0],
                         [70, 60, 50, 20, .99, 0], [0, 0, np.nan, 30, .9, 0]], np.float32)
        merged = merge_boxes([rows], (100, 100), .45)
        self.assertEqual(len(merged), 1)
        self.assertAlmostEqual(merged[0, 4], .9)

    def test_high_scan_merges_and_updates_tracker_once(self):
        box = [100, 100, 150, 120, .8, 0]
        detector = self.detector([[box], [box], [], []])
        result, raw, info = detector.step(np.zeros((200, 300, 3), np.uint8), 0)
        self.assertEqual(len(raw), 1)
        self.assertTrue(result.boxes.is_track)
        self.assertEqual(len(result.boxes), 1)
        self.assertTrue(info["high_scan"])
        self.assertEqual(detector.tracker.frame_id, 1)
        result, raw, info = detector.step(np.zeros((200, 300, 3), np.uint8), 1)
        self.assertEqual(detector.tracker.frame_id, 2)
        self.assertEqual(len(result.boxes), 0)
        self.assertEqual(raw, [])  # Never reuse stale detections on the next frame.

    def test_recovery_restores_same_id_with_fresh_crop_detection(self):
        box = [100, 100, 150, 120, .8, 0]
        region = crop_region(np.array(box[:4]), (400, 600))
        x, y = region[:2]
        recovered = [102-x, 100-y, 152-x, 120-y, .55, 0]
        detector = self.detector([[box], [box], [], [], [], [recovered]])
        frame = np.zeros((400, 600, 3), np.uint8)
        initial, _, _ = detector.step(frame, 0)
        missing, _, _ = detector.step(frame, 1)
        result, raw, info = detector.step(frame, 2)
        self.assertEqual(len(missing.boxes), 0)
        self.assertEqual(result.boxes.id.tolist(), initial.boxes.id.tolist())
        np.testing.assert_allclose(raw[0]["xyxy"], [102, 100, 152, 120])
        self.assertEqual(info["recovery_ids"], initial.boxes.id.int().tolist())
        self.assertEqual(detector.tracker.frame_id, 3)

    def test_recovery_stops_after_three_frames_without_detection(self):
        box = [100, 100, 150, 120, .8, 0]
        detector = self.detector([[box], [box]] + [[]] * 7)
        frame = np.zeros((400, 600, 3), np.uint8)
        detector.step(frame, 0)
        for index in range(1, 5):
            result, raw, info = detector.step(frame, index)
            self.assertEqual(raw, [])
            self.assertEqual(len(result.boxes), 0)
            self.assertEqual(len(info["recovery_ids"]), int(index <= 3))
        self.assertEqual(detector.counts["recovery_calls"], 3)

    def test_roi_maps_coordinates_and_confirms_only_matching_seed(self):
        seed = [100, 100, 140, 120, .2, 0]
        region = crop_region(np.array(seed[:4]), (400, 600))
        local = [100-region[0], 100-region[1], 140-region[0], 120-region[1], .9, 0]
        detector = self.detector([[seed], [local, [0, 0, 8, 8, .99, 0]]])
        result, raw, info = detector.step(np.zeros((400, 600, 3), np.uint8), 5)
        self.assertEqual(info["roi_verified"], 1)
        self.assertEqual(len(raw), 1)
        np.testing.assert_allclose(raw[0]["xyxy"], seed[:4])
        self.assertEqual(len(result.boxes), 1)

    def test_weak_or_unrelated_crop_does_not_create_id(self):
        seed = [100, 100, 140, 120, .2, 0]
        detector = self.detector([[seed], [[44, 54, 84, 74, .4, 0], [0, 0, 8, 8, .99, 0]]])
        result, raw, info = detector.step(np.zeros((400, 600, 3), np.uint8), 5)
        self.assertEqual(info["roi_verified"], 0)
        self.assertEqual(len(result.boxes), 0)
        self.assertAlmostEqual(raw[0]["confidence"], .2)

    def test_budget_cooldown_and_validation(self):
        seeds = [[100, 100, 140, 120, .3, 0], [400, 100, 440, 120, .2, 0]]
        detector = self.detector([seeds, [], seeds, [], seeds, []], max_rois=1)
        frame = np.zeros((400, 600, 3), np.uint8)
        regions = [detector.step(frame, index)[2]["roi_regions"] for index in (5, 10, 15)]
        self.assertTrue(all(len(boxes) == 1 for boxes in regions))
        self.assertNotEqual(regions[0], regions[1])
        self.assertEqual(regions[0], regions[2])
        with self.assertRaises(ValueError):
            self.detector([], roi_interval=0)


if __name__ == "__main__":
    unittest.main()
