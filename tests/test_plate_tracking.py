"""Behavioral tests for detection scheduling, association and loss recovery."""

import unittest
from types import SimpleNamespace

from scripts.plate_tracking import PlateTracker, clip_box


class FakeTracker:
    def init(self, frame, box):
        self.box = box
        self.ok = True

    def update(self, frame):
        return self.ok, self.box


class TrackingTests(unittest.TestCase):
    def setUp(self):
        self.frame = SimpleNamespace(shape=(240, 400, 3))
        self.manager = PlateTracker(FakeTracker)

    def detect(self, index, detections):
        self.manager.reconcile(self.frame, detections, index)

    def test_periodic_detection_even_when_scene_is_empty(self):
        calls = []
        for index in range(16):
            if self.manager.update(self.frame, index):
                calls.append(index)
                self.detect(index, [])
        self.assertEqual(calls, [0, 5, 10, 15])

    def test_second_plate_gets_new_id_without_resetting_first(self):
        self.detect(0, [((20, 50, 80, 30), 0.9)])
        self.manager.update(self.frame, 5)
        # Reversed detection order must not determine IDs.
        self.detect(5, [((220, 50, 80, 30), 0.8), ((22, 50, 80, 30), 0.95)])
        self.assertEqual(self.manager.tracks[1].box, (22, 50, 80, 30))
        self.assertEqual(self.manager.tracks[2].box, (220, 50, 80, 30))
        self.manager.update(self.frame, 10)
        self.detect(10, [((222, 50, 80, 30), 0.9), ((24, 50, 80, 30), 0.9)])
        self.assertEqual(set(self.manager.tracks), {1, 2})

    def test_failure_detects_immediately_and_can_recover_same_id(self):
        self.detect(0, [((20, 50, 80, 30), 0.9)])
        self.manager.tracks[1].tracker.ok = False
        self.assertTrue(self.manager.update(self.frame, 1))
        self.assertEqual(self.manager.visible_tracks, [])
        self.detect(1, [])
        # A lost track does not force expensive detection on every frame.
        self.assertFalse(self.manager.update(self.frame, 2))
        self.assertTrue(self.manager.update(self.frame, 6))
        self.detect(6, [((22, 50, 80, 30), 0.8)])
        self.assertEqual([t.track_id for t in self.manager.visible_tracks], [1])

    def test_expired_track_does_not_reuse_old_identity(self):
        self.detect(0, [((20, 50, 80, 30), 0.9)])
        self.manager.update(self.frame, 31)
        self.assertEqual(self.manager.tracks, {})
        self.detect(31, [((20, 50, 80, 30), 0.9)])
        self.assertEqual(set(self.manager.tracks), {2})

    def test_matching_is_one_to_one(self):
        self.detect(0, [((20, 50, 80, 30), 0.9)])
        self.detect(5, [((21, 50, 80, 30), 0.9), ((30, 50, 80, 30), 0.8)])
        self.assertEqual(set(self.manager.tracks), {1, 2})
        self.assertEqual(self.manager.tracks[1].box, (21, 50, 80, 30))

    def test_detector_miss_hides_stale_box_but_retains_id_for_recovery(self):
        self.detect(0, [((20, 50, 80, 30), 0.9)])
        self.manager.update(self.frame, 5)
        self.detect(5, [])
        self.assertEqual(self.manager.visible_tracks, [])
        self.assertIn(1, self.manager.tracks)
        self.manager.update(self.frame, 10)
        self.detect(10, [((22, 50, 80, 30), 0.9)])
        self.assertEqual([t.track_id for t in self.manager.visible_tracks], [1])

    def test_mostly_outside_box_is_hidden_even_if_tracker_reports_success(self):
        self.detect(0, [((300, 50, 80, 30), 0.9)])
        self.manager.tracks[1].tracker.box = (380, 50, 80, 30)
        self.assertTrue(self.manager.update(self.frame, 1))
        self.assertEqual(self.manager.visible_tracks, [])

    def test_invalid_tracker_coordinates_are_rejected(self):
        self.assertIsNone(clip_box((float('nan'), 0, 80, 30), self.frame.shape))

    def test_clip_or_reject_out_of_image_boxes(self):
        self.assertEqual(clip_box((-5, -2, 20, 10), self.frame.shape), (0, 0, 15, 8))
        self.assertIsNone(clip_box((401, 20, 20, 10), self.frame.shape))
        self.assertIsNone(clip_box((10, 20, 0, 10), self.frame.shape))


if __name__ == "__main__":
    unittest.main()
