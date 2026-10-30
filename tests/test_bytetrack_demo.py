"""Regressions for weak detections, ID continuity and hidden lost tracks."""

from pathlib import Path
from types import SimpleNamespace
import unittest

from scripts.realtime_lp_bytetrack import observations, parse_args, read_tracker_config


class TensorStub:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class DisplayTests(unittest.TestCase):
    def test_unconfirmed_proposals_never_display_as_tracks(self):
        boxes = SimpleNamespace(is_track=False, xyxy=TensorStub([[10, 10, 40, 30]]),
                                conf=TensorStub([.15]), id=None)
        result = SimpleNamespace(boxes=boxes)
        self.assertEqual(observations(result, tracking=True), [])
        self.assertEqual(len(observations(result, tracking=False)), 1)

    def test_track_id_and_low_confidence_match_are_preserved(self):
        boxes = SimpleNamespace(is_track=True, xyxy=TensorStub([[10, 10, 40, 30]]),
                                conf=TensorStub([.2]), id=TensorStub([7.0]))
        record = observations(SimpleNamespace(boxes=boxes))[0]
        self.assertEqual(record["id"], 7)
        self.assertEqual(record["confidence"], .2)

    def test_cli_defaults_preserve_low_score_candidates(self):
        args = parse_args([])
        self.assertEqual(args.conf, .1)
        self.assertEqual(args.imgsz, 640)
        self.assertEqual(args.mode, "track")


class ByteTrackIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import numpy as np
            from ultralytics.engine.results import Boxes
            from ultralytics.trackers.byte_tracker import BYTETracker
        except ImportError as error:
            raise unittest.SkipTest(f"ByteTrack dependencies unavailable: {error}")
        cls.np, cls.Boxes, cls.BYTETracker = np, Boxes, BYTETracker

    def setUp(self):
        config = read_tracker_config(Path(__file__).resolve().parents[1] / "configs/bytetrack.yaml")
        self.tracker = self.BYTETracker(SimpleNamespace(**config), frame_rate=30)

    def update(self, rows):
        data = self.np.array(rows, dtype=self.np.float32).reshape(-1, 6)
        detections = self.Boxes(data, orig_shape=(240, 400)).cpu().numpy()
        return self.tracker.update(detections)

    def test_low_score_object_does_not_create_a_new_id(self):
        for _ in range(3):
            self.assertEqual(len(self.update([[10, 20, 90, 50, .2, 0]])), 0)

    def test_low_score_detection_can_continue_existing_id(self):
        first = self.update([[10, 20, 90, 50, .9, 0]])
        identity = int(first[0, 4])
        second = self.update([[12, 20, 92, 50, .2, 0]])
        self.assertEqual(len(second), 1)
        self.assertEqual(int(second[0, 4]), identity)

    def test_lost_track_is_hidden_and_recovered_within_buffer(self):
        first = self.update([[10, 20, 90, 50, .9, 0]])
        identity = int(first[0, 4])
        self.assertEqual(len(self.update([])), 0)
        recovered = self.update([[12, 20, 92, 50, .9, 0]])
        self.assertEqual(int(recovered[0, 4]), identity)

    def test_detection_order_does_not_determine_identity(self):
        first = self.update([[10, 20, 90, 50, .9, 0], [210, 20, 290, 50, .9, 0]])
        identity_left = int(first[first[:, 0].argmin(), 4])
        second = self.update([[212, 20, 292, 50, .9, 0], [12, 20, 92, 50, .9, 0]])
        self.assertEqual(len(second), 2)
        self.assertEqual(int(second[second[:, 0].argmin(), 4]), identity_left)


if __name__ == "__main__":
    unittest.main()
