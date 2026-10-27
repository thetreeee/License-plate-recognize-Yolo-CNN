"""Optional integration test with real CSRT on a moving textured target."""

import unittest

from scripts.plate_tracking import PlateAppearance, PlateTracker, box_iou
from scripts.async_detector import FrameHistory, propagate_detections


class CSRTIntegrationTests(unittest.TestCase):
    def test_optical_flow_moves_delayed_detection_to_current_image(self):
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("OpenCV/NumPy not installed")
        history = FrameHistory()
        texture = np.random.default_rng(42).integers(0, 256, (40, 100, 3), dtype=np.uint8)
        for index in range(6):
            frame = np.zeros((240, 400, 3), dtype=np.uint8)
            x = 40 + index * 3
            frame[80:120, x:x + 100] = texture
            history.append(index, frame)
        moved = propagate_detections(history.span(0), [((40, 80, 100, 40), .9)], PlateAppearance)
        self.assertEqual(len(moved), 1)
        self.assertGreater(box_iou(moved[0][0], (55, 80, 100, 40)), .85)
        # No trustworthy features/appearance remain after the object disappears.
        history.append(6, np.zeros((240, 400, 3), dtype=np.uint8))
        self.assertEqual(propagate_detections(history.span(0),
                                            [((40, 80, 100, 40), .9)], PlateAppearance), [])

    def test_fast_csrt_profile_tracks_a_moving_target(self):
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("OpenCV/NumPy not installed")
        if not hasattr(cv2, "TrackerCSRT_create"):
            self.skipTest("opencv-contrib-python required")
        from scripts.realtime_lp_detect import csrt_factory
        texture = np.random.default_rng(7).integers(0, 256, (40, 100, 3), dtype=np.uint8)
        manager = PlateTracker(csrt_factory, appearance_factory=PlateAppearance)
        for index in range(12):
            frame = np.zeros((240, 400, 3), dtype=np.uint8)
            x = 40 + index * 2
            frame[80:120, x:x + 100] = texture
            if manager.update(frame, index):
                manager.reconcile(frame, [((x, 80, 100, 40), .9)], index)
            self.assertEqual(len(manager.visible_tracks), 1)
            self.assertGreater(box_iou(manager.visible_tracks[0].box, (x, 80, 100, 40)), .7)

    def test_real_tracker_between_detector_frames(self):
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("OpenCV/NumPy not installed")
        factory = getattr(cv2, "TrackerCSRT_create", None)
        if factory is None:
            factory = getattr(getattr(cv2, "legacy", None), "TrackerCSRT_create", None)
        if factory is None:
            self.skipTest("opencv-contrib-python required for CSRT")

        texture = np.random.default_rng(42).integers(0, 256, (40, 100, 3), dtype=np.uint8)
        manager = PlateTracker(factory, appearance_factory=PlateAppearance)
        detector_frames = []
        for index in range(16):
            frame = np.zeros((240, 400, 3), dtype=np.uint8)
            x = 40 + 2 * index
            frame[80:120, x:x + 100] = texture
            expected = (x, 80, 100, 40)
            if manager.update(frame, index):
                detector_frames.append(index)
                # Simulate detector measurements; CSRT itself is real.
                manager.reconcile(frame, [(expected, 0.9)], index)
            self.assertEqual([t.track_id for t in manager.visible_tracks], [1])
            self.assertGreater(box_iou(manager.visible_tracks[0].box, expected), 0.7)
        self.assertEqual(detector_frames, [0, 5, 10, 15])

    def test_disappearance_hides_box_even_when_tracker_claims_success(self):
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("OpenCV/NumPy not installed")

        class StickyTracker:
            def init(self, frame, box):
                self.box = box

            def update(self, frame):
                return True, self.box  # Reproduce a tracker stuck on background.

        manager = PlateTracker(StickyTracker, detect_interval=100, max_age=300,
                               appearance_factory=PlateAppearance)
        frame = np.zeros((240, 400, 3), dtype=np.uint8)
        frame[80:120, 40:140] = np.random.default_rng(42).integers(
            0, 256, (40, 100, 3), dtype=np.uint8
        )
        manager.reconcile(frame, [((40, 80, 100, 40), 0.9)], 0)
        self.assertFalse(manager.update(frame, 1))
        self.assertEqual(len(manager.visible_tracks), 1)
        empty = np.zeros_like(frame)
        self.assertTrue(manager.update(empty, 2))
        self.assertEqual(manager.visible_tracks, [])
        manager.reconcile(empty, [], 2)
        self.assertFalse(manager.update(empty, 3))
        self.assertEqual(manager.visible_tracks, [])

    def test_template_tolerates_brightness_change_but_rejects_other_texture(self):
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("OpenCV/NumPy not installed")
        rng = np.random.default_rng(42)
        frame = rng.integers(30, 180, (80, 160, 3), dtype=np.uint8)
        box = (20, 20, 100, 40)
        appearance = PlateAppearance(frame, box)
        self.assertTrue(appearance.matches(frame + 30, box))
        other = rng.integers(30, 180, frame.shape, dtype=np.uint8)
        self.assertFalse(appearance.matches(other, box))


if __name__ == "__main__":
    unittest.main()
