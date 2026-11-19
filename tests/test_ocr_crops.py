import unittest

import cv2
import numpy as np

from scripts.ocr_crops import BestCrops, crop_quality
from scripts.temporal_ocr import TemporalOCR


class BestCropTests(unittest.TestCase):
    def setUp(self):
        self.image = np.full((40, 80, 3), 127, np.uint8)
        self.buffer = BestCrops(max_age=25)

    def test_keeps_first_clear_crop_as_plate_moves_away(self):
        for frame, quality in enumerate((100., 80., 60., 40., 20.)):
            self.buffer.retain([1], frame)
            self.buffer.offer(1, frame, self.image, quality)
        self.assertEqual(self.buffer.consume(1).frame, 0)

    def test_replaces_initial_blur_as_plate_approaches(self):
        for frame, quality in enumerate((20., 40., 60., 80., 100.)):
            self.buffer.retain([1], frame)
            self.buffer.offer(1, frame, self.image, quality)
        self.assertEqual(self.buffer.consume(1).frame, 4)

    def test_crop_from_first_frame_is_read_only_once_after_track_confirmation(self):
        scheduler = TemporalOCR()
        for frame in range(5):
            scheduler.observe([1], frame)
            self.buffer.retain([1], frame)
            self.buffer.offer(1, frame, self.image, 100. - frame)
        self.assertTrue(scheduler.due(1, 4))
        chosen = self.buffer.consume(1)
        scheduler.attempted(1, 4)
        self.buffer.offer(1, chosen.frame, self.image, 200.)
        self.assertNotIn(1, self.buffer.pending)
        self.buffer.offer(1, 5, self.image, 20.)
        self.assertEqual(self.buffer.pending[1].frame, 5)

    def test_expiry_and_missing_track_discard_pending_crops(self):
        self.buffer.offer(1, 0, self.image, 100.)
        self.buffer.offer(2, 1, self.image, 100.)
        self.buffer.retain([1, 2], 26)
        self.assertNotIn(1, self.buffer.pending)
        self.assertIn(2, self.buffer.pending)
        self.buffer.retain([1], 27)
        self.assertFalse(self.buffer.pending)

    def test_other_ids_and_batch_deferred_crops_keep_their_own_image(self):
        self.buffer.offer(1, 0, self.image, 100.)
        self.buffer.offer(2, 0, self.image, 80.)
        self.buffer.consume(1)
        self.buffer.retain([1, 2], 1)
        self.image[:] = 255
        self.assertTrue(np.all(self.buffer.pending[2].image == 127))
        self.assertEqual(self.buffer.pending[2].frame, 0)

    def test_quality_prefers_sharp_text_and_ignores_outer_border(self):
        sharp = np.full((64, 128, 3), 220, np.uint8)
        cv2.putText(sharp, "65A", (20, 29), cv2.FONT_HERSHEY_SIMPLEX,
                    .7, (20, 20, 20), 2)
        cv2.putText(sharp, "18114", (18, 51), cv2.FONT_HERSHEY_SIMPLEX,
                    .6, (20, 20, 20), 2)
        blurry = cv2.GaussianBlur(sharp, (15, 15), 4)
        self.assertGreater(crop_quality(sharp)[1], crop_quality(blurry)[1])
        bordered = blurry.copy()
        cv2.rectangle(bordered, (0, 0), (127, 63), (0, 0, 0), 3)
        self.assertAlmostEqual(crop_quality(bordered)[1], crop_quality(blurry)[1])


if __name__ == "__main__":
    unittest.main()
