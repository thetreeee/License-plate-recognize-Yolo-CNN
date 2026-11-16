from types import SimpleNamespace
import unittest

from scripts.temporal_ocr import TemporalOCR


def reading(text, ok=True, error=""):
    return SimpleNamespace(text=text, split_ok=ok, error=error, layout="two_rows")


class TemporalOCRTests(unittest.TestCase):
    def test_new_track_waits_five_consecutive_frames_and_respects_interval(self):
        scheduler = TemporalOCR()
        for frame in range(4):
            scheduler.observe([1], frame)
            self.assertFalse(scheduler.due(1, frame))
        self.assertTrue(scheduler.due(1, 3, every_frame=True))
        scheduler.observe([], 4)
        for frame in range(5, 9):
            scheduler.observe([1], frame)
            self.assertFalse(scheduler.due(1, frame))
        scheduler.observe([1], 9)
        self.assertTrue(scheduler.due(1, 9))
        scheduler.attempted(1, 9)
        self.assertFalse(scheduler.due(1, 18))
        self.assertTrue(scheduler.due(1, 19))

    def test_consensus_requires_three_agreements_and_remains_refreshable(self):
        scheduler = TemporalOCR()
        scheduler.observe([1], 0)
        for i in range(2):
            scheduler.accept(1, reading("59M120424"))
        self.assertFalse(scheduler.states[1].consensus)
        scheduler.accept(1, reading("59M120424"))
        scheduler.attempted(1, 20)
        self.assertTrue(scheduler.states[1].consensus)
        self.assertFalse(scheduler.due(1, 30))
        self.assertTrue(scheduler.due(1, 50))
        for i in range(5):
            scheduler.accept(1, reading("92D188904"))
        self.assertEqual(scheduler.states[1].text, "92D188904")

    def test_failed_reads_invalidate_old_consensus(self):
        scheduler = TemporalOCR()
        scheduler.observe([1], 0)
        for _ in range(3):
            scheduler.accept(1, reading("59M120424"))
        for _ in range(3):
            scheduler.accept(1, reading("bad", ok=False))
        self.assertFalse(scheduler.states[1].consensus)

    def test_new_identity_never_inherits_another_identity_text(self):
        scheduler = TemporalOCR(retention=60)
        scheduler.observe([1], 0)
        scheduler.accept(1, reading("59M120424"))
        scheduler.observe([2], 1)
        self.assertEqual(scheduler.states[2].text, "")
        scheduler.observe([2], 61)
        self.assertNotIn(1, scheduler.states)

    def test_ties_prefer_newest_reading(self):
        scheduler = TemporalOCR()
        scheduler.observe([1], 0)
        scheduler.accept(1, reading("A"))
        scheduler.accept(1, reading("B"))
        self.assertEqual(scheduler.states[1].text, "B")
        self.assertFalse(scheduler.states[1].consensus)


if __name__ == "__main__":
    unittest.main()
