"""Concurrency and delayed-frame behavior without YOLO or a camera."""

from threading import Event, get_ident
import time
from types import SimpleNamespace
import unittest

from scripts.async_detector import AsyncDetector, DetectionSchedule, FrameHistory, StaleResult
from scripts.plate_tracking import PlateTracker


class MovingTracker:
    def init(self, frame, box):
        self.initial = box
        self.origin = frame.shift

    def update(self, frame):
        x, y, w, h = self.initial
        return True, (x + frame.shift - self.origin, y, w, h)


def image(shift=0):
    return SimpleNamespace(shape=(240, 400, 3), shift=shift)


class AsyncTests(unittest.TestCase):
    def setUp(self):
        self.workers = []
        self.release_events = []

    def tearDown(self):
        for event in self.release_events:
            event.set()
        for worker in self.workers:
            self.assertTrue(worker.close(timeout=2))

    def worker(self, predict, history, **kwargs):
        worker = AsyncDetector(lambda: predict, MovingTracker, history,
                               max_age_seconds=5, **kwargs)
        self.workers.append(worker)
        worker.start()
        return worker

    def wait_result(self, worker):
        deadline = time.perf_counter() + 2
        while time.perf_counter() < deadline:
            result = worker.poll()
            if result is not None:
                return result
            Event().wait(.005)
        self.fail("Worker did not produce a result")

    def blocked_predictor(self):
        started, release = Event(), Event()
        self.release_events.append(release)

        def predict(frame):
            started.set()
            if not release.wait(2):
                raise RuntimeError("Test did not release inference")
            return [((10, 20, 40, 20), .9)]

        return predict, started, release

    def test_history_is_bounded_and_rejects_missing_frames(self):
        history = FrameHistory(3)
        for index in range(4):
            history.append(index, image())
        with self.assertRaises(StaleResult):
            history.span(0)
        self.assertEqual([f.index for f in history.span(1, 3)], [1, 2, 3])

    def test_inference_does_not_block_poll_and_boxes_are_advanced_before_handoff(self):
        history = FrameHistory()
        source = history.append(0, image())
        predict, started, release = self.blocked_predictor()
        worker = self.worker(predict, history)
        worker.submit(source)
        self.assertTrue(started.wait(1))
        self.assertIsNone(worker.poll())  # UI can keep reading frames.
        history.append(1, image(5))
        history.append(2, image(10))
        release.set()
        result = self.wait_result(worker)
        self.assertEqual(result.aligned_frame, 2)
        self.assertEqual(result.tracks[0].box, (20, 20, 40, 20))
        transferred_tracker = result.tracks[0].tracker
        current = history.append(3, image(15))
        manager = PlateTracker(MovingTracker)
        self.assertTrue(worker.apply(result, manager, current))
        self.assertEqual(manager.tracks[1].box, (25, 20, 40, 20))
        self.assertIs(manager.tracks[1].tracker, transferred_tracker)
        self.assertEqual(manager.tracks[1].last_detection, 0)

    def test_pending_requests_keep_latest_frame_only(self):
        history = FrameHistory()
        source = history.append(0, image())
        started, release = Event(), Event()
        self.release_events.append(release)
        observed = []

        def predict(frame):
            observed.append(frame.shift)
            if frame.shift == 0:
                started.set()
                release.wait(2)
            return []

        worker = self.worker(predict, history)
        worker.submit(source)
        self.assertTrue(started.wait(1))
        worker.submit(history.append(1, image(1)))
        worker.submit(history.append(2, image(2)))
        release.set()
        deadline = time.perf_counter() + 2
        while worker.stats()["calls"] < 2 and time.perf_counter() < deadline:
            Event().wait(.005)
        self.assertEqual(observed, [0, 2])
        self.assertEqual(worker.stats()["replaced"], 1)

    def test_foreground_discards_large_tail_without_changing_tracks(self):
        history = FrameHistory()
        source = history.append(0, image())
        worker = self.worker(lambda frame: [((10, 20, 40, 20), .9)], history)
        worker.submit(source)
        result = self.wait_result(worker)
        for index in range(1, 5):
            current = history.append(index, image(index))
        manager = PlateTracker(MovingTracker)
        self.assertFalse(worker.apply(result, manager, current))
        self.assertEqual(manager.tracks, {})
        self.assertEqual(worker.stats()["stale"], 1)

    def test_transport_branch_initializes_tracker_on_destination_frame(self):
        history = FrameHistory()
        source = history.append(0, image())
        history.append(1, image(5))
        history.append(2, image(10))

        def transport(frames, detections, appearance_factory):
            self.assertEqual(frames[-1].index, 2)
            return [((20, 20, 40, 20), .9, None)]

        worker = self.worker(lambda frame: [((10, 20, 40, 20), .9)], history,
                             propagate=transport)
        worker.submit(source)
        result = self.wait_result(worker)
        self.assertEqual(result.tracks[0].tracker.origin, 10)
        current = history.append(3, image(15))
        manager = PlateTracker(MovingTracker)
        self.assertTrue(worker.apply(result, manager, current))
        self.assertEqual(manager.tracks[1].box, (25, 20, 40, 20))
        self.assertEqual(manager.tracks[1].last_detection, 0)

    def test_expired_history_during_inference_is_discarded(self):
        history = FrameHistory(3)
        source = history.append(0, image())
        predict, started, release = self.blocked_predictor()
        worker = self.worker(predict, history)
        worker.submit(source)
        self.assertTrue(started.wait(1))
        for index in range(1, 5):
            history.append(index, image(index))
        release.set()
        deadline = time.perf_counter() + 2
        while worker.stats()["stale"] == 0 and time.perf_counter() < deadline:
            Event().wait(.005)
        self.assertIsNone(worker.poll())
        self.assertEqual(worker.stats()["stale"], 1)

    def test_worker_error_reaches_foreground(self):
        history = FrameHistory()
        source = history.append(0, image())

        def predict(frame):
            raise ValueError("inference failure")

        worker = self.worker(predict, history)
        worker.submit(source)
        worker._thread.join(1)
        with self.assertRaisesRegex(RuntimeError, "background worker failed"):
            worker.poll()

    def test_failed_transport_is_not_applied_as_a_negative_detection(self):
        history = FrameHistory()
        source = history.append(0, image())
        worker = self.worker(lambda frame: [((10, 20, 40, 20), .9)], history,
                             propagate=lambda frames, detections, appearance: [])
        worker.submit(source)
        deadline = time.perf_counter() + 2
        while worker.stats()["stale"] == 0 and time.perf_counter() < deadline:
            Event().wait(.005)
        self.assertEqual(worker.stats()["stale"], 1)
        self.assertIsNone(worker.poll())

    def test_initialization_runs_on_worker_and_errors_unblock_start(self):
        owner = []

        def factory():
            owner.append(get_ident())
            raise ValueError("bad weights")

        worker = AsyncDetector(factory, MovingTracker, FrameHistory())
        self.workers.append(worker)
        with self.assertRaises(RuntimeError):
            worker.start()
        self.assertNotEqual(owner[0], get_ident())

    def test_shutdown_discards_pending_work(self):
        history = FrameHistory()
        source = history.append(0, image())
        predict, started, release = self.blocked_predictor()
        worker = self.worker(predict, history)
        worker.submit(source)
        self.assertTrue(started.wait(1))
        worker.submit(history.append(1, image(1)))
        self.assertFalse(worker.close(timeout=.01))
        release.set()
        self.assertTrue(worker.close(timeout=1))
        self.assertEqual(worker.stats()["calls"], 1)
        self.assertFalse(worker.submit(source))

    def test_schedule_searches_faster_without_flooding_worker(self):
        schedule = DetectionSchedule(100, 5)
        self.assertTrue(schedule.due(0, False))
        schedule.requested(0)
        self.assertFalse(schedule.due(1, False, True))
        self.assertTrue(schedule.due(5, False))
        self.assertFalse(schedule.due(5, True))
        self.assertTrue(schedule.due(100, True))


if __name__ == "__main__":
    unittest.main()
