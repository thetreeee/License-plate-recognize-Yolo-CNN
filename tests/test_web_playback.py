import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Thread
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from fastapi import HTTPException
from pydantic import ValidationError

from app.main import Job, Playback, app, playback, process_video
from app.pipeline import Session


class PlaybackTests(unittest.TestCase):
    def test_pause_api_requires_boolean_and_same_origin(self):
        job = Job(Path("unused"))
        request = SimpleNamespace(headers={"origin": "http://localhost", "host": "localhost"})
        with patch.object(app.state, "jobs", {"test": job}, create=True):
            self.assertTrue(asyncio.run(playback("test", Playback(paused=True), request))["paused"])
            self.assertTrue(job.paused.is_set())
            self.assertFalse(asyncio.run(playback("test", Playback(paused=False), request))["paused"])
            request.headers["origin"] = "http://another-site"
            with self.assertRaises(HTTPException) as error:
                asyncio.run(playback("test", Playback(paused=True), request))
            self.assertEqual(error.exception.status_code, 403)
        with self.assertRaises(ValidationError):
            Playback(paused="false")

    def test_processing_stops_between_frames_and_resumes(self):
        frame = np.zeros((96, 160, 3), np.uint8)
        reached = Event()
        errors = []
        with TemporaryDirectory() as directory:
            job = Job(Path(directory))

            def step(image, seconds):
                if seconds == 0:
                    job.paused.set()
                    reached.set()
                return image, []

            def run():
                try:
                    process_video(None, job)
                except Exception as error:
                    errors.append(error)

            with patch("app.main.cv2.VideoCapture") as capture, \
                    patch("app.main.cv2.VideoWriter") as writer, \
                    patch("app.main.Session") as session, \
                    patch("app.main.subprocess.run"):
                capture.return_value.read.side_effect = [(True, frame.copy()), (True, frame.copy()), (False, None)]
                capture.return_value.get.return_value = 25
                session.return_value.step.side_effect = step
                worker = Thread(target=run, daemon=True)
                worker.start()
                try:
                    self.assertTrue(reached.wait(timeout=3))
                    time.sleep(.2)
                    self.assertEqual(session.return_value.step.call_count, 1)
                finally:
                    job.paused.clear()
                    worker.join(timeout=3)
                self.assertFalse(worker.is_alive())
                self.assertFalse(errors, errors)
                self.assertEqual(job.frames, 2)
                self.assertEqual(writer.return_value.write.call_count, 2)

    def test_consensus_event_points_to_current_track_and_frame(self):
        args = SimpleNamespace(high_imgsz=1280, high_interval=30, roi_interval=5, max_rois=2,
                               relink=False, search_roi=[], ocr_interval=10, stable_interval=30,
                               ocr_conf=.6, min_blur=20, ocr_batch=4)
        reading = SimpleNamespace(text="65A12345", layout="two_rows", error="", split_ok=True)
        reader = SimpleNamespace(config=SimpleNamespace(margin=0),
                                 read_many=lambda images: ([reading for _ in images], {}))
        models = SimpleNamespace(args=args, device=SimpleNamespace(type="cuda"),
                                 options={"imgsz": 640}, reader=reader, config={}, detector=None)
        track = dict(id=1, confidence=.9, xyxy=[10, 10, 90, 40])
        with patch("app.pipeline.AdaptiveDetector"), \
                patch("app.pipeline.observations", return_value=[track]), \
                patch("app.pipeline.crop_quality", return_value=(100., 100.)):
            session = Session(models, source_fps=25)
            session.detector.step.return_value = (None, [], {"matched_boxes": {}})
            events = []
            for index in range(25):
                _, found = session.step(np.full((100, 120, 3), 127, np.uint8), index / 25)
                events.extend(found)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["bbox"], [10 / 120, .1, 90 / 120, .4])
        self.assertAlmostEqual(events[0]["seconds"], 24 / 25)
        self.assertAlmostEqual(events[0]["seek_seconds"], 24.5 / 25)


if __name__ == "__main__":
    unittest.main()
