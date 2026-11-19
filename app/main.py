import asyncio
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
import io
import logging
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
from threading import Event
from time import monotonic, sleep
from urllib.parse import urlsplit
from uuid import uuid4

import cv2
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
from pydantic import BaseModel, StrictBool
from PIL import Image, JpegImagePlugin
from starlette.concurrency import run_in_threadpool

from app.pipeline import Models, Session, jpeg

STATIC = Path(__file__).parent / "static"
MAX_UPLOAD = 100 * 1024**2
MAX_FRAMES = 18_000
MAX_PIXELS = 1920 * 1080
TTL = 3600
log = logging.getLogger(__name__)


@dataclass
class Job:
    path: Path
    status: str = "processing"
    frames: int = 0
    total: int = 0
    plates: dict = field(default_factory=dict)
    error: str = ""
    finished: float | None = None
    preview: tuple[int, bytes] | None = None
    paused: Event = field(default_factory=Event)


def prune(app):
    for key, job in list(app.state.jobs.items()):
        if job.finished is not None and monotonic() - job.finished >= TTL:
            shutil.rmtree(job.path, ignore_errors=True)
            del app.state.jobs[key]


async def cleanup(app):
    while True:
        await asyncio.sleep(60)
        await run_in_threadpool(prune, app)


@asynccontextmanager
async def lifespan(app):
    app.state.models = await run_in_threadpool(Models)
    app.state.slot = asyncio.Lock()
    app.state.jobs = {}
    app.state.task = None
    with TemporaryDirectory(prefix="plate-web-") as directory:
        app.state.directory = Path(directory)
        cleaner = asyncio.create_task(cleanup(app))
        try:
            yield
        finally:
            cleaner.cancel()
            await asyncio.gather(cleaner, return_exceptions=True)
            for job in app.state.jobs.values():
                job.paused.clear()
            if app.state.task is not None:
                await app.state.task


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    return {"status": "ready"}


def same_origin(connection):
    origin = connection.headers.get("origin")
    return not origin or urlsplit(origin).netloc == connection.headers.get("host")


def process_video(models, job):
    source = job.path / "source"
    intermediate = job.path / "annotated.mp4"
    output = job.path / "result.mp4"
    capture = cv2.VideoCapture(str(source), cv2.CAP_FFMPEG)
    writer = None
    try:
        ok, frame = capture.read()
        if not ok:
            raise ValueError("Không đọc được video này.")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not 0 < fps <= 240:
            fps = 30
        job.total = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        if job.total > MAX_FRAMES:
            raise ValueError("Video vượt giới hạn 18.000 khung hình.")
        height, width = frame.shape[:2]
        if width * height > MAX_PIXELS:
            raise ValueError("Video vượt giới hạn 2.073.600 pixel mỗi khung hình.")
        padded = (width + width % 2, height + height % 2)
        writer = cv2.VideoWriter(str(intermediate), cv2.VideoWriter_fourcc(*"mp4v"), fps, padded)
        if not writer.isOpened():
            raise RuntimeError("Không thể tạo video kết quả.")
        session = Session(models, uploaded_video=True, source_fps=fps)
        last_preview = -float('inf')
        while ok:
            while job.paused.is_set():
                sleep(.05)
            if job.frames >= MAX_FRAMES or frame.shape[:2] != (height, width):
                raise ValueError("Video vượt giới hạn khung hình hoặc thay đổi kích thước giữa chừng.")
            frame, events = session.step(frame, job.frames / fps)
            now = monotonic()
            if now - last_preview >= 1 / 12:
                encoded, image = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if encoded:
                    job.preview = job.frames, image.tobytes()
                    last_preview = now
            if padded != (width, height):
                frame = cv2.copyMakeBorder(frame, 0, height % 2, 0, width % 2, cv2.BORDER_CONSTANT)
            writer.write(frame)
            job.plates.update((event["id"], event) for event in events)
            job.frames += 1
            ok, frame = capture.read()
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
         "-i", str(intermediate), "-c:v", "libx264", "-preset", "fast",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)],
        check=True, timeout=900,
    )


async def finish_job(job):
    try:
        await run_in_threadpool(process_video, app.state.models, job)
        job.status = "ready"
    except ValueError as exc:
        job.error = str(exc)
        job.status = "failed"
    except Exception:
        log.exception("Video processing failed")
        job.error = "Xử lý video thất bại."
        job.status = "failed"
    finally:
        job.paused.clear()
        for name in ("source", "annotated.mp4"):
            (job.path / name).unlink(missing_ok=True)
        job.finished = monotonic()
        app.state.slot.release()


@app.post("/api/videos", status_code=202)
async def submit(request: Request):
    if not same_origin(request):
        raise HTTPException(403)
    if request.headers.get("content-type") != "application/octet-stream":
        raise HTTPException(415, "Gửi video dưới dạng application/octet-stream.")
    if app.state.slot.locked():
        raise HTTPException(409, "Đang xử lý một phiên khác. Vui lòng thử lại.")
    await app.state.slot.acquire()
    key = uuid4().hex
    job = Job(app.state.directory / key)
    scheduled = False
    try:
        job.path.mkdir()
        size = 0
        with (job.path / "source").open("wb") as stream:
            async with asyncio.timeout(120):
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > MAX_UPLOAD:
                        raise HTTPException(413, "Video vượt quá 100 MB.")
                    stream.write(chunk)
        if not size:
            raise HTTPException(400, "File video trống.")
        app.state.jobs[key] = job
        app.state.task = asyncio.create_task(finish_job(job))
        scheduled = True
        return {"id": key}
    except TimeoutError as exc:
        raise HTTPException(408, "Tải video lên quá thời gian.") from exc
    finally:
        if not scheduled:
            shutil.rmtree(job.path, ignore_errors=True)
            app.state.slot.release()


def get_job(key):
    job = app.state.jobs.get(key)
    if job is None:
        raise HTTPException(404, "Phiên không tồn tại hoặc đã hết hạn.")
    return job


@app.get("/api/videos/{key}")
def status(key: str):
    job = get_job(key)
    return dict(status=job.status, frames=job.frames, total=job.total,
                plates=list(job.plates.copy().values()), error=job.error,
                paused=job.paused.is_set())


class Playback(BaseModel):
    paused: StrictBool


@app.patch("/api/videos/{key}/playback")
async def playback(key: str, body: Playback, request: Request):
    if not same_origin(request):
        raise HTTPException(403)
    job = get_job(key)
    if job.status == "processing":
        if body.paused:
            job.paused.set()
        else:
            job.paused.clear()
    return {"paused": job.paused.is_set() and job.status == "processing"}


@app.get("/api/videos/{key}/preview")
async def preview(key: str):
    job = get_job(key)

    async def frames():
        last_frame = -1
        while True:
            latest = job.preview
            if latest is not None and latest[0] != last_frame:
                last_frame, image = latest
                yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                       + str(len(image)).encode("ascii") + b"\r\n\r\n" + image + b"\r\n")
            if job.status != "processing":
                yield b"--frame--\r\n"
                return
            await asyncio.sleep(1 / 15)

    return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


@app.get("/api/videos/{key}/result")
def result(key: str):
    job = get_job(key)
    if job.status != "ready":
        raise HTTPException(409, "Video kết quả chưa sẵn sàng.")
    return FileResponse(job.path / "result.mp4", media_type="video/mp4")


def live_frame(session, data, seconds):
    try:
        with JpegImagePlugin.JpegImageFile(io.BytesIO(data)) as picture:
            if picture.width * picture.height > MAX_PIXELS:
                raise ValueError("Khung hình camera quá lớn.")
            picture.load()
            frame = cv2.cvtColor(np.asarray(picture.convert("RGB")), cv2.COLOR_RGB2BGR)
    except (OSError, SyntaxError, Image.DecompressionBombError) as exc:
        raise ValueError("Khung hình JPEG không hợp lệ.") from exc
    annotated, events = session.step(frame, seconds)
    return {"frame": jpeg(annotated), "plates": events}


@app.websocket("/api/camera")
async def camera(socket: WebSocket):
    if not same_origin(socket):
        await socket.close(code=1008)
        return
    await socket.accept()
    if app.state.slot.locked():
        await socket.send_json({"error": "Đang xử lý một phiên khác. Vui lòng thử lại."})
        await socket.close(code=1013)
        return
    async with app.state.slot:
        start = monotonic()
        try:
            session = await run_in_threadpool(Session, app.state.models)
            while monotonic() - start < TTL:
                async with asyncio.timeout(30):
                    message = await socket.receive()
                if message["type"] == "websocket.disconnect":
                    return
                data = message.get("bytes")
                if not data or len(data) > 2 * 1024**2:
                    raise ValueError("Cần gửi khung hình JPEG tối đa 2 MB.")
                seconds = monotonic() - start
                response = await run_in_threadpool(live_frame, session, data, seconds)
                await socket.send_json(response)
            await socket.send_json({"error": "Phiên camera đã hết thời gian."})
            await socket.close()
        except WebSocketDisconnect:
            return
        except (ValueError, TimeoutError) as exc:
            with suppress(WebSocketDisconnect):
                await socket.send_json({"error": str(exc) or "Camera không gửi hình trong 30 giây."})
                await socket.close(code=1008)
        except Exception:
            log.exception("Camera processing failed")
            with suppress(WebSocketDisconnect):
                await socket.send_json({"error": "Xử lý camera thất bại."})
                await socket.close(code=1011)
