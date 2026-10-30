"""Nine sequential GPU runs; preserve videos, exact commands and raw statistics."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import cv2
import torch
import ultralytics


ROOT = Path(__file__).resolve().parents[2]
# Keep the archived measurements immutable; new outputs stay local under data/.
OUT = ROOT / "data/test/diagnostics" / time.strftime("comparison_%Y%m%d_%H%M%S")
OUT.mkdir(parents=True, exist_ok=False)
RUNNER = ROOT / "scripts/realtime_lp_bytetrack.py"
OLD_RUNNER = ROOT / "scripts/realtime_lp_detect.py"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if not torch.cuda.is_available():
    raise RuntimeError("This comparison explicitly requires CUDA")

videos = []
for name in ("test", "test2", "test3"):
    path = ROOT / "data/test" / f"{name}.mp4"
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {path}")
    videos.append(dict(name=name, path=str(path), sha256=sha256(path),
                       frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), fps=cap.get(cv2.CAP_PROP_FPS),
                       width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    cap.release()

manifest = dict(
    gpu=torch.cuda.get_device_name(0), torch=torch.__version__, ultralytics=ultralytics.__version__,
    python=sys.version, weights_sha256=sha256(ROOT / "weights/best.pt"),
    tracker_config=(ROOT / "configs/bytetrack.yaml").read_text(encoding="utf-8"),
    parameters=dict(imgsz=640, conf=.5, iou=.45, device=0, cpu_threads=4,
                    display=False, write_video=True, pacing="source FPS", runs_per_case=1),
    videos=videos, cases=[],
)

for video in videos:
    # Rotate order to reduce systematic warm GPU/thermal ordering bias.
    order = {"test": ["yolo", "csrt", "bytetrack"],
             "test2": ["bytetrack", "yolo", "csrt"],
             "test3": ["csrt", "bytetrack", "yolo"]}[video["name"]]
    for mode in order:
        stem = f"{video['name']}_{mode}"
        stats_path, output = OUT / f"{stem}.json", OUT / f"{stem}.mp4"
        command = [sys.executable, str(OLD_RUNNER if mode == "csrt" else RUNNER),
                   "--source", video["path"], "--weights", str(ROOT / "weights/best.pt"),
                   "--imgsz", "640", "--conf", "0.5", "--iou", "0.45", "--device", "0",
                   "--cpu-threads", "4", "--no-display", "--output", str(output),
                   "--stats-json", str(stats_path)]
        if mode == "yolo":
            command += ["--mode", "detect"]
        elif mode == "bytetrack":
            command += ["--tracks-jsonl", str(OUT / f"{stem}_tracks.jsonl")]
        else:
            command += ["--detect-interval", "10", "--search-interval", "5"]
        print(f"START {video['name']} / {mode}", flush=True)
        started = time.perf_counter()
        run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
        (OUT / f"{stem}.log").write_text(run.stdout + "\n" + run.stderr, encoding="utf-8")
        if run.returncode:
            print(run.stdout, run.stderr, flush=True)
            raise RuntimeError(f"Case failed: {stem}")
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        if stats["frames"] != video["frames"]:
            raise AssertionError(f"Incomplete input: {stem}")
        cap = cv2.VideoCapture(str(output))
        decoded = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape[:2] != (video["height"], video["width"]):
                raise AssertionError(f"Wrong output dimensions: {stem}")
            decoded += 1
        cap.release()
        if decoded != video["frames"]:
            raise AssertionError(f"Incomplete output: {stem} ({decoded})")
        normalized = dict(
            video=video["name"], mode=mode, frames=stats["frames"], yolo_calls=stats["yolo_calls"],
            playback_fps=stats["frames"] / stats["playback_seconds"],
            active_loop_fps=stats.get("processing_fps", stats.get("active_loop_fps")),
            p50_ms=stats["loop_p50_ms"], p95_ms=stats["loop_p95_ms"], max_ms=stats["loop_max_ms"],
            visible_frames=stats["frames_with_visible_tracks"],
            unique_ids=stats.get("unique_ids"), peak_cuda_allocated_mb=stats.get("peak_cuda_allocated_mb"),
            worker=stats.get("worker"), command=command, output=str(output),
            total_process_seconds=time.perf_counter() - started,
        )
        manifest["cases"].append(normalized)
        (OUT / "comparison.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"DONE {stem}: {normalized['playback_fps']:.1f} FPS, p95={normalized['p95_ms']:.1f}ms, "
              f"YOLO={normalized['yolo_calls']}/{normalized['frames']}, "
              f"visible={normalized['visible_frames']}/{normalized['frames']}", flush=True)

print("ALL 9 CASES COMPLETE", flush=True)
