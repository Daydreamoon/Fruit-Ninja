"""AirFruit M0 environment probes.

Usage (with .venv activated):
    python vision/probes.py            # full probe: versions, GPU/CPU bench, camera fps
    python vision/probes.py --camera   # camera fps only
    python vision/probes.py --gpu      # model download + GPU/CPU inference bench only

Exit codes: 0 = all good; 1 = GPU delegate unavailable (expected on Windows,
CPU fallback is fine); 2 = CPU landmarker broken (fatal).
"""
from __future__ import annotations

import socket
import sys
import time
import urllib.request
from pathlib import Path

socket.setdefaulttimeout(30)

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "pose_landmarker_lite.task"
MODEL_URLS = [
    # official Tasks API path first; reference-project path as fallback
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
]


def ensure_model() -> Path:
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 1_000_000:
        print(f"[model] cached: {MODEL_PATH}")
        return MODEL_PATH
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    last: Exception | None = None
    for url in MODEL_URLS:
        try:
            print(f"[model] downloading: {url}")
            urllib.request.urlretrieve(url, MODEL_PATH)
            print(f"[model] saved {MODEL_PATH.stat().st_size / 1e6:.1f} MB -> {MODEL_PATH}")
            return MODEL_PATH
        except Exception as exc:  # noqa: BLE001
            last = exc
            print(f"[model] FAILED: {exc}")
    raise RuntimeError(f"all model URLs failed, last error: {last}")


def bench_landmarker(delegate: str, model_path: Path) -> tuple[bool, str]:
    """Create a PoseLandmarker with the given delegate and time dummy inferences."""
    import mediapipe as mp
    import numpy as np
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    try:
        delegate_enum = (
            mp_python.BaseOptions.Delegate.GPU
            if delegate.upper() == "GPU"
            else mp_python.BaseOptions.Delegate.CPU
        )
        # NB: model_asset_path with non-ASCII (Chinese) dirs fails inside
        # mediapipe's C++ file layer on Windows; pass the bytes instead.
        base = mp_python.BaseOptions(
            model_asset_buffer=model_path.read_bytes(),
            delegate=delegate_enum,
        )
        options = vision.PoseLandmarkerOptions(
            base_options=base,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
        )
        landmarker = vision.PoseLandmarker.create_from_options(options)
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        ts = 0

        def run(count: int) -> list[float]:
            nonlocal ts
            times: list[float] = []
            for _ in range(count):
                ts += 33
                t0 = time.perf_counter()
                landmarker.detect_for_video(image, ts)
                times.append(time.perf_counter() - t0)
            return times

        run(10)  # warmup: graph init (+ delegate bring-up if GPU)
        measured = run(30)
        landmarker.close()
        median_ms = 1000.0 * sorted(measured)[len(measured) // 2]
        return True, f"median {median_ms:.1f} ms/frame (~{1000.0 / median_ms:.0f} Hz possible)"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def camera_probe(index: int = 0, seconds: float = 4.0) -> None:
    """Measure what each backend really delivers at 1280x720@30.

    Both backends are probed because they differ wildly on some machines
    (reference box: DSHOW ~10 fps, MSMF 30 fps). Brightness is reported too:
    privacy tools can hand out an all-black fake stream instead of failing,
    which looks "open" but is unusable.
    """
    import cv2

    def measure(cap, label: str, secs: float) -> float:
        got, first, last = 0, None, None
        brightness = -1.0
        deadline = time.perf_counter() + secs
        while time.perf_counter() < deadline:
            ok, frame = cap.read()
            if ok:
                got += 1
                last = time.perf_counter()
                if first is None:
                    first = time.perf_counter()
                if frame is not None and got % 5 == 0:
                    brightness = float(frame.mean())
        elapsed = time.perf_counter() - (first or deadline)
        fps = got / elapsed if got and elapsed > 0.2 else 0.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        reported = cap.get(cv2.CAP_PROP_FPS)
        verdict = "OK"
        if fps < 20:
            verdict = "SLOW"
        if brightness == 0.0:
            verdict = "BLACK-fake-stream"
        print(
            f"[camera] {label:<20} measured {fps:6.1f} fps @ {w}x{h} "
            f"(driver reports {reported:.0f}) brightness {brightness:5.1f}  [{verdict}]"
        )
        return fps

    for backend, name in ((cv2.CAP_MSMF, "MSMF"), (cv2.CAP_DSHOW, "DSHOW")):
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            print(f"[camera] {name:<4} cannot open camera index {index}")
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        measure(cap, f"{name} 1280x720@30", seconds)
        cap.release()
    print("[camera] done")


def main() -> int:
    print("===== AirFruit M0 environment probe =====")
    import cv2
    import mediapipe
    import numpy
    import PySide6
    from PySide6 import QtCore

    print(f"python   : {sys.version.split()[0]} ({sys.executable})")
    print(f"PySide6  : {PySide6.__version__} (Qt {QtCore.qVersion()})")
    print(f"OpenCV   : {cv2.__version__}")
    print(f"MediaPipe: {mediapipe.__version__}")
    print(f"NumPy    : {numpy.__version__}")

    args = set(sys.argv[1:])
    run_all = not args
    gpu_unavailable = False

    if run_all or "--gpu" in args:
        model = ensure_model()
        ok, msg = bench_landmarker("GPU", model)
        print(f"[delegate] GPU : {'OK  ' if ok else 'FAIL'} -> {msg}")
        if not ok:
            gpu_unavailable = True
        ok, msg = bench_landmarker("CPU", model)
        print(f"[delegate] CPU : {'OK  ' if ok else 'FAIL'} -> {msg}")
        if not ok:
            return 2

    if run_all or "--camera" in args:
        camera_probe()

    print("===== probe finished =====")
    return 1 if gpu_unavailable else 0


if __name__ == "__main__":
    raise SystemExit(main())
