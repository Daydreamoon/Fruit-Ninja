"""Vision thread: camera (or synthetic) -> PoseLandmarker -> blade points.

Runs a tight loop in its own QThread: read frame -> mirror -> detect ->
extract blade points -> publish to SharedVisionState. The render thread
never touches OpenCV or MediaPipe.

Blade mapping follows the reference project: midpoint of each hand's
pinky (17/18) and index (19/20) pose landmarks; both hands can be blades
(anti-mistouch decision from the plan). Coordinates stay normalized;
the frame is mirrored here, so x is already in the player's view.
"""
from __future__ import annotations

import sys
import threading
import time

import cv2
import numpy as np
from PySide6.QtCore import QThread

from .shared import PREVIEW_W, PREVIEW_H, BladePoint, SharedVisionState

BLADE_PAIRS = ((17, 19), (18, 20))  # (pinky, index) per arm
VISIBILITY_MIN = 0.35  # 0.5 was too strict: hands in motion kept dropping out
CAMERA_W, CAMERA_H = 1280, 720
FIRST_FRAME_TIMEOUT_S = 1.5  # give a backend this long to deliver its first frame


class VisionThread(QThread):
    def __init__(
        self,
        shared: SharedVisionState,
        camera_index: int = 0,
        synthetic: bool = False,
    ) -> None:
        super().__init__()
        self.shared = shared
        self.camera_index = camera_index
        self.synthetic = synthetic
        self._stop = threading.Event()

    # ---- public ----

    def stop(self) -> None:
        self._stop.set()
        self.wait(3000)

    # ---- setup helpers ----

    def _open_camera(self):
        """Open 1280x720@30, MSMF first then DSHOW.

        Order matters: on the reference machine DSHOW tops out at ~10 fps while
        MSMF reaches 30 fps on the same camera (docs/M0 report). A backend that
        opens but never delivers a frame inside FIRST_FRAME_TIMEOUT_S is dropped,
        so a broken MSMF stack still falls back to DSHOW.
        """
        for backend in (cv2.CAP_MSMF, cv2.CAP_DSHOW):
            t_backend = time.monotonic()
            cap = cv2.VideoCapture(self.camera_index, backend)
            if not cap.isOpened():
                cap.release()
                print(
                    f"[vision] backend {backend} unavailable ({time.monotonic() - t_backend:.1f}s)",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_H)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            deadline = time.monotonic() + FIRST_FRAME_TIMEOUT_S
            while time.monotonic() < deadline:
                ok, _ = cap.read()
                if ok:
                    print(
                        f"[vision] backend {backend} opened in {time.monotonic() - t_backend:.1f}s",
                        file=sys.stderr,
                        flush=True,
                    )
                    return cap
            cap.release()
            print(
                f"[vision] backend {backend} gave no frame in {FIRST_FRAME_TIMEOUT_S}s",
                file=sys.stderr,
                flush=True,
            )
        return None

    def _make_landmarker(self):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        from .probes import ensure_model

        model = ensure_model()
        base = mp_python.BaseOptions(
            # bytes, never a path: Chinese dirs break mediapipe's C++ layer (M0 report)
            model_asset_buffer=model.read_bytes(),
            delegate=mp_python.BaseOptions.Delegate.CPU,
        )
        options = vision.PoseLandmarkerOptions(
            base_options=base,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
        )
        return vision.PoseLandmarker.create_from_options(options)

    # ---- main loop ----

    def run(self) -> None:  # QThread entry
        import sys

        t_setup = time.monotonic()
        try:
            self.shared.publish_status("loading_model")
            landmarker = self._make_landmarker()
        except Exception as exc:  # noqa: BLE001
            self.shared.publish_status(f"model_error: {type(exc).__name__}")
            return
        print(f"[vision] model ready in {time.monotonic() - t_setup:.1f}s", file=sys.stderr, flush=True)

        cap = None
        if not self.synthetic:
            self.shared.publish_status("opening_camera")
            t_cam = time.monotonic()
            cap = self._open_camera()
            print(f"[vision] camera open in {time.monotonic() - t_cam:.1f}s", file=sys.stderr, flush=True)
            if cap is None:
                # camera-level failure; user-visible status, render loop unaffected
                self.shared.publish_status("camera_open_failed")
                return

        import mediapipe as mp

        self.shared.publish_status("starting")

        t_start = time.perf_counter()
        ts = 0
        frame_i = 0
        synth = np.zeros((CAMERA_H, CAMERA_W, 3), dtype=np.uint8)
        infer_ms = 0.0

        while not self._stop.is_set():
            t0 = time.perf_counter()
            if self.synthetic:
                # moving dot: frames differ so detect_for_video sees real motion;
                # no pose will be found — synthetic mode only validates loop +
                # inference cost + shared-state plumbing (see M0 report).
                synth[:] = 30
                x = (frame_i * 9) % CAMERA_W
                cv2.circle(synth, (x, CAMERA_H // 2), 80, (200, 120, 60), -1)
                frame_bgr = synth
            else:
                ok, frame_bgr = cap.read()
                if not ok:
                    self.shared.publish_status("camera_read_failed")
                    break
                frame_bgr = cv2.flip(frame_bgr, 1)  # mirror: player view

            try:
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                ts += 33
                t1 = time.perf_counter()
                result = landmarker.detect_for_video(image, ts)
                infer_ms = (time.perf_counter() - t1) * 1000.0

                blades: list[BladePoint] = []
                poses = result.pose_landmarks
                if poses:
                    lms = poses[0]
                    for pinky_i, index_i in BLADE_PAIRS:
                        p, q = lms[pinky_i], lms[index_i]
                        if p.visibility > VISIBILITY_MIN and q.visibility > VISIBILITY_MIN:
                            # raw, mirrored camera coordinates (0..1). Turning
                            # them into screen positions is the render thread's
                            # job: only it knows where this player's hands sit
                            # (see the auto-calibration in game/window.py).
                            blades.append(
                                BladePoint(
                                    x=(p.x + q.x) / 2,
                                    y=(p.y + q.y) / 2,
                                    visibility=min(p.visibility, q.visibility),
                                )
                            )
                preview = cv2.resize(frame_bgr, (PREVIEW_W, PREVIEW_H)).tobytes()
                self.shared.publish(blades, preview, infer_ms)
            except Exception as exc:  # noqa: BLE001
                # one bad frame must not kill vision during play: log, cool down,
                # keep serving. publish_status carries the exception type so the
                # HUD shows it; a later successful publish restores normal data.
                import traceback

                traceback.print_exc()
                self.shared.publish_status(f"landmark_error: {type(exc).__name__}")
                self._stop.wait(0.2)
                frame_i += 1
                continue

            if self.synthetic:
                # model the real source: a camera self-paces at ~30 fps, and the
                # plan says sampling follows the source (not faster). Pure sleep:
                # Windows granularity ~15.6ms is fine at a 33ms target period.
                target = 1.0 / 30.0
                rem = target - (time.perf_counter() - t0)
                if rem > 0:
                    time.sleep(rem)
            frame_i += 1

        print(
            f"[vision] loop exit t={time.perf_counter() - t_start:6.1f}s"
            f" frames={frame_i} stop_flag={self._stop.is_set()}",
            file=sys.stderr,
            flush=True,
        )
        if cap is not None:
            cap.release()
        try:
            landmarker.close()
        except Exception:  # noqa: BLE001
            pass
        # never mask an error with "stopped": the HUD should keep showing the
        # last abnormal state after a silent exit (observed in M2 selftest).
        if not self.shared.status.startswith(("landmark_error", "camera_read_failed")):
            self.shared.publish_status("stopped")
