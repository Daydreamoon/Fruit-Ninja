"""Vision thread: camera (or synthetic) -> PoseLandmarker -> blade points.

Runs a tight loop in its own QThread: read frame -> mirror -> detect ->
extract blade points -> publish to SharedVisionState. The render thread
never touches OpenCV or MediaPipe.

The blade is the PALM CENTRE of each hand: the mean of that arm's wrist (15/16),
pinky MCP (17/18) and index MCP (19/20) pose landmarks. The earlier midpoint of
index+pinky sat between two fingers, which read as "the cursor is not on my
hand" during play. Coordinates stay normalized; the frame is mirrored here, so
x is already in the player's view.

A hand only counts as a blade while it is being used (see _extract_hands):
raised rather than hanging below its elbow, and moving at least occasionally.
That is what makes "one hand playing, the other one resting on the desk" show
a single blade instead of two.
"""
from __future__ import annotations

import math
import os
import sys
import threading
import time

import cv2
import numpy as np
from PySide6.QtCore import QThread

from .shared import PREVIEW_W, PREVIEW_H, BladePoint, SharedVisionState

# (side, wrist, pinky MCP, index MCP) -- pose landmark indices, left/right in
# the PERSON's own frame (MediaPipe's convention), which keeps the two hands
# distinguishable when only one of them is in play.
PALM_TRIPLES = (
    ("left", 15, 17, 19),
    ("right", 16, 18, 20),
)
ELBOWS = {"left": 13, "right": 14}
VISIBILITY_MIN = 0.50  # to start tracking a hand
VISIBILITY_KEEP = 0.35  # to keep tracking one already tracked (hysteresis)
HAND_HOLD_S = 0.35  # a hand that blinks out keeps its last point this long
RAISE_ACQUIRE = 0.02  # to start using a hand: wrist at least this far above the elbow
RAISE_KEEP = 0.10  # to keep using it: the wrist may sag a little below
IDLE_S = 2.0  # a hand not moving for this long counts as put down...
IDLE_SPEED = 0.004  # ...where "moving" means this much of the frame per sample
SPEED_ALPHA = 0.12  # smoothing for the per-hand speed estimate
SMOOTH_ALPHA = 0.55  # palm-centre EMA: pose jitter would otherwise wobble the blade
MAX_JUMP = 0.22  # per-sample displacement that a hand can only make while swiping
FAST_DIR_COS = 0.35  # ...and only if it keeps going the same way as the sample before
TELEPORT_DROP_S = 0.6  # a hand that keeps teleporting for this long is not a hand
TELEPORT_BLOCK_S = 1.5  # ...and is not re-acquired until this long after the drop
_DEBUG_HANDS = os.environ.get("AIRFRUIT_DEBUG_HANDS", "").lower() in ("1", "true", "yes", "on")
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
        # per-side tracking state, keyed "left"/"right":
        # {x, y, seen, anchor_x, anchor_y, anchor_t}
        self._hands: dict[str, dict] = {}
        # side -> time before which it may not be re-acquired (it was teleporting)
        self._blocked: dict[str, float] = {}
        self._noted: dict[str, str] = {}  # debug: last printed state per side

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

    # ---- blade extraction ----

    def _extract_hands(self, lms, now: float) -> list[BladePoint]:
        """Turn one pose result into blade points: palm centre per hand in play.

        Rules, in order:
        * visibility (min of wrist/MCPs) above the acquire threshold, or above
          the lower "keep" threshold for a hand already tracked -- hysteresis,
          because a hand in motion repeatedly dips below a single cutoff;
        * the hand must be raised: the wrist may not sit much below its elbow,
          which is what distinguishes a playing hand from an arm resting on the
          desk / hanging beside the chair;
        * a hand motionless for IDLE_S counts as put down (a resting hand on the
          desk is almost perfectly still). Any movement brings it straight back;
        * a hand that drops out keeps coasting for HAND_HOLD_S so a one-frame
          detection gap does not make the blade blink;
        * a sample that jumps impossibly far (MAX_JUMP per frame) is a pose
          estimate rather than a hand: the point is held, and a hand that keeps
          doing it for TELEPORT_DROP_S is dropped. This is what stops the
          "invented second hand jittering around the screen".
        """
        out: dict[str, BladePoint] = {}
        refreshed: set[str] = set()  # saw this hand in this frame at all
        for side, wrist_i, pinky_i, index_i in PALM_TRIPLES:
            if now < self._blocked.get(side, 0.0):
                continue  # dropped for teleporting moments ago: leave it out
            w, p, q = lms[wrist_i], lms[pinky_i], lms[index_i]
            vis = min(w.visibility, p.visibility, q.visibility)
            st = self._hands.get(side)
            if vis < (VISIBILITY_KEEP if st else VISIBILITY_MIN):
                self._note(side, f"invisible vis={vis:.2f}")
                continue
            if not self._is_raised(lms, side, w.y, acquiring=st is None):
                self._note(side, "lowered (wrist below elbow)")
                continue
            refreshed.add(side)
            # palm centre: wrist + the two knuckles average to the middle of the
            # palm, unlike the finger midpoint used earlier
            cx = (w.x + p.x + q.x) / 3.0
            cy = (w.y + p.y + q.y) / 3.0
            if st is None:
                st = self._hands[side] = {
                    "x": cx,
                    "y": cy,
                    "rx": cx,  # last raw (unsmoothed) sample we accepted
                    "ry": cy,
                    "prx": cx,  # last raw sample at all, for the swipe direction
                    "pry": cy,
                    "ux": 0.0,  # unit vector of the last raw step
                    "uy": 0.0,
                    "seen": now,
                    "speed": 0.0,  # smoothed per-sample displacement
                    "still": now,
                }
            else:
                # per-sample displacement BEFORE smoothing: a resting hand only
                # jitters by a fraction of a percent, a playing one covers whole
                # percent per frame. This, not raw position, separates the two.
                step = math.hypot(cx - st["rx"], cy - st["ry"])
                # direction of this raw step vs the previous one: a real swipe
                # keeps going one way, a hallucinated wrist reverses constantly
                rdx, rdy = cx - st["prx"], cy - st["pry"]
                rlen = math.hypot(rdx, rdy)
                st["prx"], st["pry"] = cx, cy
                aligned = rlen > 1e-6 and (st["ux"] * rdx + st["uy"] * rdy) / rlen > FAST_DIR_COS
                if step > MAX_JUMP and not aligned:
                    # A pose model asked for one person still estimates the arm it
                    # cannot see; that estimate jumps between frames in no
                    # consistent direction. A real hand moving fast (a slice) does
                    # exceed MAX_JUMP -- but it keeps its direction, which is what
                    # aligned tracks. Hold the point, drop a hand that only jumps.
                    if st.get("tele_since") is None:
                        st["tele_since"] = now
                    if now - st["tele_since"] >= TELEPORT_DROP_S:
                        del self._hands[side]
                        self._blocked[side] = now + TELEPORT_BLOCK_S
                        self._note(side, f"teleporting jump={step:.2f} -> dropped")
                        continue
                    st["seen"] = now
                    out[side] = BladePoint(x=st["x"], y=st["y"], visibility=vis, side=side)
                    self._note(side, f"held jump={step:.2f}")
                    continue
                st["tele_since"] = None
                if rlen > 1e-6:
                    st["ux"], st["uy"] = rdx / rlen, rdy / rlen
                st["speed"] += (step - st["speed"]) * SPEED_ALPHA
                st["rx"], st["ry"] = cx, cy
                st["x"] += (cx - st["x"]) * SMOOTH_ALPHA
                st["y"] += (cy - st["y"]) * SMOOTH_ALPHA
                st["seen"] = now
                if st["speed"] >= IDLE_SPEED:
                    st["still"] = now
            if now - st["still"] <= IDLE_S:
                out[side] = BladePoint(x=st["x"], y=st["y"], visibility=vis, side=side)
                self._note(side, f"active vis={vis:.2f} speed={st['speed']:.4f}")
            else:
                self._note(side, f"idle vis={vis:.2f} speed={st['speed']:.4f}")
        # coast / retire hands that were not refreshed this frame
        for side, st in list(self._hands.items()):
            if side in out or side in refreshed:
                # refreshed but idle: deliberately not a blade, do not coast it
                if side not in out and now - st["seen"] > HAND_HOLD_S:
                    del self._hands[side]
                continue
            if now - st["seen"] <= HAND_HOLD_S:
                out[side] = BladePoint(x=st["x"], y=st["y"], visibility=0.0, side=side)
            else:
                del self._hands[side]
        # stable order: left first, so the renderer's labels never swap
        return [out[s] for s in ("left", "right") if s in out]

    @staticmethod
    def _is_raised(lms, side: str, wrist_y: float, acquiring: bool) -> bool:
        """Playing hand vs arm put down, with hysteresis.

        Acquiring is strict (the wrist must be clearly above the elbow);
        keeping is lenient (it may sag a little below). Without the gap the
        gate flip-flops every frame near the boundary -- the pose elbow estimate
        jitters by a few percent and the blade would blink on and off.
        """
        elbow = lms[ELBOWS[side]]
        if elbow.visibility < VISIBILITY_KEEP:
            return True  # elbow hidden behind the desk: do not punish the hand
        margin = RAISE_ACQUIRE if acquiring else RAISE_KEEP
        return wrist_y <= elbow.y + margin

    def _log_speeds(self) -> None:
        """Debug: per-hand speed vs the idle threshold, every 60 frames."""
        parts = []
        for side in ("left", "right"):
            st = self._hands.get(side)
            if st is None:
                parts.append(f"{side}=none")
            else:
                still = time.perf_counter() - st["still"]
                parts.append(f"{side}=speed {st['speed']:.4f} still {still:.1f}s")
        print("[speed] " + "  ".join(parts), file=sys.stderr, flush=True)

    def _note(self, side: str, state: str) -> None:
        """Print per-hand decisions when AIRFRUIT_DEBUG_HANDS is set.

        "Why does the game still show two hands?" is impossible to answer from
        the outside; this prints the reason each side is (not) a blade without
        spamming: only transitions are reported.
        """
        if not _DEBUG_HANDS:
            return
        prev = self._noted.get(side)
        bucket = state.split(" ")[0]
        if bucket != prev:
            self._noted[side] = bucket
            print(f"[hands] {side}: {state}", file=sys.stderr, flush=True)

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
                    blades = self._extract_hands(poses[0], time.perf_counter())
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
            if _DEBUG_HANDS and frame_i % 60 == 0:
                self._log_speeds()

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
