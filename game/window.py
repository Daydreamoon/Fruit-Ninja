"""AirFruit game window: QTimer-driven 60fps rendering (M1) + vision preview (M2).

Render thread owns nothing vision-related: every tick it snapshots the
SharedVisionState written by the vision thread and draws from it.
"""
from __future__ import annotations

import math
import sys
import time

from PySide6.QtCore import QRect, Qt, QTimer
from PySide6.QtGui import (
    QColor,
    QFont,
    QImage,
    QLinearGradient,
    QPainter,
    QPen,
    QPixmap,
    QRadialGradient,
)
from PySide6.QtWidgets import QWidget

from game.audio import AudioBank
from game.recorder import Recorder
from game.state import GameState
from game.trail import BladeTrail
from vision.shared import PREVIEW_W, PREVIEW_H, SharedVisionState
from vision.thread import VisionThread

GAME_W, GAME_H = 1280, 720
DEMO_PUSH_PERIOD = 1.0 / 30.0  # trail demo feeds the trail at vision rate
DEMO_BLADE_SPEED = 4200.0  # px/s — fast enough to reach a falling fruit
HAND_LABELS = {"left": "左手", "right": "右手"}
HAND_MODE_LABELS = {"auto": "自动（举起来的手）", "left": "只认左手", "right": "只认右手"}

# Camera-box -> screen mapping. The default maps most of the frame; a 3 second
# auto-calibration at start-up shrinks it to the box the player's hands
# actually move in, so a relaxed arm movement covers the whole screen instead
# of demanding a dance across the field of view.
CALIB_SECONDS = 3.0
CALIB_MIN_SPAN = 0.10  # less movement than this in an axis = keep the default
CALIB_PAD = 0.06  # margin around the measured box, as a fraction of the frame
# A box smaller than this gets widened (centred on what was measured): a
# player who calibrates with one hand only would otherwise push the other hand
# into a clamp at the screen edge.
CALIB_MIN_BOX = 0.50
CALIB_DEFAULT = (0.10, 0.90, 0.06, 0.94)  # x0, x1, y0, y1
CALIB_BANNER_S = 2.0  # keep the "done" note on screen this long


def _widen(lo: float, hi: float, min_span: float) -> tuple[float, float]:
    """Grow [lo, hi] to at least min_span, keeping it centred and inside 0..1."""
    span = hi - lo
    if span >= min_span:
        return lo, hi
    centre = (lo + hi) / 2
    lo = max(0.0, centre - min_span / 2)
    hi = min(1.0, lo + min_span)
    lo = max(0.0, hi - min_span)
    return lo, hi


class GameWindow(QWidget):
    def __init__(
        self,
        use_synthetic: bool = False,
        camera_index: int = 0,
        audio: bool = True,
        record_path: str | None = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle("AirFruit · 隔空切水果")
        self.setFixedSize(GAME_W, GAME_H)

        self._t0 = time.perf_counter()
        self._last_tick = self._t0
        self._tick_times: list[float] = []
        self._paint_times: list[float] = []
        self._pulse_t = 0.0  # dt-integrated idle animation
        self.hand_mode = "auto"  # auto | left | right (H / 1 / 2 / 3 keys)
        self.tick_count = 0
        self.paint_count = 0

        self._bg: QPixmap | None = None
        self._make_background()

        # --- vision layer (M2) ---
        self.vision_shared = SharedVisionState()
        self._vision = VisionThread(self.vision_shared, camera_index=camera_index, synthetic=use_synthetic)
        self._vis: tuple = ([], None, "init", 0.0, 0, 0, 0.0)
        self._vision_started = False

        # --- blade trails (M3) ---
        # keyed by side: a hand keeps its own trail even when the other one
        # leaves, so a single-hand player never inherits a stale second blade
        self._trails: dict[str, BladeTrail] = {"left": BladeTrail(), "right": BladeTrail()}
        self._last_pub_t = 0.0
        # synthetic vision sees no person, so the selftest feeds the trails a
        # scripted two-hand sweep instead — this exercises the M3 render path.
        self._trail_demo = use_synthetic
        self._demo_t = 0.0
        self._demo_acc = 0.0
        self._last_blade_t = 0.0  # for the "no hand detected" hint
        self._not_ready_since: float | None = None  # start-up clock for the hint
        # auto-calibration of the camera->screen box (see _update_calibration)
        self._calib_box: tuple[float, float, float, float] | None = None
        self._calib_start_t: float | None = None
        self._calib_done_t: float | None = None
        self._calib_pts: list[tuple[float, float]] = []
        # scripted hands chase fruit, so the synthetic selftest slices reliably
        self._demo_pos = {
            "left": (GAME_W * 0.42, GAME_H * 0.62),
            "right": (GAME_W * 0.58, GAME_H * 0.38),
        }

        # --- game state (M4) ---
        self.state = GameState()

        # --- polish layer (M5): procedural sound + optional MP4 capture ---
        self.audio = AudioBank(enabled=audio)
        self.recorder = Recorder(record_path) if record_path else None
        self._grabbing = False

        self._timer = QTimer(self)
        self._timer.setInterval(16)  # ~60 Hz request
        self._timer.timeout.connect(self._tick)

    # ---- lifecycle ----

    def start(self) -> None:
        self._timer.start()
        self._vision.start()
        self._vision_started = True
        if self.recorder is not None:
            self.recorder.start((GAME_W, GAME_H))

    def shutdown(self) -> None:
        """Idempotent stop of both loops; safe from closeEvent and selftest."""
        if self._timer.isActive():
            self._timer.stop()
        if self._vision_started:
            self._vision.stop()
            self._vision_started = False
        if self.recorder is not None:
            self.recorder.stop()

    def closeEvent(self, event) -> None:  # noqa: N802
        self.shutdown()
        super().closeEvent(event)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        """Hand-mode override: the automatic rules (raised + moving) cover the
        usual case, a key is the escape hatch when they guess wrong."""
        key = event.key()
        if key in (Qt.Key.Key_H, Qt.Key.Key_H_multiple):
            order = ["auto", "left", "right"]
            self.hand_mode = order[(order.index(self.hand_mode) + 1) % len(order)]
        elif key == Qt.Key.Key_1:
            self.hand_mode = "left"
        elif key == Qt.Key.Key_2:
            self.hand_mode = "right"
        elif key == Qt.Key.Key_3:
            self.hand_mode = "auto"
        else:
            super().keyPressEvent(event)
            return
        print(f"[AirFruit] 用手模式：{HAND_MODE_LABELS[self.hand_mode]}", file=sys.stderr, flush=True)
        self.update()

    def _tick(self) -> None:
        now = time.perf_counter()
        dt = min(now - self._last_tick, 0.05)  # clamp stalls (window drags etc.)
        self._last_tick = now
        self._tick_times.append(now)
        self.tick_count += 1

        self._pulse_t += dt
        self._vis = self.vision_shared.snapshot()
        if self.hand_mode != "auto":
            # manual override: keep only the chosen hand, downstream code paths
            # (calibration, trails, cursors, preview dots) then all agree
            self._vis = (
                [b for b in self._vis[0] if b.side == self.hand_mode],
                *self._vis[1:],
            )
        if self._vis[0]:
            self._last_blade_t = now
        self._update_calibration(now)
        self._feed_trails(dt)

        # --- M4 game step: physics/waves, then blade-vs-fruit collisions ---
        # Real-camera start-up takes a few seconds (model + camera). Freezing the
        # world until vision reports "ok" keeps the first round fair: no fruit is
        # spawned (and missed) while the player cannot see their own hands yet.
        if self._world_ready():
            self._not_ready_since = None
            self.state.update(dt)
            if not self.state.game_over:
                now = time.perf_counter()
                segs: list[tuple[float, float, float, float]] = []
                for tr in self._trails.values():
                    segs.extend(tr.collision_segments(now))
                if segs:
                    self.state.collide_blade(segs)
        elif self._not_ready_since is None:
            self._not_ready_since = now  # start of the "still starting" clock
        self._flush_audio_events()
        self.update()
        if self.recorder is not None and self.recorder.status == "recording":
            # grab() paints synchronously into a pixmap; flag the extra pass so
            # the fps counters keep measuring real presentation frames only.
            self._grabbing = True
            try:
                self.recorder.capture(self.grab().toImage())
            finally:
                self._grabbing = False

    def _flush_audio_events(self) -> None:
        """Turn game events raised this tick into sound (M5)."""
        events = self.state.events
        if not events:
            return
        del events[:]  # drain regardless of whether audio is enabled
        if not self.audio.enabled:
            return
        for name in events:
            if name == "boom":
                self.audio.play("boom")
                QTimer.singleShot(450, lambda: self.audio.play("gameover"))
            else:
                self.audio.play(name)

    def _world_ready(self) -> bool:
        """True once blades can actually arrive (synthetic mode is always ready).

        Camera start-up costs a few seconds; spawning fruit before that would
        hand the player a pile of unavoidable misses.
        """
        if self._trail_demo:
            return True
        return self._vis[2] == "ok"

    def _push_demo_blades(self, now: float) -> None:
        """Scripted hands for --selftest: chase fruit, sweep when there is none.

        The synthetic camera sees no person, so the selftest drives the trails
        itself. Aiming at the lowest *fruit* (never a bomb, value==0) makes the
        slice assertion deterministic instead of depending on where a fixed
        sweep happens to cross a randomly spawned fruit. Speed is bounded so the
        trail still reads as a hand and the collision segments look real.
        """
        targets = sorted(
            (f for f in self.state.fruits if f.value > 0), key=lambda f: f.y, reverse=True
        )
        for k, side in enumerate(("left", "right")):
            x, y = self._demo_pos[side]
            if k < len(targets):
                tx, ty = targets[k].x, targets[k].y
            else:  # idle sweep while waiting for the next wave
                tx = GAME_W * (0.5 + 0.35 * math.sin(self._demo_t * 1.7 + k * 2.1))
                ty = GAME_H * (0.5 + 0.28 * math.sin(self._demo_t * 2.3 + k * 1.3))
            dx, dy = tx - x, ty - y
            dist = math.hypot(dx, dy)
            if dist > 1e-6:
                step = min(dist, DEMO_BLADE_SPEED * DEMO_PUSH_PERIOD)
                x += dx / dist * step
                y += dy / dist * step
            self._demo_pos[side] = (x, y)
            self._trails[side].push(x, y, now)

    def _update_calibration(self, now: float) -> None:
        """Learn the box where this player's hands move (3 s, once, at start).

        Without it, a seated player whose hands sit low in the frame can only
        reach the bottom of the screen and cannot understand why -- the classic
        "I don't know where my hand is pointing" complaint.
        """
        if self._trail_demo or self._calib_box is not None or not self._world_ready():
            return
        if self._calib_start_t is None:
            self._calib_start_t = now
        for b in self._vis[0]:
            self._calib_pts.append((b.x, b.y))
        if now - self._calib_start_t >= CALIB_SECONDS:
            self._finish_calibration(now)

    def _finish_calibration(self, now: float) -> None:
        xs = [p[0] for p in self._calib_pts]
        ys = [p[1] for p in self._calib_pts]
        if xs and (max(xs) - min(xs)) >= CALIB_MIN_SPAN and (max(ys) - min(ys)) >= CALIB_MIN_SPAN:
            x0 = max(0.0, min(xs) - CALIB_PAD)
            x1 = min(1.0, max(xs) + CALIB_PAD)
            y0 = max(0.0, min(ys) - CALIB_PAD)
            y1 = min(1.0, max(ys) + CALIB_PAD)
            x0, x1 = _widen(x0, x1, CALIB_MIN_BOX)
            y0, y1 = _widen(y0, y1, CALIB_MIN_BOX)
            self._calib_box = (x0, x1, y0, y1)
            print(
                f"[calib] hands move in x {x0:.2f}..{x1:.2f}, y {y0:.2f}..{y1:.2f}"
                f" ({len(xs)} samples)",
                file=sys.stderr,
            )
        else:
            self._calib_box = CALIB_DEFAULT
            print(
                f"[calib] too little movement ({len(xs)} samples) -- default box",
                file=sys.stderr,
            )
        self._calib_done_t = now

    def _map_blade(self, b) -> tuple[float, float]:
        """Camera coordinates (0..1) -> screen coordinates (0..1)."""
        x0, x1, y0, y1 = self._calib_box or CALIB_DEFAULT
        nx = (b.x - x0) / max(1e-6, x1 - x0)
        ny = (b.y - y0) / max(1e-6, y1 - y0)
        return min(1.0, max(0.0, nx)), min(1.0, max(0.0, ny))

    def _feed_trails(self, dt: float) -> None:
        """Push the newest blade samples into the trails.

        Real mode: a new vision publish is detected by a changed pub_t; each
        hand index present in the publish extends its trail (the other one
        just stops receiving and fades out). Demo mode (synthetic vision sees
        no person): a scripted two-hand sweep at the vision rate, so
        --selftest exercises the full M3 render path.
        """
        now = time.perf_counter()
        if self._trail_demo:
            self._demo_t += dt
            self._demo_acc += dt
            if self._demo_acc >= DEMO_PUSH_PERIOD:
                self._demo_acc = 0.0
                self._push_demo_blades(now)
            return
        blades, _preview, _status, _infer, _hz, _total, pub_t = self._vis
        if pub_t == self._last_pub_t:
            return  # same vision sample: nothing new to push
        self._last_pub_t = pub_t
        for b in blades:
            trail = self._trails.get(b.side)
            if trail is None:
                continue
            sx, sy = self._map_blade(b)
            trail.push(sx * GAME_W, sy * GAME_H, pub_t)

    # ---- painting ----

    def _make_background(self) -> None:
        """Render the static background once (plan: never rebuild per frame)."""
        px = QPixmap(GAME_W, GAME_H)
        p = QPainter(px)
        grad = QLinearGradient(0, 0, 0, GAME_H)
        grad.setColorAt(0.0, QColor(18, 24, 38))
        grad.setColorAt(1.0, QColor(8, 10, 16))
        p.fillRect(0, 0, GAME_W, GAME_H, grad)
        glow = QRadialGradient(GAME_W / 2, GAME_H + 120, GAME_H)
        glow.setColorAt(0.0, QColor(255, 120, 60, 60))
        glow.setColorAt(1.0, QColor(0, 0, 0, 0))
        p.fillRect(0, 0, GAME_W, GAME_H, glow)
        p.end()
        self._bg = px

    def paintEvent(self, event) -> None:  # noqa: N802
        now = time.perf_counter()
        if not self._grabbing:  # recorder's grab() pass is not a shown frame
            self._paint_times.append(now)
            self.paint_count += 1

        p = QPainter(self)
        p.drawPixmap(0, 0, self._bg)

        blades, preview, status, infer_ms, vis_hz, _total, _pub_t = self._vis

        # M3 blade trails: fade-out tail + extrapolated tip between samples
        for trail in self._trails.values():
            trail.draw(p, now)
        n_trail_pts = sum(t.stats()["points"] for t in self._trails.values())

        # --- M4 game objects ---
        st = self.state
        for f in st.fruits:
            self._draw_fruit(p, f)
        for h in st.halves:
            self._draw_half(p, h)
        for q in st.particles:
            alpha = max(0.0, min(1.0, q.life / 0.45))
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(q.color[0], q.color[1], q.color[2], int(200 * alpha)))
            p.drawEllipse(int(q.x - 4), int(q.y - 4), 8, 8)
        p.setFont(QFont("Microsoft YaHei", 14, QFont.Weight.Bold))
        for t in st.texts:
            alpha = max(0.0, min(1.0, t.life / 0.9))
            p.setPen(QColor(t.color[0], t.color[1], t.color[2], int(230 * alpha)))
            p.drawText(int(t.x), int(t.y), t.text)
        p.setBrush(Qt.NoBrush)
        if st.game_over:
            self._draw_game_over(p)

        # idle placeholder only while nothing is tracked at all (never during
        # the camera warm-up: it collided with the hint text)
        if not blades and n_trail_pts == 0 and self._world_ready():
            r = 60 + 18 * math.sin(self._pulse_t * 2 * math.pi * 0.5)
            cx, cy = GAME_W / 2, GAME_H / 2
            pen = QPen(QColor(255, 170, 60))
            pen.setWidth(3)
            p.setPen(pen)
            p.setBrush(QColor(255, 170, 60, 40))
            p.drawEllipse(int(cx - r), int(cy - r), int(2 * r), int(2 * r))
            p.setBrush(Qt.NoBrush)

        # real-camera warm-up hint: say what we are waiting for instead of
        # letting the player stare at an empty field
        if not self._world_ready():
            ready_text = {
                "loading_model": "正在加载姿态模型…",
                "opening_camera": "正在打开摄像头…",
                "starting": "正在启动识别…",
            }.get(status, f"视觉未就绪（{status}）")
            p.setFont(QFont("Microsoft YaHei", 22, QFont.Weight.Bold))
            p.setPen(QColor(255, 210, 120))
            waited = max(0.0, now - (self._not_ready_since or now))
            p.drawText(
                QRect(0, int(GAME_H * 0.40), GAME_W, 60),
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
                ready_text,
            )
            p.setFont(QFont("Microsoft YaHei", 14))
            p.setPen(QColor(200, 170, 120))
            p.drawText(
                QRect(0, int(GAME_H * 0.40) + 44, GAME_W, 30),
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
                f"已等待 {waited:.0f} 秒（摄像头首次打开较慢，最多约 10 秒）· 就绪后请把手举到镜头前",
            )
        elif not self._trail_demo and not blades and now - self._last_blade_t > 1.2:
            # camera is running but nothing is tracked: almost always "hand not
            # in frame / too close to the lens / in the dark"
            p.setFont(QFont("Microsoft YaHei", 20, QFont.Weight.Bold))
            p.setPen(QColor(255, 180, 90))
            p.drawText(
                QRect(0, int(GAME_H * 0.30), GAME_W, 40),
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
                "没检测到手 —— 把手举到镜头前，正对摄像头（右下角是摄像头画面）",
            )

        # the scripted selftest must never be mistaken for real input
        if self._trail_demo:
            p.setFont(QFont("Microsoft YaHei", 16, QFont.Weight.Bold))
            p.setPen(QColor(255, 120, 120))
            p.drawText(
                QRect(16, int(GAME_H) - 46, GAME_W - 32, 30),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                "自动演示模式：刀刃由脚本驱动，与你的手无关（不加 --selftest 就是真机模式）",
            )

        # where the blade currently is: the published hand points in live mode,
        # the extrapolated trail tip in demo mode (synthetic vision has no hands)
        if blades:
            cursors = [
                (b.side, *[v * s for v, s in zip(self._map_blade(b), (GAME_W, GAME_H))])
                for b in blades
            ]
        elif self._trail_demo:
            cursors = [
                (side, *t.tip(now))
                for side, t in self._trails.items()
                if t.stats()["points"] > 0
            ]
        else:
            cursors = []

        for side, x, y in cursors:
            # big pulsing reticle: at 60fps a 24px circle reads as "a dot you
            # cannot find" -- the player must see where their hand maps to
            pulse = 0.5 + 0.5 * math.sin(self._pulse_t * 2 * math.pi * 0.8)
            outer = 30 + 4 * pulse
            pen = QPen(QColor(255, 220, 120, 190), 3)
            p.setPen(pen)
            p.setBrush(QColor(255, 200, 80, 40))
            p.drawEllipse(int(x - outer), int(y - outer), int(2 * outer), int(2 * outer))
            p.setBrush(QColor(255, 235, 180, 150))
            p.drawEllipse(int(x - 6), int(y - 6), 12, 12)  # exact mapped point
            p.setPen(QPen(QColor(255, 200, 80, 150), 2))
            p.drawLine(int(x - outer - 12), int(y), int(x - outer), int(y))
            p.drawLine(int(x + outer), int(y), int(x + outer + 12), int(y))
            p.drawLine(int(x), int(y - outer - 12), int(x), int(y - outer))
            p.drawLine(int(x), int(y + outer), int(x), int(y + outer + 12))
            p.setFont(QFont("Microsoft YaHei", 12, QFont.Weight.Bold))
            p.setPen(QColor(255, 220, 140, 200))
            p.drawText(int(x + outer + 6), int(y - outer), HAND_LABELS.get(side, side))

        # the preview is a real-camera aid; in demo mode there is none to show
        if preview is not None and not self._trail_demo:
            img = QImage(preview, PREVIEW_W, PREVIEW_H, 3 * PREVIEW_W, QImage.Format.Format_BGR888)
            # 1.5x the stored 240x135 buffer: big enough to recognise your own
            # hand position, still cheap to upload every frame
            pw, ph = int(PREVIEW_W * 1.5), int(PREVIEW_H * 1.5)
            px_, py_ = GAME_W - pw - 16, GAME_H - ph - 32
            p.drawImage(QRect(px_, py_, pw, ph), img)
            p.setPen(QPen(QColor(90, 90, 90)))
            p.drawRect(px_, py_, pw, ph)
            # the calibrated region, so the player sees what the game reads
            if self._calib_box is not None:
                x0, x1, y0, y1 = self._calib_box
                rx, ry = px_ + x0 * pw, py_ + y0 * ph
                rw, rh = (x1 - x0) * pw, (y1 - y0) * ph
                p.setPen(QPen(QColor(120, 220, 140, 160), 1, Qt.PenStyle.DashLine))
                p.drawRect(int(rx), int(ry), int(rw), int(rh))
            # overlay the tracked hand points (camera coordinates) so "where
            # does my hand go?" is answerable at a glance instead of by trial
            for b in blades:
                mx, my = px_ + b.x * pw, py_ + b.y * ph
                p.setPen(QPen(QColor(255, 220, 120), 2))
                p.setBrush(QColor(255, 220, 120, 180))
                p.drawEllipse(int(mx - 5), int(my - 5), 10, 10)
            p.setBrush(Qt.NoBrush)
            p.setFont(QFont("Microsoft YaHei", 9))
            p.setPen(QColor(180, 180, 180))
            p.drawText(int(px_), int(py_) - 6, "摄像头预览 · 黄点=已识别的手（绿框=校准后的有效区）")

        if not self._trail_demo:
            # calibration narration
            if self._calib_box is None and self._calib_start_t is not None:
                left = max(0.0, CALIB_SECONDS - (now - self._calib_start_t))
                p.setFont(QFont("Microsoft YaHei", 16, QFont.Weight.Bold))
                p.setPen(QColor(120, 220, 140))
                p.drawText(
                    QRect(0, int(GAME_H * 0.30), GAME_W, 30),
                    Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
                    f"校准中 {left:.1f}s —— 请举双手在镜头前左右挥动（决定你的手势映射范围）",
                )
            elif self._calib_done_t is not None and now - self._calib_done_t < CALIB_BANNER_S:
                p.setFont(QFont("Microsoft YaHei", 16, QFont.Weight.Bold))
                p.setPen(QColor(120, 220, 140))
                p.drawText(
                    QRect(0, int(GAME_H * 0.30), GAME_W, 30),
                    Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
                    "校准完成 —— 现在挥手就是刀刃",
                )

        self._draw_hud(p, status, infer_ms, vis_hz, len(blades))

    def _draw_fruit(self, p: QPainter, f) -> None:
        x, y, r = int(f.x), int(f.y), int(f.radius)
        if f.bomb:
            blink = 0.5 + 0.5 * math.sin(self._pulse_t * 8)
            p.setPen(QPen(QColor(220, 60, 40, int(120 + 120 * blink)), 3))
            p.setBrush(QColor(45, 45, 52))
            p.drawEllipse(x - r, y - r, 2 * r, 2 * r)
            p.setPen(QColor(255, 140, 80))
            p.drawLine(x, y - r - 10, x, y - r + 6)  # fuse stub
            return
        grad = QRadialGradient(f.x - f.radius * 0.35, f.y - f.radius * 0.35, f.radius * 1.35)
        grad.setColorAt(0.0, QColor(*f.core))
        grad.setColorAt(1.0, QColor(f.rim[0], f.rim[1], f.rim[2], 170))
        p.setPen(Qt.NoPen)
        p.setBrush(grad)
        p.drawEllipse(x - r, y - r, 2 * r, 2 * r)

    def _draw_half(self, p: QPainter, h) -> None:
        alpha = max(0.0, min(1.0, h.life / h.max_life))
        p.save()
        p.translate(h.x, h.y)
        p.rotate(math.degrees(h.angle))
        p.setPen(Qt.NoPen)
        c = h.color
        p.setBrush(QColor(c[0], c[1], c[2], int(220 * alpha)))
        r = int(h.radius)
        p.drawPie(-r, -r, 2 * r, 2 * r, 0, 180 * 16)
        p.restore()

    def _draw_game_over(self, p: QPainter) -> None:
        p.fillRect(0, 0, GAME_W, GAME_H, QColor(0, 0, 0, 140))
        p.setPen(QColor(255, 90, 70))
        p.setFont(QFont("Microsoft YaHei", 42, QFont.Weight.Bold))
        p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, f"GAME OVER · {self.state.score} 分")

    def _draw_hud(self, p: QPainter, status: str, infer_ms: float, vis_hz: int, n_blades: int) -> None:
        fps = self.paint_fps()
        ms = self.frame_ms()
        p.setPen(QColor(230, 230, 230))
        p.setFont(QFont("Microsoft YaHei", 13))
        p.drawText(16, 26, f"FPS {fps:5.1f}   frame {ms:4.1f} ms   ticks {self.tick_count}")
        p.setPen(QColor(150, 150, 150))
        p.drawText(16, 48, f"M4 玩法闭环 · M5 打磨 · sfx {self.audio.status}")
        if status == "camera_open_failed" or status.startswith("camera_"):
            p.setPen(QColor(255, 120, 90))
        p.drawText(16, 70, f"vision: {status}   {vis_hz} Hz   infer {infer_ms:4.1f} ms   blades {n_blades}")
        p.setPen(QColor(150, 150, 150))
        p.drawText(16, 92, f"用手：{HAND_MODE_LABELS[self.hand_mode]}   按键 H 切换 / 1 左手 / 2 右手 / 3 自动")
        p.setPen(QColor(255, 230, 140))
        p.drawText(16, 114, self.state.hud())

    # ---- stats (sliding 1-second windows) ----

    def _trim(self, stamps: list[float]) -> int:
        now = time.perf_counter()
        keep = 0
        for i, t in enumerate(stamps):
            if now - t <= 1.0:
                keep = len(stamps) - i
                break
        del stamps[: len(stamps) - keep]
        return keep

    def paint_fps(self) -> float:
        return float(self._trim(self._paint_times))

    def frame_ms(self) -> float:
        self._trim(self._tick_times)
        if len(self._tick_times) < 2:
            return 0.0
        gaps = [b - a for a, b in zip(self._tick_times, self._tick_times[1:])]
        return 1000.0 * sum(gaps) / len(gaps)

    def summary(self) -> dict:
        elapsed = max(time.perf_counter() - self._t0, 1e-9)
        return {
            "seconds": elapsed,
            "ticks": self.tick_count,
            "paints": self.paint_count,
            "avg_tick_fps": self.tick_count / elapsed,
            "avg_paint_fps": self.paint_count / elapsed,
            "trail_points": max(
                (t.stats()["points"] for t in self._trails.values()), default=0
            ),
            "game": self.state.stats.as_dict(),
            "audio": self.audio.status,
            "muted": self.audio.muted,
            "record": self.recorder.summary() if self.recorder is not None else None,
        }
