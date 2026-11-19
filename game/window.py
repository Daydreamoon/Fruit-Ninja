"""AirFruit game window: QTimer-driven 60fps rendering (M1) + vision preview (M2).

Render thread owns nothing vision-related: every tick it snapshots the
SharedVisionState written by the vision thread and draws from it.
"""
from __future__ import annotations

import math
import time

from PySide6.QtCore import Qt, QTimer
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
        self._trails = [BladeTrail(), BladeTrail()]
        self._last_pub_t = 0.0
        # synthetic vision sees no person, so the selftest feeds the trails a
        # scripted two-hand sweep instead — this exercises the M3 render path.
        self._trail_demo = use_synthetic
        self._demo_t = 0.0
        self._demo_acc = 0.0

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

    def _tick(self) -> None:
        now = time.perf_counter()
        dt = min(now - self._last_tick, 0.05)  # clamp stalls (window drags etc.)
        self._last_tick = now
        self._tick_times.append(now)
        self.tick_count += 1

        self._pulse_t += dt
        self._vis = self.vision_shared.snapshot()
        self._feed_trails(dt)

        # --- M4 game step: physics/waves, then blade-vs-fruit collisions ---
        self.state.update(dt)
        if not self.state.game_over:
            now = time.perf_counter()
            segs: list[tuple[float, float, float, float]] = []
            for tr in self._trails:
                segs.extend(tr.collision_segments(now))
            if segs:
                self.state.collide_blade(segs)
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
                # hand 0: wide sweep; hand 1: tight orbit through the fruit
                # corridor (x 0.32..0.68W, apex band) so the demo actually slices
                for k in range(2):
                    if k == 0:
                        x = GAME_W * (0.5 + 0.35 * math.sin(self._demo_t * 1.7))
                        y = GAME_H * (0.5 + 0.28 * math.sin(self._demo_t * 2.3))
                    else:
                        x = GAME_W * (0.5 + 0.20 * math.sin(self._demo_t * 3.1 + 1.0))
                        y = GAME_H * (0.42 + 0.16 * math.sin(self._demo_t * 4.3 + 1.7))
                    self._trails[k].push(x, y, now)
            return
        blades, _preview, _status, _infer, _hz, _total, pub_t = self._vis
        if pub_t == self._last_pub_t:
            return  # same vision sample: nothing new to push
        self._last_pub_t = pub_t
        for k, b in enumerate(blades[:2]):
            self._trails[k].push(b.x * GAME_W, b.y * GAME_H, pub_t)

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
        for trail in self._trails:
            trail.draw(p, now)
        n_trail_pts = sum(t.stats()["points"] for t in self._trails)

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

        # idle placeholder only while nothing is tracked at all
        if not blades and n_trail_pts == 0:
            r = 60 + 18 * math.sin(self._pulse_t * 2 * math.pi * 0.5)
            cx, cy = GAME_W / 2, GAME_H / 2
            pen = QPen(QColor(255, 170, 60))
            pen.setWidth(3)
            p.setPen(pen)
            p.setBrush(QColor(255, 170, 60, 40))
            p.drawEllipse(int(cx - r), int(cy - r), int(2 * r), int(2 * r))
            p.setBrush(Qt.NoBrush)

        for b in blades:
            x, y = b.x * GAME_W, b.y * GAME_H
            pen = QPen(QColor(255, 200, 80))
            pen.setWidth(2)
            p.setPen(pen)
            p.setBrush(QColor(255, 200, 80, 60))
            p.drawEllipse(int(x - 12), int(y - 12), 24, 24)
            p.setBrush(Qt.NoBrush)
            p.drawLine(int(x - 20), int(y), int(x + 20), int(y))
            p.drawLine(int(x), int(y - 20), int(x), int(y + 20))

        if preview is not None:
            img = QImage(preview, PREVIEW_W, PREVIEW_H, 3 * PREVIEW_W, QImage.Format.Format_BGR888)
            px_, py_ = GAME_W - PREVIEW_W - 16, GAME_H - PREVIEW_H - 16
            p.drawImage(px_, py_, img)
            p.setPen(QPen(QColor(90, 90, 90)))
            p.drawRect(px_, py_, PREVIEW_W, PREVIEW_H)

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
        p.setPen(QColor(255, 230, 140))
        p.drawText(16, 92, self.state.hud())

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
            "trail_points": max((t.stats()["points"] for t in self._trails), default=0),
            "game": self.state.stats.as_dict(),
            "audio": self.audio.status,
            "muted": self.audio.muted,
            "record": self.recorder.summary() if self.recorder is not None else None,
        }
