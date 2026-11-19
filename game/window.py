"""AirFruit game window: QTimer-driven 60fps rendering skeleton (M1).

Design (from the project plan):
- One QTimer at 16 ms drives the whole game; every tick integrates dt-based
  physics then calls update(); paintEvent draws from cached surfaces.
- Background is rendered ONCE into a QPixmap; per-frame painting is cheap.
- HUD shows live paint FPS / frame ms so the 60fps hard target is visible.
"""
from __future__ import annotations

import math
import time

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import (
    QColor,
    QFont,
    QLinearGradient,
    QPainter,
    QPen,
    QPixmap,
    QRadialGradient,
)
from PySide6.QtWidgets import QWidget

GAME_W, GAME_H = 1280, 720


class GameWindow(QWidget):
    """Owns the tick timer; state grows milestone by milestone."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AirFruit · 隔空切水果")
        self.setFixedSize(GAME_W, GAME_H)

        self._t0 = time.perf_counter()
        self._last_tick = self._t0
        self._tick_times: list[float] = []
        self._paint_times: list[float] = []
        self._pulse_t = 0.0  # dt-integrated demo animation (M1 placeholder)
        self.tick_count = 0
        self.paint_count = 0

        self._bg: QPixmap | None = None
        self._make_background()

        self._timer = QTimer(self)
        self._timer.setInterval(16)  # ~60 Hz request
        self._timer.timeout.connect(self._tick)

    # ---- lifecycle ----

    def start(self) -> None:
        self._timer.start()

    def _tick(self) -> None:
        now = time.perf_counter()
        dt = min(now - self._last_tick, 0.05)  # clamp stalls (window drags etc.)
        self._last_tick = now
        self._tick_times.append(now)
        self.tick_count += 1

        self._pulse_t += dt
        self.update()

    # ---- painting ----

    def _make_background(self) -> None:
        """Render the static background once (plan: never rebuild per frame)."""
        px = QPixmap(GAME_W, GAME_H)
        p = QPainter(px)
        grad = QLinearGradient(0, 0, 0, GAME_H)
        grad.setColorAt(0.0, QColor(18, 24, 38))
        grad.setColorAt(1.0, QColor(8, 10, 16))
        p.fillRect(0, 0, GAME_W, GAME_H, grad)
        # warm floor glow, hinting where fruits will rise
        glow = QRadialGradient(GAME_W / 2, GAME_H + 120, GAME_H)
        glow.setColorAt(0.0, QColor(255, 120, 60, 60))
        glow.setColorAt(1.0, QColor(0, 0, 0, 0))
        p.fillRect(0, 0, GAME_W, GAME_H, glow)
        p.end()
        self._bg = px

    def paintEvent(self, event) -> None:  # noqa: N802
        now = time.perf_counter()
        self._paint_times.append(now)
        self.paint_count += 1

        p = QPainter(self)
        p.drawPixmap(0, 0, self._bg)

        # dt-driven pulse circle: placeholder for real entities, and proof
        # that the loop integrates time (radius depends on _pulse_t, not frame no.)
        r = 60 + 18 * math.sin(self._pulse_t * 2 * math.pi * 0.5)
        cx, cy = GAME_W / 2, GAME_H / 2
        pen = QPen(QColor(255, 170, 60))
        pen.setWidth(3)
        p.setPen(pen)
        p.setBrush(QColor(255, 170, 60, 40))
        p.drawEllipse(int(cx - r), int(cy - r), int(2 * r), int(2 * r))
        p.setBrush(Qt.NoBrush)

        self._draw_hud(p)

    def _draw_hud(self, p: QPainter) -> None:
        fps = self.paint_fps()
        ms = self.frame_ms()
        p.setPen(QColor(230, 230, 230))
        p.setFont(QFont("Microsoft YaHei", 13))
        p.drawText(16, 26, f"FPS {fps:5.1f}   frame {ms:4.1f} ms   ticks {self.tick_count}")
        p.setPen(QColor(150, 150, 150))
        p.drawText(16, 48, "M1 渲染骨架 · 视觉线程待接入")

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
        }
