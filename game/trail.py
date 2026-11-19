"""Blade trail rendering: fade-out tail + inter-sample interpolation (M3).

The vision thread pushes one point per hand at ~30 Hz. The render loop runs
at 60 fps, so between two pushes the visible tip would stutter (a fast hand
moves ~50 px per 33 ms sample). BladeTrail keeps the pushed history for the
glowing tail (0.25 s lifetime, like the reference project) and extrapolates
a virtual tip from the last two samples so the blade tip glides at render rate.

Coordinates: game pixels. The window maps normalized vision coords once,
right after snapshot; nothing here knows about vision.
"""
from __future__ import annotations

import math

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPen

TRAIL_LIFETIME = 0.25  # s — matches the reference project's KNIFE_TRAIL_LIFETIME
MAX_POINTS = 48  # hard cap; 0.25 s @30 Hz ≈ 8, cap is pure safety


class BladeTrail:
    """One hand's trail: pushed samples + an extrapolated virtual tip."""

    def __init__(self, lifetime: float = TRAIL_LIFETIME) -> None:
        self._lifetime = lifetime
        # history of (x, y, t_perf)
        self._pts: list[tuple[float, float, float]] = []
        self._last: tuple[float, float, float] | None = None
        self._prev: tuple[float, float, float] | None = None

    # ---- input (vision rate ~30 Hz) ----

    def push(self, x: float, y: float, t: float) -> None:
        if self._last is not None and t <= self._last[2]:
            return  # out-of-order / duplicate sample: keep history monotonic
        self._prev = self._last
        self._last = (x, y, t)
        self._pts.append(self._last)
        if len(self._pts) > MAX_POINTS:
            del self._pts[: len(self._pts) - MAX_POINTS]

    # ---- rendering (game rate ~60 fps) ----

    def tip(self, now: float) -> tuple[float, float] | None:
        """Extrapolated blade position at render time `now`.

        Linear velocity from the last two samples, capped at 1.5 sample
        periods (no runaway when the hand stops between samples).
        """
        if self._last is None:
            return None
        lx, ly, lt = self._last
        if self._prev is None:
            return lx, ly
        px, py, pt = self._prev
        dt_sample = max(lt - pt, 1e-6)
        vx = (lx - px) / dt_sample
        vy = (ly - py) / dt_sample
        ahead = min(max(now - lt, 0.0), 1.5 * dt_sample)
        return lx + vx * ahead, ly + vy * ahead

    def draw(self, p: QPainter, now: float) -> None:
        self._prune(now)
        if len(self._pts) < 2:
            return
        pts = self._pts
        tip = self.tip(now)
        segs: list[tuple[float, float, float, float, float]] = []
        for i in range(len(pts) - 1):
            x0, y0, t0 = pts[i]
            x1, y1, t1 = pts[i + 1]
            if math.hypot(x1 - x0, y1 - y0) < 0.5:
                continue  # sample noise: skip zero-length hair
            segs.append((x0, y0, x1, y1, t1))
        if tip is not None:
            lx, ly, lt = self._last
            tx, ty = tip
            if math.hypot(tx - lx, ty - ly) >= 0.5:
                segs.append((lx, ly, tx, ty, now))

        for x0, y0, x1, y1, t_new in segs:
            newness = max(0.0, 1.0 - (now - t_new) / self._lifetime)
            w = 2.0 + 9.0 * newness
            # Soft glow, then the bright core. Round caps/joins are what make
            # consecutive segments read as one smooth stroke instead of a row
            # of rectangles (M5: butt caps looked chunky at 30 Hz sampling).
            glow = QPen(QColor(255, 190, 110, int(60 * newness)), w * 2.2)
            glow.setCapStyle(Qt.PenCapStyle.RoundCap)
            glow.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            p.setPen(glow)
            p.drawLine(int(x0), int(y0), int(x1), int(y1))
            core = QPen(QColor(255, 252, 240, int(235 * newness)), w)
            core.setCapStyle(Qt.PenCapStyle.RoundCap)
            core.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            p.setPen(core)
            p.drawLine(int(x0), int(y0), int(x1), int(y1))

    def _prune(self, now: float) -> None:
        while self._pts and now - self._pts[0][2] > self._lifetime:
            self._pts.pop(0)

    def collision_segments(
        self, now: float, window: float = 0.1
    ) -> list[tuple[float, float, float, float]]:
        """Recent blade pieces (x0, y0, x1, y1) for slicing tests.

        Uses only the last `window` seconds of samples plus the extrapolated
        tip segment, so a fast swipe stays 'live' for slicing briefly after
        the hand moved on.
        """
        pts = [p for p in self._pts if now - p[2] <= window]
        segs: list[tuple[float, float, float, float]] = []
        for a, b in zip(pts, pts[1:]):
            segs.append((a[0], a[1], b[0], b[1]))
        tip = self.tip(now)
        if pts and tip is not None:
            lx, ly, _lt = self._last
            tx, ty = tip
            segs.append((lx, ly, tx, ty))
        return segs

    def stats(self) -> dict:
        return {"points": len(self._pts), "active": self._last is not None}
