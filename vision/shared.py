"""Shared state between the vision thread and the render thread (M2).

The vision thread writes, the render thread reads; a plain threading.Lock
guards everything — semantically the plan's QMutex, but stdlib Lock gives a
real context manager. Blade points are stored NORMALIZED (0..1, already
mirrored to the player's view); the window maps them to game pixels.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass

PREVIEW_W, PREVIEW_H = 240, 135  # camera preview sent to the render thread  # 16:9 mini camera preview


@dataclass
class BladePoint:
    x: float  # normalized 0..1, mirrored (player view)
    y: float
    visibility: float
    side: str = "right"  # "left" / "right": stable identity so trails don't swap


class SharedVisionState:
    """Latest vision results + rolling sample-rate stats, lock-guarded."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.blades: list[BladePoint] = []
        self.preview_bgr: bytes | None = None
        self.status = "init"
        self.infer_ms: float = 0.0
        self.pub_t: float = 0.0  # perf_counter of the last publish (new-sample marker)
        self._samples: deque[float] = deque(maxlen=120)
        self.sample_count = 0

    def publish(self, blades: list[BladePoint], preview_bgr: bytes, infer_ms: float) -> None:
        with self._lock:
            self.blades = blades
            self.preview_bgr = preview_bgr
            self.infer_ms = infer_ms
            self.status = "ok"
            self.pub_t = time.perf_counter()
            self._samples.append(self.pub_t)
            self.sample_count += 1

    def publish_status(self, status: str) -> None:
        with self._lock:
            self.status = status

    def snapshot(self):
        """(blades, preview_bgr, status, infer_ms, hz_last_1s, total, pub_t)."""
        with self._lock:
            now = time.perf_counter()
            while self._samples and now - self._samples[0] > 1.0:
                self._samples.popleft()
            return (
                list(self.blades),
                self.preview_bgr,
                self.status,
                self.infer_ms,
                len(self._samples),
                self.sample_count,
                self.pub_t,
            )
