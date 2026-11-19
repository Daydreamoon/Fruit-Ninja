"""Frame recorder (M5): grabs the widget each tick and writes an MP4.

Uses cv2.VideoWriter (mp4v). Recording is off by default and never affects
gameplay timing: `capture()` is a no-op until `start()` is called, and any
encoder failure permanently disables it instead of raising inside a paint tick.
"""
from __future__ import annotations

import time

import cv2
import numpy as np
from PySide6.QtGui import QImage


class Recorder:
    """Buffers widget grabs and writes them at a fixed nominal fps."""

    def __init__(self, path: str, fps: float = 60.0) -> None:
        self.path = path
        self.fps = float(fps)
        self.status = "idle"
        self.frames = 0
        self._writer: cv2.VideoWriter | None = None
        self._size: tuple[int, int] | None = None
        self._t0 = 0.0

    def start(self, size: tuple[int, int]) -> None:
        w, h = size
        if w <= 0 or h <= 0:
            self.status = "bad size"
            return
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(self.path, fourcc, self.fps, (w, h))
        if not writer.isOpened():
            self.status = "open failed"
            return
        self._writer = writer
        self._size = (w, h)
        self._t0 = time.perf_counter()
        self.status = "recording"
        self.frames = 0

    def capture(self, image: QImage) -> None:
        """Append one frame; silently stops recording on any conversion error."""
        if self._writer is None:
            return
        try:
            if image.format() != QImage.Format.Format_RGB888:
                image = image.convertToFormat(QImage.Format.Format_RGB888)
            w, h = image.width(), image.height()
            if self._size is not None and (w, h) != self._size:
                image = image.scaled(self._size[0], self._size[1])
                w, h = self._size
            ptr = image.constBits()
            arr = np.frombuffer(ptr, dtype=np.uint8, count=image.sizeInBytes())
            arr = arr.reshape(h, image.bytesPerLine() // 3, 3)[:, :w, :]
            self._writer.write(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
            self.frames += 1
        except Exception as exc:  # noqa: BLE001 - recording must never break a frame
            self.status = f"error ({type(exc).__name__})"
            self.stop()

    def stop(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            if self.status == "recording":
                self.status = f"done ({self.frames} frames)"

    def summary(self) -> dict:
        return {
            "status": self.status,
            "frames": self.frames,
            "fps": self.fps,
            "path": self.path,
        }
