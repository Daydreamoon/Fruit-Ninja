"""Procedural sound effects (M5): synthesized in numpy, played by QSoundEffect.

No asset files: every effect is generated at import time as 16-bit mono WAV in
a temp dir. All of it is optional — if PySide6's multimedia module or the
platform audio backend is missing, `AudioBank` silently becomes a no-op so the
game never depends on sound to run (and the selftest stays silent).
"""
from __future__ import annotations

import math
import os
import tempfile
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 44100
# Written next to the package, not into a temp dir: on this machine TMP was not
# set for child processes, so mkdtemp() landed the WAVs in the working
# directory (and git happily committed 50 of them). A fixed project-local
# cache also lets repeated launches skip re-synthesis.
CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache" / "sfx"


def _write_wav(path: str, samples: np.ndarray) -> None:
    """Write float samples in [-1, 1] as 16-bit mono PCM."""
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())


def _env(n: int, attack: float = 0.005, decay: float = 0.25) -> np.ndarray:
    """Fast attack + exponential decay envelope."""
    t = np.arange(n) / SAMPLE_RATE
    a = np.clip(t / max(attack, 1e-6), 0.0, 1.0)
    return a * np.exp(-t / max(decay, 1e-6))


def _slice(name: str) -> np.ndarray:
    """Blade whoosh: filtered noise sweeping down, very short."""
    n = int(SAMPLE_RATE * 0.18)
    rng = np.random.default_rng(7)
    noise = rng.standard_normal(n)
    # crude one-pole lowpass whose cutoff falls over the effect
    out = np.empty(n)
    y = 0.0
    for i in range(n):
        k = 0.85 - 0.6 * (i / n)
        y = k * y + (1.0 - k) * noise[i]
        out[i] = y
    sweep = np.sin(2 * np.pi * np.cumsum(np.linspace(1400, 380, n)) / SAMPLE_RATE)
    return (0.5 * out + 0.25 * sweep) * _env(n, 0.004, 0.06)


def _splat(name: str) -> np.ndarray:
    """Juicy hit: short low thump + noise burst."""
    n = int(SAMPLE_RATE * 0.22)
    t = np.arange(n) / SAMPLE_RATE
    rng = np.random.default_rng(11)
    thump = np.sin(2 * np.pi * (170 * np.exp(-t * 9))) * np.exp(-t * 22)
    burst = rng.standard_normal(n) * np.exp(-t * 34) * 0.5
    return (0.75 * thump + burst) * _env(n, 0.002, 0.09)


def _boom(name: str) -> np.ndarray:
    """Bomb: low rumble + wideband crash, longer decay."""
    n = int(SAMPLE_RATE * 0.75)
    t = np.arange(n) / SAMPLE_RATE
    rng = np.random.default_rng(3)
    rumble = np.sin(2 * np.pi * (70 * np.exp(-t * 2.2))) * np.exp(-t * 3.5)
    crash = rng.standard_normal(n) * np.exp(-t * 5.0)
    body = np.cumsum(crash) / 60.0  # integrate -> brown-ish noise
    return (0.9 * rumble + 0.55 * crash + 0.5 * body) * _env(n, 0.002, 0.3)


def _spawn(name: str) -> np.ndarray:
    """Soft pop as an object leaves the bottom edge."""
    n = int(SAMPLE_RATE * 0.12)
    t = np.arange(n) / SAMPLE_RATE
    return np.sin(2 * np.pi * (300 + 500 * t / 0.12) * t) * _env(n, 0.003, 0.05) * 0.35


def _gameover(name: str) -> np.ndarray:
    """Descending two-tone game-over sting."""
    n = int(SAMPLE_RATE * 0.9)
    t = np.arange(n) / SAMPLE_RATE
    f = np.where(t < 0.35, 330.0, 220.0)
    tone = np.sin(2 * np.pi * np.cumsum(f) / SAMPLE_RATE)
    return tone * _env(n, 0.01, 0.45) * 0.55


_BUILDERS = {
    "slice": _slice,
    "splat": _splat,
    "boom": _boom,
    "spawn": _spawn,
    "gameover": _gameover,
}


class AudioBank:
    """Lazily built, best-effort sound bank.

    Usage: `bank.play("slice")`. Errors are swallowed by design: audio is
    polish, never a dependency. `enabled=False` forces a silent bank (selftest).
    """

    def __init__(self, enabled: bool = True, muted: bool = False) -> None:
        self.enabled = enabled
        self.muted = muted
        self.status = "off" if not enabled else "init"
        self._sounds: dict = {}
        if enabled:
            self._build()

    def _build(self) -> None:
        try:
            from PySide6.QtCore import QUrl
            from PySide6.QtMultimedia import QSoundEffect
        except Exception as exc:  # noqa: BLE001 - multimedia is optional
            self.status = f"unavailable ({type(exc).__name__})"
            self.enabled = False
            return
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            for name, builder in _BUILDERS.items():
                path = CACHE_DIR / f"{name}.wav"
                if not path.exists():
                    _write_wav(str(path), builder(name))
                eff = QSoundEffect()
                eff.setSource(QUrl.fromLocalFile(str(path)))
                eff.setVolume(0.55)
                self._sounds[name] = eff
            self.status = f"ready ({len(self._sounds)} sounds)"
        except Exception as exc:  # noqa: BLE001
            self.status = f"failed ({type(exc).__name__})"
            self.enabled = False
            self._sounds.clear()

    def play(self, name: str) -> None:
        if not self.enabled or self.muted:
            return
        eff = self._sounds.get(name)
        if eff is None:
            return
        try:
            # restart instead of layering: keeps rapid slicing from clipping
            eff.stop()
            eff.play()
        except Exception:  # noqa: BLE001 - never let audio break a frame
            pass
