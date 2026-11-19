"""Game state machine: rounds, spawning, slicing, scoring, game over (M4).

Pure logic, no Qt: the window calls update(dt) per tick, feeds blade
segments, and renders from the public lists. Round flow: throw a wave of
2..5 objects (fruits + maybe bombs from round 3), wait until all fell or
were sliced, brief cooldown, next round. Slicing a bomb ends the run.
"""
from __future__ import annotations

import random
import time

from game.fruits import (
    BOMB_FROM_ROUND,
    GAME_H,
    MISS_PENALTY,
    FloatText,
    Fruit,
    HalfFruit,
    Particle,
    spawn_object,
)

WAVE_SIZE = (2, 5)  # objects per round (reference project range)
WAVE_STAGGER = (0.0, 0.9)  # intra-wave launch delay seconds
ROUND_COOLDOWN = 2.0  # s between rounds (reference ROUND_COOLDOWN)
GAME_OVER_RESTART = 4.0  # auto-restart delay (demo/selftest keeps exercising)


class Stats:
    """Cumulative (never reset by restart) counters for the selftest."""

    def __init__(self) -> None:
        self.spawned = 0
        self.sliced = 0
        self.missed = 0
        self.bombs_sliced = 0
        self.rounds_played = 0

    def as_dict(self) -> dict:
        return {
            "spawned": self.spawned,
            "sliced": self.sliced,
            "missed": self.missed,
            "bombs_sliced": self.bombs_sliced,
            "rounds_played": self.rounds_played,
        }


class GameState:
    def __init__(self) -> None:
        self.stats = Stats()
        self.score = 0
        self.round = 0
        self.game_over = False
        self._restart_in = 0.0
        self._cooldown = 1.0  # small breath before round 1
        self._queue: list[tuple[float, Fruit]] = []  # (launch_at, fruit) pending
        self.fruits: list[Fruit] = []
        self.halves: list[HalfFruit] = []
        self.particles: list[Particle] = []
        self.texts: list[FloatText] = []
        self.events: list[str] = []  # drained by the window for sound (M5)

    # ---- queries for the renderer ----

    def hud(self) -> str:
        if self.game_over:
            return f"GAME OVER · score {self.score} · restarting…"
        return f"round {self.round} · score {self.score}"

    # ---- driving ----

    def update(self, dt: float) -> None:
        if self.game_over:
            self._restart_in -= dt
            self._decay_fx(dt)
            if self._restart_in <= 0:
                self.restart()
            return

        # launch due objects from the pending wave queue
        if self._queue:
            for i, (launch_at, fruit) in enumerate(self._queue):
                if launch_at <= 0.0:
                    self.fruits.append(fruit)
                    self.stats.spawned += 1
                    if not fruit.bomb:
                        self.events.append("spawn")
            self._queue = [(at - dt, f) for at, f in self._queue if at > 0.0]

        # physics + miss accounting
        alive: list[Fruit] = []
        for f in self.fruits:
            f.update(dt)
            if f.sliced:
                continue
            if f.below_screen():
                if not f.bomb:  # a fallen bomb is a dodged bomb: no penalty
                    self.score -= MISS_PENALTY * f.value
                    self.stats.missed += 1
                    self.texts.append(FloatText(f.x, GAME_H - 40, f"-{MISS_PENALTY * f.value}", (255, 90, 90)))
                continue
            alive.append(f)
        self.fruits = alive

        self._decay_fx(dt)

        # wave finished -> cooldown -> next wave
        if not self._queue and not self.fruits:
            self._cooldown -= dt
            if self._cooldown <= 0:
                self._next_wave()

    def _next_wave(self) -> None:
        self.round += 1
        self.stats.rounds_played += 1
        n = random.randint(*WAVE_SIZE)
        base = random.uniform(*WAVE_STAGGER)
        self._queue = []
        for i in range(n):
            delay = base + i * random.uniform(0.12, 0.4)
            self._queue.append((delay, spawn_object(self.round)))
        self._cooldown = ROUND_COOLDOWN

    def restart(self) -> None:
        self.score = 0
        self.round = 0
        self.game_over = False
        self._cooldown = 0.8
        self._queue = []
        self.fruits = []
        # fx lists keep decaying naturally

    # ---- blade input ----

    def collide_blade(self, segments: list[tuple[float, float, float, float]]) -> None:
        """segments: (x0, y0, x1, y1) blade pieces in the recent past."""
        if self.game_over or not segments or not self.fruits:
            return
        from game.fruits import segment_hits_circle

        hit: list[tuple[Fruit, tuple[float, float, float, float]]] = []
        for f in self.fruits:
            for x0, y0, x1, y1 in segments:
                if segment_hits_circle(x0, y0, x1, y1, f.x, f.y, f.radius):
                    hit.append((f, (x0, y0, x1, y1)))
                    break
        for f, seg in hit:
            if f.bomb:
                self._explode_bomb(f)
            else:
                self._slice_fruit(f, seg)

    def _slice_fruit(self, f: Fruit, seg: tuple[float, float, float, float]) -> None:
        x0, y0, x1, y1 = seg
        dx, dy = x1 - x0, y1 - y0
        length = max((dx * dx + dy * dy) ** 0.5, 1e-6)
        nx, ny = -dy / length, dx / length  # perpendicular to the blade
        kick = 260.0
        f.slice()
        self.score += f.value
        self.stats.sliced += 1
        c = f.core
        self.halves.append(HalfFruit(f.x, f.y, f.vx + nx * kick, f.vy + ny * kick - 120, f.radius * 0.8, c, 0.0))
        self.halves.append(HalfFruit(f.x, f.y, f.vx - nx * kick, f.vy - ny * kick - 120, f.radius * 0.8, c, 3.1415926))
        for _ in range(10):
            self.particles.append(
                Particle(
                    f.x + random.uniform(-8, 8),
                    f.y + random.uniform(-8, 8),
                    f.vx * 0.3 + random.uniform(-220, 220),
                    f.vy * 0.3 + random.uniform(-260, 60),
                    c,
                )
            )
        self.texts.append(FloatText(f.x, f.y - f.radius, f"+{f.value}", (255, 230, 120)))
        self.events.append("splat")

    def _explode_bomb(self, f: Fruit) -> None:
        f.slice()
        self.stats.bombs_sliced += 1
        self.game_over = True
        self._restart_in = GAME_OVER_RESTART
        for _ in range(26):
            self.particles.append(
                Particle(
                    f.x + random.uniform(-10, 10),
                    f.y + random.uniform(-10, 10),
                    random.uniform(-420, 420),
                    random.uniform(-420, 220),
                    (255, 140, 60),
                )
            )
        self.texts.append(FloatText(f.x, f.y - f.radius, "BOOM", (255, 90, 60)))
        self.events.append("boom")
        self.events.append("gameover")

    def _decay_fx(self, dt: float) -> None:
        self.halves = [h for h in self.halves if h.update(dt)]
        self.particles = [q for q in self.particles if q.update(dt)]
        self.texts = [t for t in self.texts if t.update(dt)]

    def collision_selfcheck(self) -> bool:
        """Offline sanity of the plan's point-to-segment rule (selftest)."""
        from game.fruits import segment_hits_circle

        assert segment_hits_circle(100, 100, 200, 100, 150, 90, 20) is True  # center near
        assert segment_hits_circle(100, 100, 200, 100, 150, 140, 20) is True  # endpoint zone
        assert segment_hits_circle(100, 100, 200, 100, 150, 300, 20) is False  # far away
        assert segment_hits_circle(150, 100, 150, 100, 150, 90, 20) is True  # degenerate seg
        return True
