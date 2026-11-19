"""Fruit entities, spawn table and segment-circle collision (M4).

All physics in game pixels, dt-driven (plan: never frame-counted).
The plan's fruit table: orange/kiwi 2 pts @1500 px/s, pineapple 3 @1000,
watermelon 4 @1000, bomb 875; bombs appear from round 3; a missed fruit
costs 2x its value. Placeholders are code-drawn (plan decision: no art
assets in v1).
"""
from __future__ import annotations

import math
import random

GAME_W, GAME_H = 1280, 720  # duplicated from window to keep this module standalone
GRAVITY = 1800.0  # px/s^2 — tuned so 1500 px/s launches peak near mid-screen
MISS_PENALTY = 2  # missed fruit subtracts 2x its value (plan)

# name, radius, value, launch speed px/s, core color, rim color
FRUIT_TABLE = [
    ("orange", 34, 2, 1500, (255, 150, 50), (255, 200, 80)),
    ("kiwi", 30, 2, 1500, (130, 180, 70), (180, 220, 110)),
    ("pineapple", 42, 3, 1000, (235, 200, 70), (205, 155, 45)),
    ("watermelon", 50, 4, 1000, (70, 165, 70), (200, 70, 70)),
]
BOMB_TABLE = ("bomb", 36, 0, 875, (45, 45, 52), (220, 60, 40))
BOMB_FROM_ROUND = 3
BOMB_CHANCE = 1 / 8  # per spawned object once bombs are allowed

# --- M4 difficulty ramp (M5 tuning) -------------------------------------
# Each round is a little faster and, once bombs are legal, a little meaner.
# Kept mild: at round 8 launches are +42% and bombs +5 points of probability.
ROUND_SPEED_STEP = 0.06
ROUND_SPEED_CAP = 1.6
ROUND_BOMB_STEP = 0.01
ROUND_BOMB_CAP = 0.22


def _round_scale(round_no: int) -> float:
    return min(ROUND_SPEED_CAP, 1.0 + ROUND_SPEED_STEP * max(0, round_no - 1))


def _bomb_chance(round_no: int) -> float:
    extra = ROUND_BOMB_STEP * max(0, round_no - BOMB_FROM_ROUND)
    return min(ROUND_BOMB_CAP, BOMB_CHANCE + extra)


class Fruit:
    """A whole fruit (or bomb) flying ballistically until sliced or missed."""

    __slots__ = ("name", "x", "y", "vx", "vy", "radius", "value", "core", "rim", "bomb", "sliced")

    def __init__(self, name, x, y, vx, vy, radius, value, core, rim, bomb: bool) -> None:
        self.name = name
        self.x, self.y = x, y
        self.vx, self.vy = vx, vy
        self.radius = radius
        self.value = value
        self.core, self.rim = core, rim
        self.bomb = bomb
        self.sliced = False

    def update(self, dt: float) -> None:
        self.vy += GRAVITY * dt
        self.x += self.vx * dt
        self.y += self.vy * dt

    def below_screen(self) -> bool:
        return self.y - self.radius > GAME_H

    def slice(self) -> None:
        self.sliced = True


class HalfFruit:
    """One half of a sliced fruit: keeps flying, spinning, fading."""

    __slots__ = ("x", "y", "vx", "vy", "radius", "color", "spin", "angle", "life", "max_life")

    def __init__(self, x, y, vx, vy, radius, color, angle0: float) -> None:
        self.x, self.y = x, y
        self.vx, self.vy = vx, vy
        self.radius = radius
        self.color = color
        self.spin = random.uniform(-7.0, 7.0)
        self.angle = angle0
        self.max_life = 1.2
        self.life = self.max_life

    def update(self, dt: float) -> bool:
        """Returns False when expired."""
        self.vy += GRAVITY * dt
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.angle += self.spin * dt
        self.life -= dt
        return self.life > 0


class Particle:
    """Juice splash dot."""

    __slots__ = ("x", "y", "vx", "vy", "color", "life")

    def __init__(self, x, y, vx, vy, color) -> None:
        self.x, self.y = x, y
        self.vx, self.vy = vx, vy
        self.color = color
        self.life = 0.45

    def update(self, dt: float) -> bool:
        self.vy += GRAVITY * 0.35 * dt  # juice is light: falls slower
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.life -= dt
        return self.life > 0


class FloatText:
    """Score popup (+2 / -4 / GAME OVER hint)."""

    __slots__ = ("x", "y", "text", "color", "life")

    def __init__(self, x, y, text, color=(255, 255, 255)) -> None:
        self.x, self.y = x, y
        self.text = text
        self.color = color
        self.life = 0.9

    def update(self, dt: float) -> bool:
        self.y -= 60 * dt
        self.life -= dt
        return self.life > 0


def segment_hits_circle(
    x0: float, y0: float, x1: float, y1: float, cx: float, cy: float, r: float
) -> bool:
    """Plan's collision rule: distance from circle center to blade SEGMENT <= r.

    Classic point-to-segment distance with the degenerate ends handled.
    """
    dx, dy = x1 - x0, y1 - y0
    seg2 = dx * dx + dy * dy
    if seg2 <= 1e-12:
        return math.hypot(cx - x0, cy - y0) <= r
    t = ((cx - x0) * dx + (cy - y0) * dy) / seg2
    t = max(0.0, min(1.0, t))
    px, py = x0 + t * dx, y0 + t * dy
    return math.hypot(cx - px, cy - py) <= r


def spawn_object(round_no: int):
    """Build one launch: a random fruit, or a bomb (round >= BOMB_FROM_ROUND).

    Launch point sits below the bottom edge; angle is mostly up with spread.
    Returns the Fruit (caller sets exact launch timing).
    """
    if round_no >= BOMB_FROM_ROUND and random.random() < _bomb_chance(round_no):
        name, radius, value, speed, core, rim = BOMB_TABLE
        bomb = True
    else:
        name, radius, value, speed, core, rim = random.choice(FRUIT_TABLE)
        bomb = False
    x = random.uniform(0.32, 0.68) * GAME_W  # corridor where the demo blade orbits
    y = GAME_H + radius + 6
    # aim up with lateral spread; faster objects fly straighter (less spread)
    angle = math.radians(random.uniform(78, 102))
    speed = speed * random.uniform(1.0, 1.5) * _round_scale(round_no)
    vx = speed * math.cos(angle) * random.choice((-1, 1))
    vy = -speed * math.sin(angle)
    return Fruit(name, x, y, vx, vy, radius, value, core, rim, bomb)
