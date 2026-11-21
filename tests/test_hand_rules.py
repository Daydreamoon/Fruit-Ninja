"""Offline checks for the two rules that decide what a blade is.

These do not need a camera: the detection rules are pure functions of the pose
landmarks handed in, and the hand-mode selection is pure state. Run with the
project interpreter:

    D:\\FruitNinjaEnv\\Scripts\\python.exe tests\\test_hand_rules.py

Why this file exists: "two blades on screen although one hand is in front of the
camera" and "the second blade jitters on its own" are both rules, not physics,
so they can be pinned down without a person waving at a lens. Every assert below
came out of a real complaint during play.
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from game.window import GameWindow  # noqa: E402
from vision.shared import BladePoint, SharedVisionState  # noqa: E402
from vision.thread import VisionThread  # noqa: E402

DT = 1.0 / 30.0


def _landmarks(left_xy=None, right_xy=None, left_vis=0.9, right_vis=0.9, elbow_vis=0.9):
    """33 pose landmarks with only the arm joints meaningful."""
    lms = [SimpleNamespace(x=0.5, y=0.5, visibility=0.0) for _ in range(33)]

    def arm(ids, xy, vis, elbow_vis):
        wx, wy = xy
        for wrist, pinky, index, elbow in ids:
            lms[wrist] = SimpleNamespace(x=wx, y=wy, visibility=vis)
            lms[pinky] = SimpleNamespace(x=wx + 0.02, y=wy - 0.02, visibility=vis)
            lms[index] = SimpleNamespace(x=wx - 0.02, y=wy - 0.02, visibility=vis)
            lms[elbow] = SimpleNamespace(x=wx, y=wy + 0.25, visibility=elbow_vis)

    if left_xy is not None:
        arm(((15, 17, 19, 13),), left_xy, left_vis, elbow_vis)
    if right_xy is not None:
        arm(((16, 18, 20, 14),), right_xy, right_vis, elbow_vis)
    return lms


def _thread() -> VisionThread:
    return VisionThread(SharedVisionState(), synthetic=True)


def test_palm_centre_is_the_blade() -> None:
    t = _thread()
    blades = t._extract_hands(_landmarks(right_xy=(0.60, 0.40)), 0.0)
    assert len(blades) == 1, blades
    b = blades[0]
    # palm = mean of wrist (0.60, .40), pinky (.62, .38), index (.58, .38)
    assert abs(b.x - 0.60) < 1e-6 and abs(b.y - (0.40 + 0.38 + 0.38) / 3) < 1e-6, (b.x, b.y)
    print("ok  palm centre is the blade")


def test_resting_hand_is_not_a_blade() -> None:
    """A hand that sits still for IDLE_S counts as put down."""
    t = _thread()
    t0 = 0.0
    assert t._extract_hands(_landmarks(right_xy=(0.60, 0.40)), t0), "first sample must track"
    t1 = t0 + 2.6
    blades = t._extract_hands(_landmarks(right_xy=(0.60, 0.40)), t1)
    assert blades == [], f"a motionless hand should have been dropped: {blades}"
    print("ok  motionless hand is dropped")


def test_lowered_hand_is_not_a_blade() -> None:
    """Wrist below the elbow = arm hanging by the chair, not playing."""
    t = _thread()
    lms = _landmarks(right_xy=(0.60, 0.40))
    lms[16] = SimpleNamespace(x=0.60, y=0.80, visibility=0.9)  # wrist below elbow (0.65)
    lms[18] = SimpleNamespace(x=0.62, y=0.78, visibility=0.9)
    lms[20] = SimpleNamespace(x=0.58, y=0.78, visibility=0.9)
    assert t._extract_hands(lms, 0.0) == []
    print("ok  lowered hand is ignored")


def test_teleporting_hand_is_held_then_dropped() -> None:
    """A wrist estimate that jumps around cannot be a hand: hold, then drop."""
    t = _thread()
    now = 0.0
    first = t._extract_hands(_landmarks(right_xy=(0.50, 0.40)), now)
    assert len(first) == 1 and abs(first[0].x - 0.50) < 1e-6, first
    for i in range(5):  # 5 impossible jumps, each ~0.4 of the frame
        now += DT
        blades = t._extract_hands(_landmarks(right_xy=(0.95 if i % 2 else 0.05, 0.40)), now)
        assert blades and abs(blades[0].x - 0.50) < 0.02, f"jump {i} followed: {blades}"
    for _ in range(25):  # keep teleporting past TELEPORT_DROP_S
        now += DT
        t._extract_hands(_landmarks(right_xy=(0.95, 0.40)), now)
    assert t._extract_hands(_landmarks(right_xy=(0.95, 0.40)), now + DT) == []
    print("ok  teleporting hand is held then dropped")


def test_fast_swipe_is_followed() -> None:
    """A real slice does exceed MAX_JUMP -- but it keeps its direction."""
    t = _thread()
    now = 0.0
    x = 0.10
    t._extract_hands(_landmarks(right_xy=(x, 0.40)), now)
    for dx in (0.05, 0.09, 0.12, 0.30, 0.30):  # ramping up into a swipe
        now += DT
        x += dx
        blades = t._extract_hands(_landmarks(right_xy=(x, 0.40)), now)
        assert len(blades) == 1, f"swipe sample dropped at x={x:.2f}: {blades}"
    assert blades[0].x > 0.55, f"blade should have followed the swipe: {blades[0].x}"
    print("ok  fast same-direction swipe is followed")


def test_hand_modes() -> None:
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])  # noqa: F841  (must outlive the widget)
    win = GameWindow(use_synthetic=True, audio=False)
    try:
        left = BladePoint(x=0.4, y=0.4, visibility=0.90, side="left")
        right = BladePoint(x=0.6, y=0.4, visibility=0.70, side="right")

        win.hand_mode = "single"
        win._primary = None
        picked = win._select_blades([left, right], 0.0)
        assert len(picked) == 1 and picked[0].side == "left", picked

        # stickiness: a slightly better rival must not steal the blade
        better = BladePoint(x=0.6, y=0.4, visibility=0.95, side="right")
        for i in range(12):
            assert win._select_blades([left, better], i * DT)[0].side == "left"

        win.hand_mode = "both"
        assert len(win._select_blades([left, right], 1.0)) == 2

        win.hand_mode = "left"
        assert [b.side for b in win._select_blades([left, right], 2.0)] == ["left"]
        win.hand_mode = "right"
        assert [b.side for b in win._select_blades([left, right], 2.0)] == ["right"]
        print("ok  hand modes: single picks one and stays put")
    finally:
        win.deleteLater()


if __name__ == "__main__":
    test_palm_centre_is_the_blade()
    test_resting_hand_is_not_a_blade()
    test_lowered_hand_is_not_a_blade()
    test_teleporting_hand_is_held_then_dropped()
    test_fast_swipe_is_followed()
    test_hand_modes()
    print("ALL HAND RULES OK")
