"""AirFruit entry point.

Interactive:  python main.py
Self-test:    python main.py --selftest [seconds]   (offscreen, prints metrics,
              exits 0 when average paint FPS >= 55)
"""
from __future__ import annotations

import os
import sys


def main() -> int:
    args = sys.argv[1:]
    selftest = "--selftest" in args
    seconds = 5.0
    if selftest:
        i = args.index("--selftest")
        if i + 1 < len(args):
            try:
                seconds = float(args[i + 1])
            except ValueError:
                pass
        # headless raster render: validates the loop without touching the screen;
        # pass --visible to run the timed selftest in a real on-screen window.
        if "--visible" not in args:
            os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    from game.window import GameWindow

    app = QApplication(sys.argv)
    win = GameWindow()
    win.show()
    win.start()

    if selftest:
        QTimer.singleShot(int(seconds * 1000), app.quit)

    app.exec()

    s = win.summary()
    print(f"SELFTEST seconds={s['seconds']:.1f} ticks={s['ticks']} paints={s['paints']}")
    print(f"SELFTEST avg tick fps={s['avg_tick_fps']:.1f} avg paint fps={s['avg_paint_fps']:.1f}")
    if selftest:
        ok = s["avg_paint_fps"] >= 55.0
        print(f"SELFTEST {'PASS' if ok else 'FAIL'} (require >=55fps)")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
