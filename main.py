"""AirFruit entry point.

Interactive:      python main.py [--camera N]
Self-test:        python main.py --selftest [seconds] [--visible]
                  offscreen render + SYNTHETIC vision input; exits 0 when
                  render fps >= 55 and vision sample rate >= 20 Hz.
"""
from __future__ import annotations

import os
import sys


def _camera_index(args: list[str]) -> int:
    if "--camera" in args:
        i = args.index("--camera")
        if i + 1 < len(args):
            try:
                return int(args[i + 1])
            except ValueError:
                pass
    return 0


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
    # selftest uses synthetic vision input: no camera needed (regression mode)
    win = GameWindow(use_synthetic=selftest, camera_index=_camera_index(args))
    win.show()
    win.start()

    exit_code = [0]

    if selftest:
        def _finish() -> None:
            # MUST sample before app.quit(): Qt6 quit() closes all windows
            # first, which triggers closeEvent -> shutdown() and kills the
            # vision thread -- sampling after exec() would be a post-mortem
            # (status "stopped", empty rate window) even when vision ran fine.
            vis = win.vision_shared.snapshot()
            s = win.summary()
            blades, _preview, status, infer_ms, vis_hz, samples = vis
            print(f"SELFTEST seconds={s['seconds']:.1f} ticks={s['ticks']} paints={s['paints']}")
            print(f"SELFTEST avg tick fps={s['avg_tick_fps']:.1f} avg paint fps={s['avg_paint_fps']:.1f}")
            print(f"SELFTEST vision status={status} hz={vis_hz} samples={samples} infer_ms={infer_ms:.1f}")
            ok = s["avg_paint_fps"] >= 55.0 and vis_hz >= 20
            print(f"SELFTEST {'PASS' if ok else 'FAIL'} (require render>=55fps, vision>=20Hz)")
            exit_code[0] = 0 if ok else 1
            app.quit()

        QTimer.singleShot(int(seconds * 1000), _finish)

    app.exec()

    # closeEvent (Qt6 quit() closes windows) usually already ran shutdown();
    # this is the belt-and-braces path for the interactive mode.
    win.shutdown()
    return exit_code[0] if selftest else 0


if __name__ == "__main__":
    raise SystemExit(main())
