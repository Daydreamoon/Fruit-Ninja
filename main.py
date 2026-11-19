"""AirFruit entry point.

Interactive:      python main.py [--camera N] [--mute] [--record out.mp4]
Self-test:        python main.py --selftest [seconds] [--visible] [--record out.mp4]
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


def _value(args: list[str], flag: str) -> str | None:
    if flag in args:
        i = args.index(flag)
        if i + 1 < len(args):
            return args[i + 1]
    return None


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
    # audio stays enabled (so the sfx backend is exercised) but muted: CI-quiet.
    win = GameWindow(
        use_synthetic=selftest,
        camera_index=_camera_index(args),
        audio="--mute" not in args,
        record_path=_value(args, "--record"),
    )
    if selftest:
        win.audio.muted = True
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
            blades, _preview, status, infer_ms, vis_hz, samples, _pub_t = vis
            print(f"SELFTEST seconds={s['seconds']:.1f} ticks={s['ticks']} paints={s['paints']}")
            print(f"SELFTEST avg tick fps={s['avg_tick_fps']:.1f} avg paint fps={s['avg_paint_fps']:.1f}")
            print(f"SELFTEST vision status={status} hz={vis_hz} samples={samples} infer_ms={infer_ms:.1f}")
            g = s["game"]
            print(f"SELFTEST trail_points={s['trail_points']}")
            print(f"SELFTEST audio={s['audio']} muted={s['muted']}")
            if s["record"]:
                r = s["record"]
                print(f"SELFTEST record status={r['status']} frames={r['frames']} path={r['path']}")
            print(
                f"SELFTEST game spawned={g['spawned']} sliced={g['sliced']}"
                f" missed={g['missed']} bombs={g['bombs_sliced']} round={g['rounds_played']}"
                f" score={win.state.score}"
            )
            # spawn expectation scales with the run length: a 20 s run must see
            # a full wave or two, a 6 s smoke test only needs to prove the loop.
            spawn_needed = max(2, min(6, round(seconds / 4)))
            ok = (
                s["avg_paint_fps"] >= 55.0
                and vis_hz >= 20
                and s["trail_points"] >= 3
                and g["spawned"] >= spawn_needed
                and g["sliced"] >= 1
            )
            print(
                "SELFTEST "
                + ("PASS" if ok else "FAIL")
                + f" (render>=55fps, vision>=20Hz, trail>=3pts, spawn>={spawn_needed}, slice>=1)"
            )
            exit_code[0] = 0 if ok else 1
            app.quit()

        QTimer.singleShot(int(seconds * 1000), _finish)

    app.exec()

    # closeEvent (Qt6 quit() closes windows) usually already ran shutdown();
    # this is the belt-and-braces path for the interactive mode.
    win.shutdown()
    if selftest and win.recorder is not None:
        print(f"SELFTEST recorder {win.recorder.summary()}")
    return exit_code[0] if selftest else 0


if __name__ == "__main__":
    raise SystemExit(main())
