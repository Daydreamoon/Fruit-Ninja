"""AirFruit entry point.

Interactive:      python main.py [--camera N] [--mute] [--record out.mp4]
Self-test:        python main.py --selftest [seconds] [--visible] [--record out.mp4]
                  offscreen render + SYNTHETIC vision input; exits 0 when
                  render fps >= 55 and vision sample rate >= 20 Hz.
Live-camera test: python main.py --selftest 20 --camera-live --camera 0 --visible
                  same timed run, but vision comes from the real camera; the
                  gate then checks the pipeline (status ok, >=20 Hz), not hands.
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


# interpreter check -----------------------------------------------------------
# The app needs a Python with PySide6 + mediapipe + opencv + numpy. A stray
# interpreter (conda base, system python) fails deep inside an import with a
# message that says nothing about what to do -- so probe the imports up front
# and print the fix instead of a traceback.
REQUIRED_IMPORTS = ("PySide6.QtCore", "PySide6.QtWidgets", "cv2", "mediapipe", "numpy")
RECOMMENDED_PYTHON = r"D:\FruitNinjaEnv\Scripts\python.exe"


def _preflight() -> list[tuple[str, str]]:
    """Import-probe the hard requirements; return [(module, error), ...]."""
    broken: list[tuple[str, str]] = []
    for mod in REQUIRED_IMPORTS:
        try:
            __import__(mod)
        except Exception as exc:  # noqa: BLE001 - report, never raise
            broken.append((mod, f"{type(exc).__name__}: {exc}"))
    return broken


def _explain_broken_env(broken: list[tuple[str, str]]) -> None:
    print(f"[AirFruit] 这个 Python 解释器跑不了本项目：\n           {sys.executable}\n")
    for mod, err in broken:
        print(f"  - {mod:<20} {err}")
    print("\n项目依赖装在独立环境里，请改用：")
    print(f"  {RECOMMENDED_PYTHON} main.py")
    print("或直接双击项目根目录的 run.bat（会自动选中正确的解释器）。\n")
    if any(mod.startswith("PySide6") for mod, _ in broken):
        print("QtCore 报 DLL load failed / 找不到指定的程序，常见原因：")
        print("  Anaconda base 目录（如 D:\\Anaconda2\\）里自带旧版 msvcp140.dll (14.4x)，")
        print("  python.exe 同目录的 DLL 优先级高于 System32，Qt 6.12 需要 14.5x，")
        print("  于是加载 Qt6Core.dll 时报\"找不到指定的程序\"。本项目 env 没有这个问题。")
    if any(mod == "mediapipe" for mod, _ in broken):
        print("\nmediapipe 没有 cp313 轮子（官方只到 3.12），conda base 的 3.13 装不上，")
        print("这也是为什么项目用 Python 3.12 的独立环境。")


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

    broken = _preflight()
    if broken:
        _explain_broken_env(broken)
        return 3

    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    from game.window import GameWindow

    app = QApplication(sys.argv)
    # selftest defaults to synthetic vision (no camera needed, regression mode);
    # --camera-live keeps the timed run but feeds it the real camera instead.
    live_camera = "--camera-live" in args
    win = GameWindow(
        use_synthetic=selftest and not live_camera,
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
            if live_camera:
                print(f"SELFTEST blades_now={len(blades)} (hands visible to the camera)")
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
            if live_camera:
                # nobody guarantees hands in frame during a timed run, so the
                # live gate checks the pipeline (camera -> pose -> publish),
                # not the player's hand gestures.
                ok = (
                    s["avg_paint_fps"] >= 55.0
                    and vis_hz >= 20
                    and status == "ok"
                    and g["spawned"] >= spawn_needed
                )
                criteria = f"render>=55fps, vision>=20Hz status=ok, spawn>={spawn_needed} (live camera)"
            else:
                ok = (
                    s["avg_paint_fps"] >= 55.0
                    and vis_hz >= 20
                    and s["trail_points"] >= 3
                    and g["spawned"] >= spawn_needed
                    and g["sliced"] >= 1
                )
                criteria = f"render>=55fps, vision>=20Hz, trail>=3pts, spawn>={spawn_needed}, slice>=1"
            print("SELFTEST " + ("PASS" if ok else "FAIL") + f" ({criteria})")
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
