# Fruit-Ninja · AirFruit 隔空切水果

数字图像处理课程设计：基于 MediaPipe 姿态识别的隔空切水果体感游戏。

对着摄像头挥手，屏幕上跟随手部画出刀光，砍掉飞上来的水果，避开炸弹。
渲染 PySide6（QTimer + paintEvent，60fps 目标）· 识别 MediaPipe Tasks（Pose Landmarker lite，GPU delegate 可选、CPU 兜底）· 采集 OpenCV。

## 架构（输入与游戏分开的两层）

```
视觉线程（QThread）：摄像头 → PoseLandmarker → 手部刀刃点 → 共享状态（QMutex）
渲染主线程（60fps）：QTimer(16ms) → dt 物理 → paintEvent（背景缓存 / 水果 / 刀光 / HUD）
```

识别与渲染解耦：采样率随摄像头走（约 30Hz），渲染独立跑 60fps，刀光在两次采样之间线性插值。

## 进度（对照计划书 v1）

- [x] M0 环境探测（依赖 / 模型 / CPU 基准 116Hz ✅ / 摄像头 MSMF 30.4fps @1280×720 ✅，详见 docs/M0-环境探测报告.md）
- [x] M1 渲染骨架（自测：离屏 62fps / 真屏 60.4fps 锁垂直同步 ✅）
- [x] M2 视觉线程接入（合成自测 62.5fps ✅；**真机复测：真实摄像头 30Hz + 推理 8.7ms + 渲染 62fps ✅**，详见 docs/M2-视觉接入测试.md）
- [x] M3 刀光轨迹（0.25s 渐隐 + 30Hz 采样间外推插值；离屏 62.5fps / 真屏 61.0fps PASS，详见 docs/M3-刀光轨迹测试.md）
- [x] M4 玩法闭环（水果 / 半片 / 果汁 / 计分 / 炸弹 GAME OVER / 自动重开；合成自测 62.5fps 全绿，炸弹路径已验证，详见 docs/M4-玩法闭环测试.md）
- [x] M5 打磨（程序化音效 5 个 / MP4 录屏 / 刀光圆头平滑 / 回合难度曲线；真屏录制 61.3fps + 视觉 30Hz，详见 docs/M5-打磨与录屏.md）

## 运行（开发期）

> ⚠️ 本机虚拟环境放在 **`D:\FruitNinjaEnv`**（中文目录之外）。原因：位于 `D:\数字图像处理\...` 下的解释器会被本机安全软件拦掉摄像头（黑帧/拒绝访问），同一套包放到中文路径外即恢复正常，详见 docs/M0-环境探测报告.md 第 6 条。代码本身与本问题无关。

```
# 一次性准备（在任意位置执行）
uv venv D:\FruitNinjaEnv --python <python3.12>
uv pip install -r requirements.txt --python D:\FruitNinjaEnv\Scripts\python.exe   # 必须用 uv：见 M0 报告

D:\FruitNinjaEnv\Scripts\python main.py                 # 真机：摄像头 + 挥手
D:\FruitNinjaEnv\Scripts\python main.py --mute          # 静音
D:\FruitNinjaEnv\Scripts\python main.py --selftest 20   # 离屏回归自测（合成视觉，退出码 0=PASS）
D:\FruitNinjaEnv\Scripts\python main.py --selftest 20 --camera-live --camera 0 --visible   # 真机链路自测（需人在镜头前挥手才有刀光）
D:\FruitNinjaEnv\Scripts\python main.py --selftest 25 --visible --record demo/airfruit-demo.mp4
D:\FruitNinjaEnv\Scripts\python vision\probes.py --camera   # 摄像头/后端体检
```

首次运行会自动下载姿态模型到 `models/`（约 5.5 MB）。**真机模式启动需要约 10 秒**（摄像头冷启动 3~10s + 模型 1.8s），期间画面中央会提示"正在打开摄像头… 请把手举到镜头前"，就绪后提示消失、水果才开始出现。录屏请用 `--visible`：离屏模式没有字体，HUD 会渲染成豆腐块。

## 里程碑自测

每个里程碑的程序都带 `--selftest` 模式：自动运行若干秒、打印指标后退出，用于回归测试。
