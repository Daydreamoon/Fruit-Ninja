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

- [x] M0 环境探测（依赖 / 模型 / CPU 基准 116Hz ✅；摄像头绑定异常待硬件侧恢复，详见 docs/M0-环境探测报告.md）
- [ ] M1 渲染骨架（60fps 空场景 + HUD）
- [ ] M2 视觉线程接入（手点 + 迷你预览）
- [ ] M3 刀光轨迹（渐隐 + 插值）
- [ ] M4 玩法闭环（水果 / 计分 / 炸弹 / 状态机）
- [ ] M5 打磨（参数 / 音效 / 演示）

## 运行（开发期）

```
python -m venv .venv          # 需要 Python 3.12
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python main.py
```

首次运行会自动下载姿态模型到 `models/`（约 5.5 MB）。

## 里程碑自测

每个里程碑的程序都带 `--selftest` 模式：自动运行若干秒、打印指标后退出，用于回归测试。
