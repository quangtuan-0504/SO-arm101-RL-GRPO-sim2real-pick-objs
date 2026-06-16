"""Vòng lặp điều khiển end-to-end bằng VLM (Qwen-VL).

Chu trình mỗi bước:
    1. Render 2 camera (wrist_cam, stand_cam) offscreen -> data URL.
    2. Dựng messages (prompt + 2 ảnh) và gọi Qwen-VL.
    3. Parse JSON -> góc mục tiêu 6 khớp.
    4. set_ctrl (tự clip trong giới hạn) cho từng khớp.
    5. Bước mô phỏng để cánh tay tiến tới góc mục tiêu.
    6. Lặp đến khi model báo done hoặc đạt max_steps.

Chạy thử KHÔNG cần API (mock):
    python -m vlm.control_loop --target cube_red --mock

Chạy thật với Qwen-VL (qwen3.6-flash, DashScope compatible-mode):
    # API key đọc từ .env (QWEN_API_KEY=...) hoặc biến môi trường
    python -m vlm.control_loop --target cube_red
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import yaml

DEFAULT_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3.6-flash"


def load_dotenv(path: Path):
    """Nạp biến môi trường từ file .env (định dạng KEY=value), không cần lib."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        # Không ghi đè biến đã có sẵn trong môi trường
        os.environ.setdefault(key, val)

from sim.scene_builder import JOINT_NAMES, ROOT
from sim.simulation import Simulation
from vlm.prompt_builder import build_messages
from vlm.render_utils import CameraRenderer, rgb_to_data_url, save_png
from vlm.qwen_client import QwenVLClient, MockQwenClient
from vlm.worker import VLMWorker


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_prediction(sim, pred: dict) -> dict:
    """Áp góc dự đoán vào sim (clip trong giới hạn). Trả về góc đã áp dụng."""
    angles = pred.get("joint_angles_rad", {}) or {}
    applied = {}
    for name in JOINT_NAMES:
        if name in angles:
            try:
                val = float(angles[name])
            except (TypeError, ValueError):
                continue
            sim.set_ctrl(name, val)          # set_ctrl tự clip vào ctrlrange
            applied[name] = sim.get_ctrl(name)
    return applied


def run_loop(config: dict, target_desc: str, client, *,
             max_steps: int = 12, settle_seconds: float = 0.6,
             img_w: int = 480, img_h: int = 360, save_debug: bool = True):
    """Chạy vòng lặp điều khiển. Trả về True nếu model báo done."""
    sim = Simulation(config)
    renderer = CameraRenderer(sim.model, width=img_w, height=img_h)

    debug_dir = ROOT / "vlm_runs" / time.strftime("%Y%m%d_%H%M%S")
    if save_debug:
        debug_dir.mkdir(parents=True, exist_ok=True)

    settle_steps = max(1, round(settle_seconds / sim.dt))
    done = False
    try:
        for step in range(1, max_steps + 1):
            wrist_rgb = renderer.render(sim.data, "wrist_cam")
            stand_rgb = renderer.render(sim.data, "stand_cam")

            if save_debug:
                save_png(debug_dir / f"step{step:02d}_wrist.png", wrist_rgb)
                save_png(debug_dir / f"step{step:02d}_stand.png", stand_rgb)

            messages = build_messages(
                sim, target_desc,
                rgb_to_data_url(wrist_rgb), rgb_to_data_url(stand_rgb))

            try:
                pred = client.predict(messages)
            except Exception as e:
                print(f"[B{step}] Lỗi gọi/parse model: {e}")
                break

            applied = apply_prediction(sim, pred)
            phase = pred.get("phase", "?")
            grip = pred.get("gripper_action", "?")
            done = bool(pred.get("done", False))
            conf = pred.get("confidence", "?")
            reason = str(pred.get("reasoning", ""))[:80]

            print(f"[B{step}] phase={phase} grip={grip} done={done} "
                  f"conf={conf} | {reason}")
            print(f"      goc ap dung: "
                  + ", ".join(f"{k}={v:+.2f}" for k, v in applied.items()))

            sim.step(settle_steps)  # để cánh tay tiến tới góc mục tiêu

            if done:
                print(f"[B{step}] Model bao HOAN THANH.")
                break
    finally:
        renderer.close()

    if save_debug:
        print(f"Anh debug tung buoc luu tai: {debug_dir}")
    return done


class VLMDriver:
    """Điều phối VLM chạy SONG SONG với cửa sổ hiển thị (gọi mỗi khung GLFW).

    Main thread (qua ``on_frame``) render ảnh và áp góc; worker nền lo phần gọi
    mạng. Chỉ giữ một yêu cầu suy luận tại một thời điểm; sau khi áp kết quả thì
    chờ ``settle_seconds`` để cánh tay tiến tới góc mục tiêu rồi mới gửi ảnh mới.
    """

    def __init__(self, worker: VLMWorker, *, settle_seconds: float = 0.6,
                 img_w: int = 480, img_h: int = 360, max_steps: int = 0,
                 save_debug: bool = False, debug_dir: Path | None = None):
        self.worker = worker
        self.settle_seconds = settle_seconds
        self.img_w = img_w
        self.img_h = img_h
        self.max_steps = max_steps  # 0 = không giới hạn
        self.save_debug = save_debug
        self.debug_dir = debug_dir

        self._renderer: CameraRenderer | None = None
        self._next_submit_time = 0.0
        self._step_idx = 0
        self._limit_announced = False

    def on_frame(self, app):
        # Tạo renderer offscreen sau khi GL context của GLFW đã sẵn sàng.
        if self._renderer is None:
            self._renderer = CameraRenderer(app.model, self.img_w, self.img_h)

        # 1) Áp kết quả mới nếu worker vừa trả về.
        pred = self.worker.poll()
        if pred is not None:
            if "error" in pred:
                print(f"[VLM] Loi worker: {pred['error']}")
            else:
                applied = apply_prediction(app.sim, pred)
                phase = pred.get("phase", "?")
                grip = pred.get("gripper_action", "?")
                print(f"[VLM] phase={phase} grip={grip} "
                      f"done={pred.get('done')} | "
                      + ", ".join(f"{k}={v:+.2f}" for k, v in applied.items()))
            self._next_submit_time = time.time() + self.settle_seconds

        # 2) Gửi ảnh mới nếu rảnh, đã đến hạn và chưa hoàn thành.
        now = time.time()
        reached_limit = self.max_steps and self._step_idx >= self.max_steps
        if reached_limit:
            if not self._limit_announced and not self.worker.busy:
                print(f"[VLM] Da dat max_steps={self.max_steps}, dung suy luan. "
                      "Cua so van mo - nhan ESC de thoat.")
                self._limit_announced = True
            return
        if (not self.worker.busy and not self.worker.done
                and now >= self._next_submit_time):
            wrist = self._renderer.render(app.data, "wrist_cam")
            stand = self._renderer.render(app.data, "stand_cam")
            if self.worker.submit(wrist, stand):
                self._step_idx += 1
                if self.save_debug and self.debug_dir is not None:
                    i = self._step_idx
                    save_png(self.debug_dir / f"step{i:02d}_wrist.png", wrist)
                    save_png(self.debug_dir / f"step{i:02d}_stand.png", stand)


def run_visual(config: dict, target_desc: str, client, *,
               settle_seconds: float = 0.6, img_w: int = 480, img_h: int = 360,
               max_steps: int = 0, save_debug: bool = True):
    """Mở 3 cửa sổ camera và chạy vòng lặp VLM song song theo thời gian thực."""
    from sim.viewer3 import ThreeViewApp  # import trễ: chỉ cần khi --view

    sim = Simulation(config)
    app = ThreeViewApp(sim, config)

    worker = VLMWorker(sim, target_desc, client)
    worker.start()

    debug_dir = None
    if save_debug:
        debug_dir = ROOT / "vlm_runs" / time.strftime("%Y%m%d_%H%M%S")
        debug_dir.mkdir(parents=True, exist_ok=True)

    driver = VLMDriver(worker, settle_seconds=settle_seconds,
                       img_w=img_w, img_h=img_h, max_steps=max_steps,
                       save_debug=save_debug, debug_dir=debug_dir)

    print("[VLM] Mo cua so hien thi. VLM chay nen; nhan ESC de thoat.")
    print("      (Ban van dieu khien tay bang ban phim nhu binh thuong.)")
    try:
        app.run(on_frame=driver.on_frame)
    finally:
        worker.stop()
    if save_debug:
        print(f"Anh debug tung buoc luu tai: {debug_dir}")
    return worker.done


def build_client(args, sim_for_mock):
    if args.mock:
        print("[VLM] Dung MockQwenClient (khong goi mang).")
        return MockQwenClient(sim_for_mock, done_after=args.mock_done_after)

    api_key = (args.api_key or os.environ.get("QWEN_API_KEY")
               or os.environ.get("DASHSCOPE_API_KEY") or "")
    base_url = args.base_url or os.environ.get("QWEN_BASE_URL", DEFAULT_BASE_URL)
    if not api_key:
        sys.exit("Thieu API key. Dat QWEN_API_KEY trong .env / bien moi truong, "
                 "hoac --api-key, hoac dung --mock de chay thu.")
    print(f"[VLM] Goi {args.model} @ {base_url} "
          f"(thinking={'on' if not args.no_thinking else 'off'})")
    return QwenVLClient(api_key=api_key, base_url=base_url, model=args.model,
                        enable_thinking=not args.no_thinking,
                        verbose=args.verbose)


def main():
    p = argparse.ArgumentParser(description="Vong lap dieu khien SO-101 bang VLM")
    p.add_argument("--config", default=str(ROOT / "config.yaml"))
    p.add_argument("--target", default="khoi lap phuong mau do (cube_red)",
                   help="Mo ta vat the can gap")
    p.add_argument("--max-steps", type=int, default=12,
                   help="So lan inference toi da. Voi --view, dat 0 = khong gioi han")
    p.add_argument("--settle-seconds", type=float, default=0.6)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--api-key", default=None)
    p.add_argument("--base-url", default=None)
    p.add_argument("--no-thinking", action="store_true",
                   help="Tat che do deep-thinking (goi khong streaming)")
    p.add_argument("--verbose", action="store_true",
                   help="In qua trinh suy luan (reasoning) cua model")
    p.add_argument("--mock", action="store_true",
                   help="Chay thu khong goi mang (client gia lap)")
    p.add_argument("--mock-done-after", type=int, default=3)
    p.add_argument("--no-debug-img", action="store_true")
    p.add_argument("--view", action="store_true",
                   help="Mo cua so hien thi (ThreeViewApp) chay song song")
    args = p.parse_args()

    load_dotenv(ROOT / ".env")
    config = load_config(args.config)
    # Sim tạm chỉ để mock biết home_pose/ctrlrange (run_loop tạo sim riêng).
    sim_tmp = Simulation(config)
    client = build_client(args, sim_tmp)

    if args.view:
        ok = run_visual(config, args.target, client,
                        settle_seconds=args.settle_seconds,
                        max_steps=args.max_steps,
                        save_debug=not args.no_debug_img)
    else:
        ok = run_loop(config, args.target, client,
                      max_steps=args.max_steps,
                      settle_seconds=args.settle_seconds,
                      save_debug=not args.no_debug_img)
    print("KET QUA:", "DONE" if ok else "CHUA HOAN THANH")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
# .venv/bin/python -m vlm.control_loop --target cube_red --max-steps 3 --no-thinking --verbose 
# .venv/bin/python -m vlm.control_loop --target "khoi lap phuong do cube_red" --max-steps 5 --no-thinking --verbose --view
