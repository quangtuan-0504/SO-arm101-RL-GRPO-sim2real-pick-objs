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

Chạy thật với Qwen-VL (DashScope compatible-mode):
    export QWEN_API_KEY=sk-xxx
    export QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
    python -m vlm.control_loop --target cube_red --model qwen-vl-max-latest
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import yaml

from sim.scene_builder import JOINT_NAMES, ROOT
from sim.simulation import Simulation
from vlm.prompt_builder import build_messages
from vlm.render_utils import CameraRenderer, rgb_to_data_url, save_png
from vlm.qwen_client import QwenVLClient, MockQwenClient


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


def build_client(args, sim_for_mock):
    if args.mock:
        print("[VLM] Dung MockQwenClient (khong goi mang).")
        return MockQwenClient(sim_for_mock, done_after=args.mock_done_after)

    api_key = args.api_key or os.environ.get("QWEN_API_KEY", "")
    base_url = args.base_url or os.environ.get(
        "QWEN_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1")
    if not api_key:
        sys.exit("Thieu API key. Dat QWEN_API_KEY hoac --api-key, "
                 "hoac dung --mock de chay thu.")
    print(f"[VLM] Goi Qwen-VL model={args.model} @ {base_url}")
    return QwenVLClient(api_key=api_key, base_url=base_url, model=args.model)


def main():
    p = argparse.ArgumentParser(description="Vong lap dieu khien SO-101 bang VLM")
    p.add_argument("--config", default=str(ROOT / "config.yaml"))
    p.add_argument("--target", default="khoi lap phuong mau do (cube_red)",
                   help="Mo ta vat the can gap")
    p.add_argument("--max-steps", type=int, default=12)
    p.add_argument("--settle-seconds", type=float, default=0.6)
    p.add_argument("--model", default="qwen-vl-max-latest")
    p.add_argument("--api-key", default=None)
    p.add_argument("--base-url", default=None)
    p.add_argument("--mock", action="store_true",
                   help="Chay thu khong goi mang (client gia lap)")
    p.add_argument("--mock-done-after", type=int, default=3)
    p.add_argument("--no-debug-img", action="store_true")
    args = p.parse_args()

    config = load_config(args.config)
    # Sim tạm chỉ để mock biết home_pose/ctrlrange (run_loop tạo sim riêng).
    sim_tmp = Simulation(config)
    client = build_client(args, sim_tmp)

    ok = run_loop(config, args.target, client,
                  max_steps=args.max_steps,
                  settle_seconds=args.settle_seconds,
                  save_debug=not args.no_debug_img)
    print("KET QUA:", "DONE" if ok else "CHUA HOAN THANH")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
