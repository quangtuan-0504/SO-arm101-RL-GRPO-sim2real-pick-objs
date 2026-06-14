"""Ứng dụng mô phỏng cánh tay robot SO-101 gắp vật thể.

Chạy:  python app.py            (dùng config.yaml mặc định)
       python app.py my.yaml    (dùng file cấu hình khác)
"""
import sys
from pathlib import Path

import yaml

from sim.simulation import Simulation
from sim.viewer3 import ThreeViewApp

ROOT = Path(__file__).resolve().parent


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "config.yaml")
    print(f"[SO-101] Tải cấu hình: {cfg_path}")
    config = load_config(cfg_path)

    sim = Simulation(config)
    print(f"[SO-101] Model nạp xong: {sim.model.nu} actuator, "
          f"{sim.model.ncam} camera, {len(config.get('objects', []))} vật thể.")
    print("[SO-101] Mở cửa sổ 3 khung nhìn. Nhấn ESC để thoát.")

    app = ThreeViewApp(sim, config)
    app.run()


if __name__ == "__main__":
    main()
