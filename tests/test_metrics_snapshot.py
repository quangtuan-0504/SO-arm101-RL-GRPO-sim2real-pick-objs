"""Kiểm thử nhanh (Mac, không cần GPU): snapshot/restore + metrics.

Chạy:  ./.venv/bin/python -m tests.test_metrics_snapshot
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sim.simulation import Simulation
from sim.metrics import (basket_geometry, measure_state, potential, dist)


def main():
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    sim = Simulation(cfg)

    # --- 1) snapshot/restore -----------------------------------------
    s0 = sim.get_state()
    pan_adr = sim.qpos_adr["shoulder_pan"]
    sim.set_ctrl("shoulder_pan", 1.0)
    sim.step(300)
    moved = float(sim.data.qpos[pan_adr])
    assert abs(moved - float(s0["qpos"][pan_adr])) > 0.1, "khớp chưa di chuyển sau step"
    sim.set_state(s0)
    restored = sim.data.qpos.copy()
    assert np.allclose(restored, s0["qpos"]), "restore qpos không khớp"
    print("[OK] snapshot/restore: qpos khôi phục đúng sau khi branch.")

    # --- 2) branch nhiều nhánh từ cùng 1 trạng thái ------------------
    base = sim.get_state()
    ends = []
    for pan in (-1.0, 0.0, 1.0):
        sim.set_state(base)
        sim.set_ctrl("shoulder_pan", pan)
        sim.step(150)
        ends.append(sim.grasp_point().copy())
    spread = dist(ends[0], ends[-1])
    assert spread > 0.05, "các nhánh không tách biệt"
    print(f"[OK] branch 3 nhánh từ cùng state -> đầu kẹp tách biệt {spread:.3f} m.")

    # --- 3) metrics + potential --------------------------------------
    sim.set_state(base)
    basket = basket_geometry(cfg)
    assert basket is not None, "config chưa bật giỏ"
    target = sim.object_names[0]
    m = measure_state(sim, target, basket)
    phi = potential(m)
    print(f"[OK] đo trạng thái vật {target!r}: d_grip_obj={m.d_grip_obj:.3f} "
          f"d_obj_basket={m.d_obj_basket:.3f} grasped={m.grasped} "
          f"in_basket={m.in_basket} | Phi={phi:.3f}")

    print("\nTẤT CẢ KIỂM THỬ PASS.")


if __name__ == "__main__":
    main()
