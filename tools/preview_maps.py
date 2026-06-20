"""Render ảnh xem trước từng MAP để kiểm tra bố trí vật thể.

Mỗi map có 3 màn nhưng CÙNG bố trí vật (chỉ khác mục tiêu), nên ta chỉ cần
render 1 lần mỗi map. Với mỗi map xuất 3 ảnh:
  - <map>_stand.png : góc nhìn camera giá đỡ (stand_cam)
  - <map>_top.png   : góc nhìn từ trên xuống (kiểm tra vị trí + khoảng cách vật)
  - <map>_oper.png  : góc nhìn người điều khiển (operator_camera trong config)

Ảnh lưu ở: logs/map_preview/

Chạy:  ./.venv/bin/python -m tools.preview_maps
       ./.venv/bin/python -m tools.preview_maps --map map2
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import yaml
import numpy as np
import mujoco

from dataset.generator import load_levels, level_to_config, DEFAULT_OUT
from sim.simulation import Simulation
from vlm.render_utils import CameraRenderer, save_png

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "logs" / "map_preview"


def _free_cam(lookat, distance, azimuth, elevation):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation
    return cam


def _render_free(renderer, data, cam):
    renderer._renderer.update_scene(data, camera=cam)
    return renderer._renderer.render()


def _basket_keepout(config):
    b = config.get("basket", {})
    if not b.get("enabled", False):
        return None
    pos = b.get("pos", [0.12, -0.20, 0.0])
    inner = b.get("inner_size", [0.05, 0.05, 0.04])
    r = math.hypot(inner[0], inner[1]) + 0.06
    return float(pos[0]), float(pos[1]), r


def _check_layout(level, keepout):
    """Kiểm tra các yêu cầu bố trí: đủ 3 loại, <=5 vật, khoảng cách, tránh giỏ."""
    objs = level["objects"]
    issues = []
    # đủ loại cơ bản
    names = [o["name"].split("_")[0] + "_" + o["name"].split("_")[1]
             if len(o["name"].split("_")) >= 2 else o["name"] for o in objs]
    base_types = set()
    for o in objs:
        # suy ra loại từ tên: bỏ hậu tố _<số>
        parts = o["name"].rsplit("_", 1)
        base = o["name"]
        if len(parts) == 2 and parts[1].isdigit():
            base = parts[0]
        base_types.add(base)
    for need in ("cube_red", "cyl_green", "sphere_blue"):
        if need not in base_types:
            issues.append(f"thiếu loại {need}")
    # số lượng
    if len(objs) > 5:
        issues.append(f"quá 5 vật ({len(objs)})")
    # khoảng cách tối thiểu 0.05 và tránh giỏ
    pts = [(o["pos"][0], o["pos"][1], o["name"]) for o in objs]
    for i in range(len(pts)):
        xi, yi, ni = pts[i]
        if keepout is not None:
            cx, cy, r = keepout
            if math.hypot(xi - cx, yi - cy) < r:
                issues.append(f"{ni} nằm trong vùng giỏ")
        for j in range(i + 1, len(pts)):
            xj, yj, nj = pts[j]
            d = math.hypot(xi - xj, yi - yj)
            if d < 0.05:
                issues.append(f"{ni}<->{nj} quá gần ({d:.3f}m)")
    return issues


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--levels", default=str(DEFAULT_OUT))
    p.add_argument("--map", default=None, help="chỉ render 1 map, ví dụ map2")
    p.add_argument("--w", type=int, default=640)
    p.add_argument("--h", type=int, default=480)
    args = p.parse_args()

    base_config = yaml.safe_load(
        (ROOT / "config.yaml").read_text(encoding="utf-8"))
    levels = load_levels(Path(args.levels))
    keepout = _basket_keepout(base_config)

    # gom 1 level đại diện cho mỗi map
    by_map = {}
    for lv in levels:
        by_map.setdefault(lv["map_id"], lv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    oper = base_config.get("operator_camera", {})
    map_ids = sorted(by_map) if args.map is None else [args.map]

    print(f"Render {len(map_ids)} map -> {OUT_DIR}\n")
    for mid in map_ids:
        lv = by_map.get(mid)
        if lv is None:
            print(f"  [!] không có {mid}")
            continue
        config = level_to_config(base_config, lv)
        sim = Simulation(config)
        renderer = CameraRenderer(sim.model, width=args.w, height=args.h)

        stand = renderer.render(sim.data, "stand_cam")
        save_png(OUT_DIR / f"{mid}_stand.png", stand)

        top = _render_free(renderer, sim.data,
                           _free_cam([0.17, 0.0, 0.0], 0.55, 90, -89))
        save_png(OUT_DIR / f"{mid}_top.png", top)

        oper_cam = _free_cam(
            oper.get("look_at", [0.12, 0.0, 0.12]),
            float(oper.get("distance", 0.9)),
            float(oper.get("azimuth", 135)),
            float(oper.get("elevation", -25)))
        oper_img = _render_free(renderer, sim.data, oper_cam)
        save_png(OUT_DIR / f"{mid}_oper.png", oper_img)

        renderer.close()

        issues = _check_layout(lv, keepout)
        status = "OK" if not issues else "CẦN XEM"
        n = len(lv["objects"])
        print(f"  {mid}: {n} vật  [{status}]")
        for it in issues:
            print(f"       - {it}")

    print(f"\nXong. Mở thư mục: {OUT_DIR}")


if __name__ == "__main__":
    main()
