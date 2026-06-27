"""Sinh các map game + bảng level.

Gồm hai nhóm map:
  - Map "đầy đủ" (mặc định 5 map × 3 màn): mỗi map có ÍT NHẤT 1 vật mỗi loại
    (3 loại) và TỐI ĐA 5 vật (thêm 0..2 vật "distractor" trùng loại để tăng độ
    khó thị giác). Mỗi map có 3 màn, mỗi màn yêu cầu gắp 1 LOẠI vật.
  - Map "curriculum dễ" (mặc định 5 map × 1 màn): mỗi map CHỈ có 1 vật và 1 màn,
    KHÔNG có vật gây nhiễu. Mục đích cho mô hình học từ dễ lên khó. Các map này
    đánh số nối tiếp sau các map đầy đủ, loại vật luân phiên qua 3 loại cơ bản.

Quy tắc chung (giai đoạn đầu — đơn giản, camera CỐ ĐỊNH cho mọi map):
  - Vị trí vật khác nhau giữa các map.
  - Mỗi màn chạy tối đa 20 step.

Vị trí vật được rải trong vùng với tới của cánh tay, cách nhau tối thiểu, và
TRÁNH vùng đặt giỏ. Camera/giỏ/cánh tay lấy từ ``config.yaml`` gốc và giữ nguyên.

Đầu ra: file JSON ``dataset/levels.json`` chứa toàn bộ level table. Mỗi level
là một bản ghi tự chứa (đủ để dựng lại cảnh mà không cần state ngoài).

Chạy:  ./.venv/bin/python -m dataset.generator --seed 0 --maps 5
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import yaml

from .object_palette import OBJECT_TYPES, TYPES_BY_ID

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "dataset" / "levels.json"

# Vùng với tới của cánh tay để rải vật (mặt phẳng xy, mét).
WORKSPACE = {
    "x": (0.12, 0.22),
    "y": (-0.10, 0.10),
}
MIN_SEPARATION = 0.05   # khoảng cách tối thiểu giữa 2 vật (m)
MAX_OBJECTS = 5
LEVELS_PER_MAP = 3
MAX_STEPS = 20


def _basket_keepout(config: dict):
    """Trả về (cx, cy, bán kính) vùng cấm quanh giỏ để vật không đè giỏ."""
    b = config.get("basket", {})
    if not b.get("enabled", False):
        return None
    pos = b.get("pos", [0.12, -0.20, 0.0])
    inner = b.get("inner_size", [0.05, 0.05, 0.04])
    r = math.hypot(inner[0], inner[1]) + 0.06  # lề an toàn
    return float(pos[0]), float(pos[1]), r


def _sample_positions(rng: random.Random, n: int, keepout) -> list:
    """Rải ``n`` vị trí (x, y) cách nhau >= MIN_SEPARATION, tránh giỏ."""
    pts: list[tuple[float, float]] = []
    tries = 0
    while len(pts) < n and tries < 2000:
        tries += 1
        x = rng.uniform(*WORKSPACE["x"])
        y = rng.uniform(*WORKSPACE["y"])
        if keepout is not None:
            cx, cy, r = keepout
            if math.hypot(x - cx, y - cy) < r:
                continue
        if all(math.hypot(x - px, y - py) >= MIN_SEPARATION for px, py in pts):
            pts.append((x, y))
    if len(pts) < n:
        raise RuntimeError(f"Không rải đủ {n} vị trí (chỉ được {len(pts)}). "
                           "Nới WORKSPACE hoặc giảm số vật.")
    return pts


def _build_objects(rng: random.Random, keepout) -> list:
    """Tạo danh sách vật cho 1 map: 1 mỗi loại + 0..2 distractor, <= 5 vật."""
    base_types = list(OBJECT_TYPES)
    n_extra = rng.randint(0, MAX_OBJECTS - len(base_types))
    chosen = list(base_types) + [rng.choice(base_types) for _ in range(n_extra)]

    positions = _sample_positions(rng, len(chosen), keepout)
    counts: dict[str, int] = {}
    objects = []
    for t, (x, y) in zip(chosen, positions):
        counts[t.type_id] = counts.get(t.type_id, 0) + 1
        suffix = "" if counts[t.type_id] == 1 else f"_{counts[t.type_id]}"
        objects.append({
            "name": f"{t.type_id}{suffix}",
            "type_id": t.type_id,
            "shape": t.shape,
            "size": list(t.size),
            "pos": [round(x, 4), round(y, 4), t.rest_z],
            "rgba": list(t.rgba),
            "mass": t.mass,
        })
    return objects


def _target_for_type(objects: list, type_id: str) -> str:
    """Tên instance đầu tiên thuộc ``type_id`` (mục tiêu của màn)."""
    for o in objects:
        if o["type_id"] == type_id:
            return o["name"]
    raise KeyError(f"Map thiếu loại {type_id}")


def _instruction_vi(t, target_name: str) -> str:
    return f"Gắp {t.desc_vi} ({target_name}) rồi thả vào giỏ."


def _instruction_en(t, target_name: str) -> str:
    return f"Grasp the {t.desc_en} ({target_name}) and drop it into the basket."


def _clean_objects(objects: list) -> list:
    """Chỉ giữ các khoá cần để scene_builder dựng lại cảnh."""
    return [
        {k: o[k] for k in ("name", "shape", "size", "pos", "rgba", "mass")}
        for o in objects
    ]


def _build_single_object(rng: random.Random, t, keepout) -> dict:
    """Tạo 1 vật duy nhất (đúng 1 instance loại ``t``) cho map curriculum dễ."""
    (x, y), = _sample_positions(rng, 1, keepout)
    return {
        "name": t.type_id,
        "type_id": t.type_id,
        "shape": t.shape,
        "size": list(t.size),
        "pos": [round(x, 4), round(y, 4), t.rest_z],
        "rgba": list(t.rgba),
        "mass": t.mass,
    }


def generate(config: dict, *, n_maps: int = 5, n_easy_maps: int = 5,
             seed: int = 0) -> list:
    """Sinh toàn bộ level table. Trả về list các bản ghi level.

    Gồm hai phần:
      - ``n_maps`` map "đầy đủ": mỗi map >=1 vật mỗi loại + 0..2 distractor,
        3 màn (1 màn / loại). Đây là phần khó.
      - ``n_easy_maps`` map "curriculum dễ": mỗi map CHỈ 1 vật và 1 màn, không
        có vật gây nhiễu. Đánh số nối tiếp sau các map đầy đủ. Loại vật luân
        phiên qua 3 loại cơ bản để mô hình làm quen từng loại.
    """
    rng = random.Random(seed)
    keepout = _basket_keepout(config)
    levels = []
    for map_idx in range(n_maps):
        objects = _build_objects(rng, keepout)
        for lvl_idx, t in enumerate(OBJECT_TYPES):
            target_name = _target_for_type(objects, t.type_id)
            level_id = f"map{map_idx}_lvl{lvl_idx}_{t.type_id}"
            levels.append({
                "level_id": level_id,
                "map_id": f"map{map_idx}",
                "target_type": t.type_id,
                "target_object_name": target_name,
                "instruction_vi": _instruction_vi(t, target_name),
                "instruction_en": _instruction_en(t, target_name),
                "max_steps": MAX_STEPS,
                # objects tự chứa: đủ để scene_builder dựng lại cảnh map này.
                "objects": _clean_objects(objects),
            })

    # Phần curriculum dễ: map đơn-vật, đánh số nối tiếp sau các map đầy đủ.
    for i in range(n_easy_maps):
        map_idx = n_maps + i
        t = OBJECT_TYPES[i % len(OBJECT_TYPES)]
        obj = _build_single_object(rng, t, keepout)
        target_name = obj["name"]
        levels.append({
            "level_id": f"map{map_idx}_lvl0_{t.type_id}",
            "map_id": f"map{map_idx}",
            "target_type": t.type_id,
            "target_object_name": target_name,
            "instruction_vi": _instruction_vi(t, target_name),
            "instruction_en": _instruction_en(t, target_name),
            "max_steps": MAX_STEPS,
            "objects": _clean_objects([obj]),
        })
    return levels


def robot_spec(config: dict) -> dict:
    """Trích giới hạn khớp + home pose (cố định) để build prompt KHÔNG cần sim.

    Cần MuJoCo (chạy trên Mac). Server GPU chỉ đọc file JSON kết quả, không cần
    dựng lại sim chỉ để lấy giới hạn khớp.
    """
    from sim.simulation import Simulation
    from sim.scene_builder import JOINT_NAMES

    sim = Simulation(config)
    home = config.get("arm", {}).get("home_pose", {})
    joints = []
    for name in JOINT_NAMES:
        lo, hi = sim.get_ctrl_range(name)
        joints.append({"name": name, "lo": lo, "hi": hi,
                       "home": float(home.get(name, 0.0))})
    return {"joints": joints}


def level_to_config(base_config: dict, level: dict) -> dict:
    """Ghép 1 level lên config gốc -> config đầy đủ để dựng Simulation.

    Giữ nguyên cánh tay/camera/giỏ của ``base_config`` (camera CỐ ĐỊNH),
    chỉ thay danh sách ``objects`` bằng của level.
    """
    cfg = json.loads(json.dumps(base_config))  # deep copy đơn giản
    cfg["objects"] = [dict(o) for o in level["objects"]]
    return cfg


def load_levels(path: Path = DEFAULT_OUT) -> list:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    p = argparse.ArgumentParser(description="Sinh map game + level table")
    p.add_argument("--config", default=str(ROOT / "config.yaml"))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--maps", type=int, default=5)
    p.add_argument("--easy-maps", type=int, default=5,
                   help="Số map curriculum dễ (1 vật, 1 màn), nối sau map đầy đủ.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    levels = generate(config, n_maps=args.maps, n_easy_maps=args.easy_maps,
                      seed=args.seed)
    Path(args.out).write_text(
        json.dumps(levels, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Đã sinh {len(levels)} level "
          f"({args.maps} map đầy đủ × {LEVELS_PER_MAP} + "
          f"{args.easy_maps} map dễ × 1) -> {args.out}")

    # Xuất robot_spec.json (giới hạn khớp + home) cho server GPU build prompt.
    spec_path = Path(args.out).parent / "robot_spec.json"
    spec_path.write_text(json.dumps(robot_spec(config), ensure_ascii=False,
                                    indent=2), encoding="utf-8")
    print(f"Đã xuất robot_spec -> {spec_path}")
    # In tóm tắt
    by_map: dict[str, int] = {}
    for lv in levels:
        by_map[lv["map_id"]] = by_map.get(lv["map_id"], 0) + 1
    for mid in sorted(by_map):
        sample = next(l for l in levels if l["map_id"] == mid)
        print(f"  {mid}: {len(sample['objects'])} vật, {by_map[mid]} màn")


if __name__ == "__main__":
    main()
