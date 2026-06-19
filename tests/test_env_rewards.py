"""Kiểm thử end-to-end phần Mac (không GPU): level -> env -> branch -> reward.

Chạy:  ./.venv/bin/python -m tests.test_env_rewards
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dataset.generator import load_levels, generate
from game.env import GameEnv
from rewards.format_reward import check_format, format_reward
from rewards.geometric_reward import geometric_reward


def _action_json(angles: dict, phase="approach", grip="open", done=False) -> str:
    return json.dumps({
        "reasoning": "test",
        "target_object": "x",
        "phase": phase,
        "gripper_action": grip,
        "joint_angles_rad": angles,
        "done": done,
        "confidence": 0.5,
    })


def main():
    base_cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    levels_path = ROOT / "dataset" / "levels.json"
    levels = load_levels(levels_path) if levels_path.exists() else generate(base_cfg)
    level = levels[0]
    print(f"Level: {level['level_id']} | mục tiêu: {level['target_object_name']} "
          f"| {len(level['objects'])} vật")

    env = GameEnv(base_cfg, level, img_w=320, img_h=240)
    try:
        obs = env.reset()
        assert obs.wrist_rgb.shape[2] == 3 and obs.stand_rgb.shape[2] == 3
        print(f"[OK] reset: step={obs.step_idx}/{obs.max_steps} "
              f"d_grip_obj={obs.measure.d_grip_obj:.3f}")

        # --- format reward ---
        good = _action_json(obs.ctrl_angles)
        bad = '{"phase": "approach"}'           # thiếu khoá
        not_json = "xin chào tôi không trả JSON"
        assert format_reward(good) == 1.0
        assert format_reward(bad) == 0.0
        assert format_reward(not_json) == 0.0
        print("[OK] format_reward: hợp lệ=1.0, thiếu khoá=0.0, không-JSON=0.0")

        # --- branch G=3 nhánh: nhắm shoulder_pan về 3 hướng ---
        base_angles = dict(obs.ctrl_angles)
        actions = []
        for pan in (-0.5, 0.0, 0.5):
            a = dict(base_angles)
            a["shoulder_pan"] = pan
            actions.append(a)
        results = env.branch(actions)
        assert len(results) == 3
        # các nhánh phải cho số đo khác nhau (đã tách biệt)
        ds = [round(r.measure.d_grip_obj, 4) for r in results]
        print(f"[OK] branch 3 nhánh: d_grip_obj = {ds}")
        assert len(set(ds)) > 1, "các nhánh không tách biệt"

        # --- geometric reward cho từng nhánh (before = obs hiện tại) ---
        rewards = [
            geometric_reward(obs.measure, r.measure, obj_speed_after=r.obj_speed)
            for r in results
        ]
        print(f"[OK] geometric_reward 3 nhánh = "
              f"{[round(x, 4) for x in rewards]}")
        assert len(set(round(x, 4) for x in rewards)) > 1, \
            "reward không phân biệt giữa các nhánh (advantage sẽ = 0)"

        # --- commit winner (reward cao nhất) rồi xác nhận step tăng ---
        winner = max(range(3), key=lambda i: rewards[i])
        obs2 = env.commit(results[winner])
        assert obs2.step_idx == 1
        print(f"[OK] commit nhánh {winner}: step -> {obs2.step_idx}")

    finally:
        env.close()

    print("\nTẤT CẢ KIỂM THỬ PASS.")


if __name__ == "__main__":
    main()
