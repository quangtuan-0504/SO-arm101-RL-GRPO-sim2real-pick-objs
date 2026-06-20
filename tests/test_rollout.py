"""Test vòng rollout GRPO end-to-end với MockPolicy (server game chạy local).

Chạy:  ./.venv/bin/python -m tests.test_rollout
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from train.env_client import EnvClient
from train.rollout import EpisodeRollout, MockPolicy
from dataset.generator import load_levels

import os
BASE = os.environ.get("GAME_URL", "http://127.0.0.1:8200")


def main():
    joints = json.loads((ROOT / "dataset" / "robot_spec.json").read_text())["joints"]
    levels = load_levels(ROOT / "dataset" / "levels.json")
    level = levels[0]

    env = EnvClient(BASE)
    print("health:", env.health())

    roll = EpisodeRollout(env, joints, level, group_size=4)
    policy = MockPolicy(joints, bad_ratio=0.25, seed=1)

    roll.start()
    print(f"start: {level['level_id']} | {level['instruction_vi']}")

    for step in range(5):
        s = roll.collect_step(policy)
        print(f"step {s.info['step_idx']}: n_valid={s.info['n_valid']}/4 "
              f"fmt={[round(x,1) for x in s.format_rewards]} "
              f"geo={[round(x,3) for x in s.geometric_rewards]} "
              f"winner={s.winner_idx} advanced={s.advanced}")
        # group phải có ít nhất vài completion; reward valid phải phân biệt
        assert len(s.completions) == 4
        if s.info["n_valid"] >= 2:
            geo_valid = [s.geometric_rewards[i] for i, ok in enumerate(s.valid_mask) if ok]
            assert len(set(round(x, 4) for x in geo_valid)) > 1, \
                "reward không phân biệt giữa các nhánh hợp lệ"
        if roll.finished:
            print("  -> màn kết thúc.")
            break

    roll.close()
    print("\nTEST ROLLOUT PASS.")


if __name__ == "__main__":
    main()
