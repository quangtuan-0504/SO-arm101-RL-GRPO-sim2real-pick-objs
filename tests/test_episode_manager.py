"""Test EpisodeManager: pool 2 màn song song, chấm điểm + advance (Mac, no GPU).

Cần game service chạy ở cổng 8000.
Chạy:  ./.venv/bin/python -m tests.test_episode_manager
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from train.episode_manager import EpisodeManager
from train.rollout import MockPolicy
from dataset.generator import load_levels

BASE = "http://127.0.0.1:8000"


def main():
    joints = json.loads((ROOT / "dataset" / "robot_spec.json").read_text())["joints"]
    levels = load_levels(ROOT / "dataset" / "levels.json")

    mgr = EpisodeManager(BASE, levels, joints, n_parallel=2, group_size=4,
                         img_w=320, img_h=240, seed=2)
    policy = MockPolicy(joints, bad_ratio=0.2, seed=3)

    print("episodes:", mgr.episode_ids)
    for eid in mgr.episode_ids:
        print(f"  {eid} -> level {mgr.current_level_id(eid)}")

    # mô phỏng vài "batch": mỗi batch chấm cho cả 2 episode (song song)
    for batch in range(6):
        for eid in mgr.episode_ids:
            messages = mgr.current_prompt(eid)
            comps = policy.generate(messages, 4)
            res = mgr.score_and_advance(eid, comps)
            assert len(res.rewards) == 4
            print(f"[batch {batch}] {eid} lvl={res.info['level_id']} "
                  f"step={res.info['step_idx']} n_valid={res.info['n_valid']} "
                  f"geo={[round(x,2) for x in res.geometric_rewards]} "
                  f"winner={res.winner_idx} finished={res.finished}")

    mgr.close()
    print("\nTEST EPISODE MANAGER PASS.")


if __name__ == "__main__":
    main()
