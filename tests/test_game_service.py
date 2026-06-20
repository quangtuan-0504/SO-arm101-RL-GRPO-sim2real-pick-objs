"""Kiểm thử game service qua HTTP (server phải đang chạy ở cổng 8000).

Chạy:  ./.venv/bin/python -m tests.test_game_service
"""
from __future__ import annotations

import requests

import os
BASE = os.environ.get("GAME_URL", "http://127.0.0.1:8200")


def main():
    h = requests.get(f"{BASE}/health").json()
    print("health:", h)
    assert h["status"] == "ok" and h["n_levels"] == 15

    levels = requests.get(f"{BASE}/levels").json()
    lid = levels[0]["level_id"]
    print(f"levels: {len(levels)} | dùng {lid} -> {levels[0]['instruction_vi']}")

    r = requests.post(f"{BASE}/reset", json={"level_id": lid,
                                             "img_w": 320, "img_h": 240}).json()
    sid = r["session_id"]
    obs = r["obs"]
    assert obs["wrist_image"].startswith("data:image/png;base64,")
    print(f"reset: session={sid} step={obs['step_idx']}/{obs['max_steps']} "
          f"d_grip_obj={obs['measure']['d_grip_obj']:.3f}")

    base_angles = dict(obs["ctrl_angles"])
    actions = []
    for pan in (-0.5, 0.0, 0.5):
        a = dict(base_angles)
        a["shoulder_pan"] = pan
        actions.append(a)
    br = requests.post(f"{BASE}/branch",
                       json={"session_id": sid, "actions": actions}).json()
    branches = br["branches"]
    assert len(branches) == 3
    ds = [round(b["measure"]["d_grip_obj"], 4) for b in branches]
    print(f"branch: 3 nhánh d_grip_obj = {ds}")
    assert len(set(ds)) > 1

    # commit nhánh gần vật nhất
    winner = min(branches, key=lambda b: b["measure"]["d_grip_obj"])
    c = requests.post(f"{BASE}/commit",
                      json={"session_id": sid, "branch_id": winner["branch_id"]}).json()
    assert c["obs"]["step_idx"] == 1
    print(f"commit: nhánh {winner['branch_id']} -> step {c['obs']['step_idx']}")

    requests.post(f"{BASE}/close", json={"session_id": sid})
    print("\nTẤT CẢ KIỂM THỬ HTTP PASS.")


if __name__ == "__main__":
    main()
