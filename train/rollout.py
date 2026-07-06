"""Thu thập rollout GRPO theo từng STEP của môi trường (single-step GRPO).

Mỗi step của một màn = MỘT mẫu GRPO:
    prompt   = (2 ảnh trạng thái i + instruction + bảng khớp)
    group    = G completions do policy sinh ra (G lệnh điều khiển)
    reward   = geometric(PBRS) cho nước HỢP LỆ; nước SAI FORMAT bị đặt thấp hơn
               mọi nước hợp lệ (gate theo thứ hạng)  [→ judge VLM, tuỳ chọn]

Luồng:
  1. Build prompt từ ``StepObs`` + ``robot_spec``.
  2. Policy sinh G completions (text).
  3. format gate: completion sai format KHÔNG gửi game; reward của nó được đặt
     = (nước hợp lệ tệ nhất − margin) nên advantage GRPO luôn âm.
  4. Gửi các lệnh HỢP LỆ cho game ``/branch`` -> G' trạng thái i+1.
  5. geometric_reward(before=measure_i, after=measure_{i+1}) cho từng nhánh hợp lệ.
  6. Chọn winner (reward cao nhất) -> ``/commit`` -> trạng thái i+1 để đi tiếp.

Phần này KHÔNG cần GPU: chạy được với ``MockPolicy`` + game service local.
Trên server GPU, thay ``MockPolicy`` bằng policy thật (vLLM/HF sinh G completions).
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Protocol

from sim.metrics import StateMeasure
from rewards.format_reward import check_format
from rewards.geometric_reward import geometric_reward, GeoWeights
from vlm.prompt_builder import build_messages_from_spec
from train.env_client import EnvClient, StepObs, Branch


class Policy(Protocol):
    def generate(self, messages: list, n: int) -> list[str]:
        """Sinh ``n`` completions (text) cho cùng một prompt."""
        ...


@dataclass
class GroupSample:
    """Một nhóm GRPO ứng với 1 step môi trường."""
    prompt_messages: list
    completions: list[str]
    format_rewards: list[float]
    geometric_rewards: list[float]
    total_rewards: list[float]
    valid_mask: list[bool]
    winner_idx: int | None
    advanced: bool                 # đã commit winner để sang i+1 chưa
    info: dict = field(default_factory=dict)


def _measure_from_dict(d: dict) -> StateMeasure:
    return StateMeasure(**d)


class EpisodeRollout:
    """Quản lý rollout của MỘT màn (một session game)."""

    def __init__(self, env: EnvClient, joints_spec: list, level: dict,
                 *, group_size: int = 4,
                 geo_weights: GeoWeights | None = None,
                 invalid_format_margin: float = 1.0,
                 stall_penalty: float = 0.5,
                 stall_eps: float = 1e-2):
        self.env = env
        self.joints = joints_spec
        self.level = level
        self.group_size = group_size
        self.geo_weights = geo_weights or GeoWeights()
        # Format là GATE CỨNG theo THỨ HẠNG: sai cú pháp KHÔNG branch, reward
        # đặt thấp hơn mọi nước hợp lệ một khoảng ``invalid_format_margin`` ->
        # advantage GRPO của nó luôn âm (không học ngược). Hợp lệ chấm bằng geo.
        self.invalid_format_margin = invalid_format_margin
        # Anti-stall: phạt completion ra góc gần y hệt pose hiện tại (copy
        # home/example/pose cũ -> tay không nhúc nhích). Tạo phương sai trong
        # nhóm GRPO để gradient ưu tiên "có di chuyển" hơn "đứng yên".
        self.stall_penalty = stall_penalty
        self.stall_eps = stall_eps
        self.target_desc = level.get("instruction_vi", level["target_object_name"])
        self.obs: StepObs | None = None
        self.finished = False

    def start(self) -> StepObs:
        self.obs = self.env.reset(self.level["level_id"])
        self.finished = self.obs.done
        return self.obs

    def collect_step(self, policy: Policy) -> GroupSample:
        """Sinh 1 nhóm GRPO cho step hiện tại và tiến môi trường sang winner."""
        assert self.obs is not None, "gọi start() trước"
        obs = self.obs
        messages = build_messages_from_spec(
            self.joints, self.target_desc, obs.wrist_image, obs.stand_image,
            current_angles=obs.ctrl_angles)

        completions = policy.generate(messages, self.group_size)

        # --- format gate (theo thứ hạng, tính reward ở cuối) ---
        valid_mask: list[bool] = []
        angles_list: list[dict | None] = []
        for text in completions:
            res = check_format(text)
            valid_mask.append(res.ok)
            angles_list.append(res.angles if res.ok else None)

        # --- branch CHỈ các lệnh hợp lệ ---
        valid_indices = [i for i, ok in enumerate(valid_mask) if ok]
        branches: list[Branch] = []
        if valid_indices:
            actions = [angles_list[i] for i in valid_indices]
            branches = self.env.branch(actions)

        # --- geometric reward ---
        before = _measure_from_dict(obs.measure)
        geo_rewards = [0.0] * len(completions)
        branch_of = {}   # completion_idx -> Branch
        for k, comp_idx in enumerate(valid_indices):
            br = branches[k]
            branch_of[comp_idx] = br
            after = _measure_from_dict(br.measure)
            geo_rewards[comp_idx] = geometric_reward(
                before, after, weights=self.geo_weights,
                obj_speed_after=br.obj_speed)

        # --- anti-stall: CHỈ phạt no-op THẬT SỰ (khớp tay LẪN gripper không
        # đổi). "Giữ tay + đóng kẹp" lúc gắp KHÔNG bị phạt oan. ---
        cur = obs.ctrl_angles or {}
        n_stalled = 0
        stall_flags = [False] * len(completions)
        for i in valid_indices:
            ang = angles_list[i] or {}
            arm_deltas = [abs(float(ang[n]) - float(cur.get(n, 0.0)))
                          for n in ang if n != "gripper" and n in cur]
            arm_delta = max(arm_deltas) if arm_deltas else 0.0
            grip_delta = (abs(float(ang["gripper"]) - float(cur["gripper"]))
                          if "gripper" in ang and "gripper" in cur else 0.0)
            if arm_delta < self.stall_eps and grip_delta < self.stall_eps:
                stall_flags[i] = True
                n_stalled += 1

        # --- tổng hợp reward: hợp lệ = geo (trừ stall); sai format = tệ nhất
        # trong nhóm hợp lệ − margin (advantage luôn âm, không học ngược). ---
        total = [0.0] * len(completions)
        for i in valid_indices:
            total[i] = geo_rewards[i] - (self.stall_penalty if stall_flags[i] else 0.0)
        if valid_indices:
            worst_valid = min(total[i] for i in valid_indices)
            invalid_value = worst_valid - self.invalid_format_margin
        else:
            invalid_value = 0.0
        format_rewards = [0.0] * len(completions)
        for i in range(len(completions)):
            if not valid_mask[i]:
                total[i] = invalid_value
                format_rewards[i] = invalid_value

        # --- chọn winner trong các nhánh hợp lệ (theo geometric) ---
        winner_idx = None
        advanced = False
        if valid_indices:
            winner_idx = max(valid_indices, key=lambda i: geo_rewards[i])
            next_obs = self.env.commit(branch_of[winner_idx].branch_id)
            self.obs = next_obs
            advanced = True
            if next_obs.done or next_obs.step_idx >= next_obs.max_steps:
                self.finished = True

        return GroupSample(
            prompt_messages=messages,
            completions=completions,
            format_rewards=format_rewards,
            geometric_rewards=geo_rewards,
            total_rewards=total,
            valid_mask=valid_mask,
            winner_idx=winner_idx,
            advanced=advanced,
            info={"step_idx": obs.step_idx,
                  "n_valid": len(valid_indices),
                  "n_stalled": n_stalled,
                  "stall_flags": stall_flags,
                  "level_id": self.level["level_id"]},
        )

    def close(self):
        self.env.close()


# ----------------------------------------------------------------------
#  MockPolicy — không cần GPU, để test toàn bộ vòng rollout trên Mac.
# ----------------------------------------------------------------------
class MockPolicy:
    """Sinh G completions JSON, đa dạng nhẹ shoulder_pan; thi thoảng sai format.

    Mục đích: xác nhận format gate + branch + reward + commit chạy đúng end-to-end.
    """

    def __init__(self, joints_spec: list, *, bad_ratio: float = 0.25, seed: int = 0):
        self.home = {j["name"]: j["home"] for j in joints_spec}
        self.bad_ratio = bad_ratio
        self.rng = random.Random(seed)

    def generate(self, messages: list, n: int) -> list[str]:
        out = []
        for k in range(n):
            if self.rng.random() < self.bad_ratio:
                out.append("Xin lỗi, tôi không chắc nên không trả JSON.")
                continue
            angles = dict(self.home)
            angles["shoulder_pan"] = round(self.rng.uniform(-0.8, 0.8), 3)
            angles["shoulder_lift"] = round(
                self.home["shoulder_lift"] + self.rng.uniform(-0.2, 0.2), 3)
            out.append(json.dumps({
                "reasoning": f"mock {k}",
                "target_object": "x",
                "phase": "approach",
                "gripper_action": "open",
                "joint_angles_rad": angles,
                "done": False,
                "confidence": 0.5,
            }, ensure_ascii=False))
        return out
