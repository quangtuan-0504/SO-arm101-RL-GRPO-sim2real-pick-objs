"""Thu thập rollout GRPO theo từng STEP của môi trường (single-step GRPO).

Mỗi step của một màn = MỘT mẫu GRPO:
    prompt   = (2 ảnh trạng thái i + instruction + bảng khớp)
    group    = G completions do policy sinh ra (G lệnh điều khiển)
    reward   = format_gate(0/1)  →  geometric(PBRS)  [→ judge VLM, tuỳ chọn]

Luồng:
  1. Build prompt từ ``StepObs`` + ``robot_spec``.
  2. Policy sinh G completions (text).
  3. format gate: completion sai format -> reward 0, KHÔNG gửi game.
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
                 format_weight: float = 1.0):
        self.env = env
        self.joints = joints_spec
        self.level = level
        self.group_size = group_size
        self.geo_weights = geo_weights or GeoWeights()
        self.format_weight = format_weight
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
            self.joints, self.target_desc, obs.wrist_image, obs.stand_image)

        completions = policy.generate(messages, self.group_size)

        # --- format gate ---
        valid_mask: list[bool] = []
        angles_list: list[dict | None] = []
        format_rewards: list[float] = []
        for text in completions:
            res = check_format(text)
            valid_mask.append(res.ok)
            format_rewards.append(self.format_weight if res.ok else 0.0)
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

        total = [format_rewards[i] + geo_rewards[i] for i in range(len(completions))]

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
