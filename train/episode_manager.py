"""Quản lý pool các màn chạy SONG SONG cho huấn luyện GRPO.

Mỗi "episode" giữ một session game (EnvClient) trên một level. Manager cung cấp:
  - ``current_prompt(eid)``      : messages cho step hiện tại của episode.
  - ``score_and_advance(eid, completions)`` : chấm G completions (format gate +
    geometric), commit winner để tiến môi trường, trả về reward + thành phần.
  - tự reset episode sang level mới khi màn kết thúc (done / hết step).

Tách khỏi trl để TEST ĐƯỢC trên Mac (với MockPolicy) — xem tests/test_episode_manager.py.
KHÔNG cần GPU.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from train.env_client import EnvClient, StepObs, Branch
from rewards.format_reward import check_format
from rewards.geometric_reward import geometric_reward, GeoWeights
from sim.metrics import StateMeasure
from vlm.prompt_builder import build_messages_from_spec


def _measure(d: dict) -> StateMeasure:
    return StateMeasure(**d)


@dataclass
class ScoreResult:
    rewards: list[float]           # tổng reward mỗi completion (theo đúng thứ tự)
    format_rewards: list[float]
    geometric_rewards: list[float]
    valid_mask: list[bool]
    winner_idx: int | None
    advanced: bool
    finished: bool
    info: dict = field(default_factory=dict)


@dataclass
class _Episode:
    env: EnvClient
    level: dict
    obs: StepObs


class EpisodeManager:
    def __init__(self, base_url: str, levels: list, joints_spec: list,
                 *, n_parallel: int = 2, group_size: int = 4,
                 geo_weights: GeoWeights | None = None,
                 format_weight: float = 1.0, img_w: int = 480, img_h: int = 360,
                 timeout: float = 180.0, seed: int = 0,
                 attach_images: bool = False):
        self.base_url = base_url
        self.levels = levels
        self.joints = joints_spec
        self.group_size = group_size
        self.geo_weights = geo_weights or GeoWeights()
        self.format_weight = format_weight
        self.img_w, self.img_h = img_w, img_h
        self.timeout = timeout
        self.attach_images = attach_images  # đính ảnh winner vào info để lưu đĩa
        self.rng = random.Random(seed)
        self._episodes: dict[str, _Episode] = {}
        self._order: list[str] = []

        # Health-check sớm: báo lỗi rõ ràng nếu tunnel/game service chưa sẵn sàng.
        self._preflight()

        for i in range(n_parallel):
            self._spawn(f"ep{i}")

    def _preflight(self):
        probe = EnvClient(self.base_url, timeout=self.timeout)
        try:
            h = probe.health()
        except Exception as e:
            raise RuntimeError(
                f"Không gọi được game service tại {self.base_url} ({e}).\n"
                "Kiểm tra: (1) game service đang chạy trên Mac "
                "('caffeinate -dimsu python -m game.service'); "
                "(2) SSH reverse tunnel còn sống "
                "('ssh -N -R 8000:localhost:8000 -p <port> root@<host>'); "
                "(3) thử 'curl http://127.0.0.1:8000/health' trên server GPU."
            ) from e
        if h.get("n_levels", 0) <= 0:
            raise RuntimeError(
                "Game service chạy nhưng chưa có level. Chạy trên Mac: "
                "'python -m dataset.generator --maps 5'.")
        print(f"[EpisodeManager] game service OK: {h}")

    # ------------------------------------------------------------------
    def _pick_level(self) -> dict:
        return self.rng.choice(self.levels)

    def _spawn(self, eid: str):
        env = EnvClient(self.base_url, timeout=self.timeout)
        level = self._pick_level()
        obs = env.reset(level["level_id"], img_w=self.img_w, img_h=self.img_h)
        self._episodes[eid] = _Episode(env=env, level=level, obs=obs)
        if eid not in self._order:
            self._order.append(eid)

    def _reset_episode(self, eid: str):
        ep = self._episodes[eid]
        ep.env.close()
        level = self._pick_level()
        obs = ep.env.reset(level["level_id"], img_w=self.img_w, img_h=self.img_h)
        ep.level = level
        ep.obs = obs

    # ------------------------------------------------------------------
    @property
    def episode_ids(self) -> list[str]:
        return list(self._order)

    def target_desc(self, eid: str) -> str:
        lv = self._episodes[eid].level
        return lv.get("instruction_en",
                      lv.get("instruction_vi", lv["target_object_name"]))

    def current_prompt(self, eid: str) -> list:
        ep = self._episodes[eid]
        return build_messages_from_spec(
            self.joints, self.target_desc(eid),
            ep.obs.wrist_image, ep.obs.stand_image)

    def current_level_id(self, eid: str) -> str:
        return self._episodes[eid].level["level_id"]

    # ------------------------------------------------------------------
    def score_and_advance(self, eid: str, completions: list[str]) -> ScoreResult:
        """Chấm G completions cho step hiện tại của episode rồi tiến môi trường."""
        ep = self._episodes[eid]
        obs = ep.obs

        # 1) format gate
        valid_mask, angles_list, fmt = [], [], []
        for text in completions:
            res = check_format(text)
            valid_mask.append(res.ok)
            fmt.append(self.format_weight if res.ok else 0.0)
            angles_list.append(res.angles if res.ok else None)

        # 2) branch chỉ lệnh hợp lệ
        valid_idx = [i for i, ok in enumerate(valid_mask) if ok]
        branches: list[Branch] = []
        if valid_idx:
            branches = ep.env.branch([angles_list[i] for i in valid_idx])

        # 3) geometric reward
        before = _measure(obs.measure)
        geo = [0.0] * len(completions)
        branch_of: dict[int, Branch] = {}
        for k, ci in enumerate(valid_idx):
            br = branches[k]
            branch_of[ci] = br
            geo[ci] = geometric_reward(before, _measure(br.measure),
                                       weights=self.geo_weights,
                                       obj_speed_after=br.obj_speed)

        total = [fmt[i] + geo[i] for i in range(len(completions))]

        # 4) chọn winner + commit (advance)
        winner_idx, advanced, finished, success = None, False, False, False
        if valid_idx:
            winner_idx = max(valid_idx, key=lambda i: geo[i])
            next_obs = ep.env.commit(branch_of[winner_idx].branch_id)
            ep.obs = next_obs
            advanced = True
            success = bool(next_obs.done)          # vật đã vào giỏ
            if next_obs.done or next_obs.step_idx >= next_obs.max_steps:
                finished = True
        else:
            # cả G đều sai format: không tiến được -> coi như hỏng, reset màn.
            finished = True

        result = ScoreResult(
            rewards=total, format_rewards=fmt, geometric_rewards=geo,
            valid_mask=valid_mask, winner_idx=winner_idx, advanced=advanced,
            finished=finished,
            info={"level_id": ep.level["level_id"], "step_idx": obs.step_idx,
                  "n_valid": len(valid_idx), "done": ep.obs.done,
                  "success": success, "max_steps": obs.max_steps,
                  # số đo trạng thái SAU khi commit winner (None nếu không advance)
                  "winner_measure": (dict(ep.obs.measure) if advanced else None)})

        # Đính ảnh trạng thái winner (nếu bật) — phải lấy TRƯỚC khi reset episode.
        # ep.obs = trạng thái sau commit nếu advanced, ngược lại là khung đang kẹt.
        if self.attach_images:
            result.info["images"] = {"wrist": ep.obs.wrist_image,
                                     "stand": ep.obs.stand_image}

        if finished:
            self._reset_episode(eid)
        return result

    def close(self):
        for ep in self._episodes.values():
            ep.env.close()
