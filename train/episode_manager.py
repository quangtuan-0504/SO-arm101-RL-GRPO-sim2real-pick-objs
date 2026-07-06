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
    format_rewards: list[float]    # 0.0 nếu hợp lệ, -invalid_format_penalty nếu sai
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
                 invalid_format_penalty: float = 1.0,
                 img_w: int = 480, img_h: int = 360,
                 timeout: float = 180.0, seed: int = 0,
                 attach_images: bool = False,
                 stall_penalty: float = 0.5, stall_eps: float = 1e-2,
                 curriculum: bool = False, curriculum_episodes: int = 200):
        self.base_url = base_url
        self.levels = levels
        self.joints = joints_spec
        self.group_size = group_size
        self.geo_weights = geo_weights or GeoWeights()
        # Format là GATE CỨNG, KHÔNG phải điểm cộng: completion sai cú pháp nhận
        # reward âm cố định (-invalid_format_penalty) và KHÔNG được branch vào
        # game. Completion hợp lệ CHỈ chấm bằng geometric (PBRS) -> tín hiệu học
        # phản ánh chất lượng chuyển động, không bị điểm format lấn át/wash-out.
        self.invalid_format_penalty = invalid_format_penalty
        # Anti-stall: phạt completion ra góc gần y hệt pose hiện tại (copy
        # home/example/pose cũ -> tay không di chuyển). Tạo phương sai trong
        # nhóm GRPO để gradient ưu tiên "đổi góc" hơn "đứng yên".
        self.stall_penalty = stall_penalty
        self.stall_eps = stall_eps
        self.img_w, self.img_h = img_w, img_h
        self.timeout = timeout
        self.attach_images = attach_images  # đính ảnh winner vào info để lưu đĩa
        self.rng = random.Random(seed)
        self._episodes: dict[str, _Episode] = {}
        self._order: list[str] = []

        # --- Curriculum: học từ dễ (ít vật) lên khó (nhiều vật) ---
        # Xếp level thành các "tier" theo độ khó = SỐ VẬT trong map. Đầu huấn
        # luyện chỉ lấy tier dễ nhất (map 1 vật), rồi mở dần các tier khó hơn
        # theo tiến trình (số episode đã bắt đầu / ``curriculum_episodes``).
        self.curriculum = curriculum
        self.curriculum_episodes = max(1, curriculum_episodes)
        self._episode_count = 0          # số episode đã bắt đầu (spawn + reset)
        self._last_unlocked = 0          # để log khi mở thêm tier
        by_diff: dict[int, list] = {}
        for lv in levels:
            by_diff.setdefault(len(lv.get("objects", [])), []).append(lv)
        # Danh sách tier theo độ khó tăng dần; mỗi tier = list level cùng số vật.
        self._tiers: list[list] = [by_diff[d] for d in sorted(by_diff)]
        if self.curriculum:
            summary = ", ".join(
                f"{len(t)}lvl@{len(t[0].get('objects', []))}vật"
                for t in self._tiers)
            print(f"[EpisodeManager] curriculum BẬT: {len(self._tiers)} tier "
                  f"({summary}); mở hết sau ~{self.curriculum_episodes} episode.")

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
    def _n_unlocked_tiers(self) -> int:
        """Số tier độ khó đang được phép lấy mẫu (theo tiến trình curriculum).

        frac = episode đã bắt đầu / curriculum_episodes (kẹp [0,1]).
          frac = 0        -> 1 tier (chỉ map dễ nhất)
          frac >= 1       -> mở hết tier
        Tăng tuyến tính, mỗi bước mở thêm 1 tier khó hơn.
        """
        n = len(self._tiers)
        if n <= 1:
            return n
        frac = min(1.0, self._episode_count / self.curriculum_episodes)
        return max(1, min(n, 1 + int(frac * (n - 1) + 1e-9)))

    def _pick_level(self) -> dict:
        if not self.curriculum or len(self._tiers) <= 1:
            return self.rng.choice(self.levels)
        k = self._n_unlocked_tiers()
        if k != self._last_unlocked:
            self._last_unlocked = k
            diffs = [len(self._tiers[i][0].get("objects", [])) for i in range(k)]
            print(f"[curriculum] episode={self._episode_count}: mở {k}/"
                  f"{len(self._tiers)} tier (độ khó <= {max(diffs)} vật).")
        pool = [lv for tier in self._tiers[:k] for lv in tier]
        return self.rng.choice(pool)

    def _spawn(self, eid: str):
        env = EnvClient(self.base_url, timeout=self.timeout)
        self._episode_count += 1
        level = self._pick_level()
        obs = env.reset(level["level_id"], img_w=self.img_w, img_h=self.img_h)
        self._episodes[eid] = _Episode(env=env, level=level, obs=obs)
        if eid not in self._order:
            self._order.append(eid)

    def _reset_episode(self, eid: str):
        ep = self._episodes[eid]
        ep.env.close()
        self._episode_count += 1
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
            ep.obs.wrist_image, ep.obs.stand_image,
            current_angles=ep.obs.ctrl_angles)

    def current_level_id(self, eid: str) -> str:
        return self._episodes[eid].level["level_id"]

    # ------------------------------------------------------------------
    def score_and_advance(self, eid: str, completions: list[str]) -> ScoreResult:
        """Chấm G completions cho step hiện tại của episode rồi tiến môi trường."""
        ep = self._episodes[eid]
        obs = ep.obs

        # 1) format gate (CỨNG): hợp lệ -> 0.0 (không cộng gì, chỉ mở cửa cho
        # geometric chấm); sai cú pháp -> -invalid_format_penalty và không branch.
        valid_mask, angles_list, fmt = [], [], []
        for text in completions:
            res = check_format(text)
            valid_mask.append(res.ok)
            fmt.append(0.0 if res.ok else -self.invalid_format_penalty)
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

        # total = geometric cho completion hợp lệ (fmt=0), = -penalty cho sai
        # format (geo=0). Format KHÔNG còn cộng điểm dương vào total.
        total = [fmt[i] + geo[i] for i in range(len(completions))]

        # 3b) anti-stall: phạt completion ra góc ~ y hệt pose hiện tại.
        # Tạo phương sai trong nhóm GRPO -> gradient đẩy model RA KHỎI việc copy
        # home/example/pose cũ (nguyên nhân tay không di chuyển).
        cur = obs.ctrl_angles or {}
        n_stalled = 0
        for i in valid_idx:
            ang = angles_list[i] or {}
            deltas = [abs(float(ang[n]) - float(cur.get(n, 0.0)))
                      for n in ang if n != "gripper" and n in cur]
            if deltas and max(deltas) < self.stall_eps:
                total[i] -= self.stall_penalty
                n_stalled += 1
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
                  "n_stalled": n_stalled,
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
