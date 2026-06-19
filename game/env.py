"""Môi trường một MÀN game, bọc quanh ``Simulation``.

Trách nhiệm:
  - ``reset``  : dựng cảnh map của level, về home, trả quan sát ban đầu.
  - ``branch`` : từ trạng thái i hiện tại, chạy thử G lệnh -> G trạng thái i+1
                 (dùng snapshot/restore để mọi nhánh xuất phát từ cùng i), trả
                 ảnh + số đo từng nhánh. KHÔNG thay đổi trạng thái hiện tại.
  - ``commit`` : áp 1 lệnh được chọn (winner) -> tiến hẳn sang trạng thái i+1,
                 tăng bộ đếm step.

Game "dumb": chỉ step + chụp ảnh + đo. KHÔNG build prompt, KHÔNG chấm điểm.
Số đo (StateMeasure) được trả kèm để service huấn luyện tính reward.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sim.simulation import Simulation
from sim.scene_builder import JOINT_NAMES
from sim.metrics import (BasketGeometry, basket_geometry, measure_state,
                         StateMeasure, obj_in_basket)
from vlm.render_utils import CameraRenderer


@dataclass
class BranchResult:
    """Kết quả chạy thử 1 lệnh từ trạng thái i."""
    applied_angles: dict           # góc thực sự áp dụng (đã clip)
    wrist_rgb: np.ndarray
    stand_rgb: np.ndarray
    measure: StateMeasure
    obj_speed: float
    done: bool                     # vật đã vào giỏ (thành công)
    state: dict                    # snapshot trạng thái i+1 (để commit winner)


@dataclass
class Observation:
    wrist_rgb: np.ndarray
    stand_rgb: np.ndarray
    ctrl_angles: dict              # góc mục tiêu hiện tại của 6 khớp
    measure: StateMeasure
    step_idx: int
    max_steps: int
    done: bool
    instruction_vi: str
    target_object_name: str


class GameEnv:
    def __init__(self, base_config: dict, level: dict, *,
                 img_w: int = 480, img_h: int = 360,
                 settle_seconds: float = 0.6):
        from dataset.generator import level_to_config

        self.level = level
        self.target = level["target_object_name"]
        self.max_steps = int(level.get("max_steps", 20))
        self.instruction = level.get("instruction_vi", "")

        self.config = level_to_config(base_config, level)
        self.sim = Simulation(self.config)
        self.basket: BasketGeometry | None = basket_geometry(self.config)
        self.renderer = CameraRenderer(self.sim.model, width=img_w, height=img_h)
        self.settle_steps = max(1, round(settle_seconds / self.sim.dt))
        self.step_idx = 0

    # ------------------------------------------------------------------
    def _render_pair(self):
        wrist = self.renderer.render(self.sim.data, "wrist_cam")
        stand = self.renderer.render(self.sim.data, "stand_cam")
        return wrist, stand

    def _apply_angles(self, angles: dict) -> dict:
        applied = {}
        for name in JOINT_NAMES:
            if name in angles:
                self.sim.set_ctrl(name, float(angles[name]))
                applied[name] = self.sim.get_ctrl(name)
        return applied

    def _current_ctrl(self) -> dict:
        return {n: self.sim.get_ctrl(n) for n in JOINT_NAMES}

    def _measure(self) -> StateMeasure:
        return measure_state(self.sim, self.target, self.basket)

    def _is_success(self) -> bool:
        return obj_in_basket(self.sim, self.target, self.basket)

    # ------------------------------------------------------------------
    def reset(self) -> Observation:
        self.sim.reset()
        self.step_idx = 0
        wrist, stand = self._render_pair()
        return Observation(
            wrist_rgb=wrist, stand_rgb=stand,
            ctrl_angles=self._current_ctrl(), measure=self._measure(),
            step_idx=self.step_idx, max_steps=self.max_steps,
            done=self._is_success(), instruction_vi=self.instruction,
            target_object_name=self.target,
        )

    def branch(self, actions: list[dict]) -> list[BranchResult]:
        """Chạy thử nhiều lệnh từ trạng thái i hiện tại (không thay đổi i)."""
        base = self.sim.get_state()
        results: list[BranchResult] = []
        for angles in actions:
            self.sim.set_state(base)
            applied = self._apply_angles(angles)
            self.sim.step(self.settle_steps)
            wrist, stand = self._render_pair()
            results.append(BranchResult(
                applied_angles=applied,
                wrist_rgb=wrist, stand_rgb=stand,
                measure=self._measure(),
                obj_speed=self.sim.object_speed(self.target),
                done=self._is_success(),
                state=self.sim.get_state(),
            ))
        # Khôi phục về trạng thái i: việc tiến bước do commit() quyết định.
        self.sim.set_state(base)
        return results

    def commit(self, branch_result: BranchResult) -> Observation:
        """Tiến hẳn sang trạng thái i+1 đã chọn (winner). Tăng step."""
        self.sim.set_state(branch_result.state)
        self.step_idx += 1
        wrist, stand = branch_result.wrist_rgb, branch_result.stand_rgb
        return Observation(
            wrist_rgb=wrist, stand_rgb=stand,
            ctrl_angles=self._current_ctrl(), measure=self._measure(),
            step_idx=self.step_idx, max_steps=self.max_steps,
            done=self._is_success(), instruction_vi=self.instruction,
            target_object_name=self.target,
        )

    def commit_angles(self, angles: dict) -> Observation:
        """Tiến bước bằng cách áp trực tiếp một bộ góc (không qua branch)."""
        self._apply_angles(angles)
        self.sim.step(self.settle_steps)
        self.step_idx += 1
        wrist, stand = self._render_pair()
        return Observation(
            wrist_rgb=wrist, stand_rgb=stand,
            ctrl_angles=self._current_ctrl(), measure=self._measure(),
            step_idx=self.step_idx, max_steps=self.max_steps,
            done=self._is_success(), instruction_vi=self.instruction,
            target_object_name=self.target,
        )

    def close(self):
        self.renderer.close()
