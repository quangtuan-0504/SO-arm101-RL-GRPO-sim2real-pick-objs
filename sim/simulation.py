"""Lõi mô phỏng: bọc MjModel/MjData, quản lý tư thế home và reset."""
from __future__ import annotations

import numpy as np
import mujoco

from .scene_builder import build_spec, JOINT_NAMES


class Simulation:
    def __init__(self, config: dict):
        self.config = config
        self.spec = build_spec(config)
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)

        # Ánh xạ tên khớp -> id actuator / địa chỉ qpos
        self.act_ids = {}
        self.qpos_adr = {}
        for name in JOINT_NAMES:
            self.act_ids[name] = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.qpos_adr[name] = self.model.jnt_qposadr[jid]

        # Giới hạn điều khiển của từng actuator
        self.ctrl_range = self.model.actuator_ctrlrange.copy()

        # Tư thế home từ config
        home = config.get("arm", {}).get("home_pose", {})
        self.home_ctrl = np.zeros(self.model.nu)
        for name in JOINT_NAMES:
            self.home_ctrl[self.act_ids[name]] = float(home.get(name, 0.0))

        # Lưu trạng thái ban đầu của các vật để reset
        self._initial_qpos = None

        self.reset()
        self._initial_qpos = self.data.qpos.copy()

    @property
    def dt(self) -> float:
        return self.model.opt.timestep

    def reset(self):
        """Đưa cánh tay về home và vật thể về vị trí ban đầu."""
        mujoco.mj_resetData(self.model, self.data)
        # Đặt khớp cánh tay đúng tư thế home (tránh rơi khi khởi động)
        for name in JOINT_NAMES:
            self.data.qpos[self.qpos_adr[name]] = self.home_ctrl[self.act_ids[name]]
        self.data.ctrl[:] = self.home_ctrl
        mujoco.mj_forward(self.model, self.data)

    def reset_arm(self):
        """Chỉ đưa cánh tay về home, giữ nguyên vật thể."""
        for name in JOINT_NAMES:
            self.data.qpos[self.qpos_adr[name]] = self.home_ctrl[self.act_ids[name]]
            self.data.qvel[self.model.jnt_dofadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)]] = 0.0
        self.data.ctrl[:] = self.home_ctrl
        mujoco.mj_forward(self.model, self.data)

    def step(self, n: int = 1):
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)

    def nudge_ctrl(self, joint_name: str, delta: float):
        """Tăng/giảm mục tiêu một khớp, kẹp trong giới hạn cho phép."""
        aid = self.act_ids[joint_name]
        lo, hi = self.ctrl_range[aid]
        self.data.ctrl[aid] = float(np.clip(self.data.ctrl[aid] + delta, lo, hi))

    def get_ctrl(self, joint_name: str) -> float:
        return float(self.data.ctrl[self.act_ids[joint_name]])

    def get_ctrl_range(self, joint_name: str):
        """Trả về (lo, hi) giới hạn điều khiển của khớp (radian)."""
        aid = self.act_ids[joint_name]
        lo, hi = self.ctrl_range[aid]
        return float(lo), float(hi)

    def set_ctrl(self, joint_name: str, value: float):
        """Đặt mục tiêu một khớp theo giá trị tuyệt đối, kẹp trong giới hạn."""
        aid = self.act_ids[joint_name]
        lo, hi = self.ctrl_range[aid]
        self.data.ctrl[aid] = float(np.clip(value, lo, hi))
