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

        # Tên các vật thể (body) từ config — để đo độ cao nghỉ.
        self.object_names = [o.get("name", f"object_{i}")
                             for i, o in enumerate(config.get("objects", []))]

        # Lưu trạng thái ban đầu của các vật để reset
        self._initial_qpos = None

        self.reset()
        self._initial_qpos = self.data.qpos.copy()

        # Độ cao nghỉ ban đầu (z) của từng vật — mốc để phát hiện "nhấc lên".
        self._object_rest_z = {}
        for name in self.object_names:
            try:
                self._object_rest_z[name] = float(self.body_pos(name)[2])
            except KeyError:
                pass

    def object_rest_z(self, name: str) -> float:
        """Độ cao z lúc nghỉ ban đầu của vật (mốc phát hiện nhấc lên)."""
        return self._object_rest_z.get(name, 0.0)

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

    # ------------------------------------------------------------------
    #  Snapshot / restore trạng thái — cốt lõi để BRANCH nhiều nhánh.
    #  Từ một trạng thái i, lưu lại bằng get_state(), chạy thử lệnh j,
    #  rồi set_state() để quay về i và thử lệnh j+1. Nhờ vậy game có thể
    #  sinh ra G trạng thái i+1 khác nhau từ cùng một trạng thái gốc.
    # ------------------------------------------------------------------
    def get_state(self) -> dict:
        """Chụp toàn bộ trạng thái động học của mô phỏng (deep copy)."""
        return {
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "ctrl": self.data.ctrl.copy(),
            "act": self.data.act.copy() if self.data.act.size else None,
            "time": float(self.data.time),
        }

    def set_state(self, state: dict):
        """Khôi phục trạng thái đã lưu bởi ``get_state`` và forward lại."""
        self.data.qpos[:] = state["qpos"]
        self.data.qvel[:] = state["qvel"]
        self.data.ctrl[:] = state["ctrl"]
        if state.get("act") is not None and self.data.act.size:
            self.data.act[:] = state["act"]
        self.data.time = state["time"]
        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------------------------
    #  Accessor hình học — dùng cho reward và phát hiện thành công.
    # ------------------------------------------------------------------
    def site_pos(self, site_name: str) -> np.ndarray:
        """Toạ độ thế giới (x,y,z) của một site."""
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if sid < 0:
            raise KeyError(f"Không có site tên {site_name!r}")
        return self.data.site_xpos[sid].copy()

    def body_pos(self, body_name: str) -> np.ndarray:
        """Toạ độ thế giới (x,y,z) của tâm một body."""
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if bid < 0:
            raise KeyError(f"Không có body tên {body_name!r}")
        return self.data.xpos[bid].copy()

    def grasp_point(self) -> np.ndarray:
        """Toạ độ thế giới của điểm gắp (site gripperframe)."""
        return self.site_pos("gripperframe")

    def object_speed(self, name: str) -> float:
        """Độ lớn vận tốc tịnh tiến của một vật (m/s), đo từ freejoint qvel."""
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT,
                                f"{name}_free")
        if jid < 0:
            return 0.0
        adr = self.model.jnt_dofadr[jid]
        v = self.data.qvel[adr:adr + 3]
        return float(np.linalg.norm(v))
