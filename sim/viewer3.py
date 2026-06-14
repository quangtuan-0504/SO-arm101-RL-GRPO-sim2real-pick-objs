"""Cửa sổ GLFW hiển thị đồng thời 3 khung nhìn:
   - Operator  (camera tự do, xoay bằng chuột)  -- khung lớn phía trên
   - Camera 1  (gắn trên cổ tay)                 -- dưới trái
   - Camera 2  (gắn trên giá đỡ)                 -- dưới phải
Đồng thời nhận điều khiển bàn phím cho cánh tay.
"""
from __future__ import annotations

import time
import numpy as np
import glfw
import mujoco

from . import controls
from .scene_builder import JOINT_NAMES


class ThreeViewApp:
    def __init__(self, sim, config: dict):
        self.sim = sim
        self.model = sim.model
        self.data = sim.data

        wcfg = config.get("window", {})
        self.win_w = int(wcfg.get("width", 1280))
        self.win_h = int(wcfg.get("height", 800))

        self._init_window()
        self._init_mujoco_render(config)
        self._init_input_state()

    # ----------------------------------------------------------------
    def _init_window(self):
        if not glfw.init():
            raise RuntimeError("Không khởi tạo được GLFW")
        # Dùng context OpenGL legacy mặc định (phù hợp với renderer của MuJoCo)
        self.window = glfw.create_window(
            self.win_w, self.win_h, "SO-101 Pick & Place - 3 Views", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Không tạo được cửa sổ GLFW")
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

    def _init_mujoco_render(self, config):
        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
        self.opt = mujoco.MjvOption()
        self.context = mujoco.MjrContext(
            self.model, mujoco.mjtFontScale.mjFONTSCALE_150)

        # Camera người điều khiển (tự do)
        op = config.get("operator_camera", {})
        self.cam_operator = mujoco.MjvCamera()
        self.cam_operator.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam_operator.azimuth = float(op.get("azimuth", 135))
        self.cam_operator.elevation = float(op.get("elevation", -25))
        self.cam_operator.distance = float(op.get("distance", 0.9))
        self.cam_operator.lookat = np.array(op.get("look_at", [0.12, 0.0, 0.12]))

        # Camera cố định 1 & 2
        self.cam_wrist = self._fixed_camera("wrist_cam")
        self.cam_stand = self._fixed_camera("stand_cam")

    def _fixed_camera(self, name):
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        cam.fixedcamid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        return cam

    def _init_input_state(self):
        self.button_left = False
        self.button_right = False
        self.button_middle = False
        self.last_x = 0.0
        self.last_y = 0.0

        glfw.set_key_callback(self.window, self._on_key)
        glfw.set_mouse_button_callback(self.window, self._on_mouse_button)
        glfw.set_cursor_pos_callback(self.window, self._on_cursor)
        glfw.set_scroll_callback(self.window, self._on_scroll)

    # ----------------------------------------------------------------
    #  Bố cục các khung nhìn (toạ độ pixel framebuffer, gốc dưới-trái)
    # ----------------------------------------------------------------
    def _layout(self, fb_w, fb_h):
        op_h = int(fb_h * 0.6)
        cam_h = fb_h - op_h
        half = fb_w // 2
        operator = mujoco.MjrRect(0, cam_h, fb_w, op_h)
        cam1 = mujoco.MjrRect(0, 0, half, cam_h)
        cam2 = mujoco.MjrRect(half, 0, fb_w - half, cam_h)
        return operator, cam1, cam2

    def _cursor_in_operator(self, x, y):
        """x,y theo toạ độ cửa sổ (gốc trên-trái). Operator = 60% phía trên."""
        win_w, win_h = glfw.get_window_size(self.window)
        return y < win_h * 0.6

    # ----------------------------------------------------------------
    #  Callbacks
    # ----------------------------------------------------------------
    def _on_key(self, window, key, scancode, action, mods):
        if action != glfw.PRESS and action != glfw.REPEAT:
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_H:
            self.sim.reset_arm()
        elif key == glfw.KEY_BACKSPACE:
            self.sim.reset()
        elif key == glfw.KEY_SPACE and action == glfw.PRESS:
            # mở/đóng kẹp nhanh
            aid = self.sim.act_ids["gripper"]
            lo, hi = self.sim.ctrl_range[aid]
            cur = self.data.ctrl[aid]
            self.data.ctrl[aid] = lo if cur > (lo + hi) / 2 else hi

    def _on_mouse_button(self, window, button, action, mods):
        self.button_left = glfw.get_mouse_button(
            window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        self.button_right = glfw.get_mouse_button(
            window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        self.button_middle = glfw.get_mouse_button(
            window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
        self.last_x, self.last_y = glfw.get_cursor_pos(window)

    def _on_cursor(self, window, xpos, ypos):
        dx = xpos - self.last_x
        dy = ypos - self.last_y
        self.last_x, self.last_y = xpos, ypos

        if not (self.button_left or self.button_right or self.button_middle):
            return
        if not self._cursor_in_operator(xpos, ypos):
            return

        _, win_h = glfw.get_window_size(window)
        if self.button_left:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_V
        elif self.button_right:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_V
        else:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM
        mujoco.mjv_moveCamera(
            self.model, action, dx / win_h, dy / win_h,
            self.scene, self.cam_operator)

    def _on_scroll(self, window, xoffset, yoffset):
        x, y = glfw.get_cursor_pos(window)
        if not self._cursor_in_operator(x, y):
            return
        mujoco.mjv_moveCamera(
            self.model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0.0, -0.05 * yoffset,
            self.scene, self.cam_operator)

    # ----------------------------------------------------------------
    #  Vẽ một khung nhìn
    # ----------------------------------------------------------------
    def _render_view(self, cam, viewport, title):
        mujoco.mjv_updateScene(
            self.model, self.data, self.opt, None, cam,
            mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        mujoco.mjr_render(viewport, self.scene, self.context)
        mujoco.mjr_overlay(
            mujoco.mjtFont.mjFONT_NORMAL,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            viewport, title, "", self.context)

    def _render_help(self, viewport):
        text = "\n".join(controls.HELP_LINES)
        mujoco.mjr_overlay(
            mujoco.mjtFont.mjFONT_NORMAL,
            mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
            viewport, text, "", self.context)
        # giá trị mục tiêu các khớp
        vals = "\n".join(f"{n}: {self.sim.get_ctrl(n):+.2f}" for n in JOINT_NAMES)
        mujoco.mjr_overlay(
            mujoco.mjtFont.mjFONT_NORMAL,
            mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT,
            viewport, "MUC TIEU KHOP\n" + vals, "", self.context)

    # ----------------------------------------------------------------
    def run(self):
        dt = self.sim.dt
        steps_per_frame = max(1, round((1.0 / 60.0) / dt))
        try:
            while not glfw.window_should_close(self.window):
                controls.apply_held_keys(self.window, self.sim, steps_per_frame * dt)
                self.sim.step(steps_per_frame)

                fb_w, fb_h = glfw.get_framebuffer_size(self.window)
                op_rect, c1_rect, c2_rect = self._layout(fb_w, fb_h)

                self._render_view(self.cam_operator, op_rect, "OPERATOR (nguoi dieu khien)")
                self._render_help(op_rect)
                self._render_view(self.cam_wrist, c1_rect, "CAMERA 1 - Co tay (Wrist)")
                self._render_view(self.cam_stand, c2_rect, "CAMERA 2 - Gia do (Stand)")

                glfw.swap_buffers(self.window)
                glfw.poll_events()
        finally:
            self.context.free()
            glfw.terminate()
