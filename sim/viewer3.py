"""Hiển thị mỗi camera trong MỘT cửa sổ GLFW riêng:
   - Operator  (camera tự do, xoay/zoom bằng chuột)
   - Camera 1  (gắn trên cổ tay - wrist_cam)
   - Camera 2  (gắn trên giá đỡ - stand_cam)
Mỗi cửa sổ có nút "Chup man hinh" (bấm chuột) và phím tắt P.
Các cửa sổ chia sẻ chung một OpenGL context nên dùng chung MjrContext.
Điều khiển bàn phím cho cánh tay hoạt động ở bất kỳ cửa sổ nào đang focus.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import glfw
import mujoco

from . import controls
from .scene_builder import JOINT_NAMES, ROOT

# Kích thước nút chụp màn hình (pixel framebuffer)
BTN_W = 150
BTN_H = 34
BTN_MARGIN = 10

# Kích thước hộp nhập góc quay (pixel framebuffer)
INPUT_W = 440
INPUT_H = 200


class CameraWindow:
    """Một cửa sổ GLFW gắn với một camera MuJoCo."""

    def __init__(self, app, title, cam, is_operator=False,
                 width=640, height=480, share=None):
        self.app = app
        self.title = title
        self.cam = cam
        self.is_operator = is_operator

        self.window = glfw.create_window(width, height, title, None, share)
        if not self.window:
            raise RuntimeError(f"Không tạo được cửa sổ GLFW: {title}")

        # Trạng thái chuột (cho operator) + cờ chụp ảnh
        self.button_left = False
        self.button_right = False
        self.button_middle = False
        self.last_x = 0.0
        self.last_y = 0.0
        self._want_shot = False
        self._btn_rect = None  # MjrRect của nút chụp, cập nhật mỗi khung
        self._activate_rect = None  # MjrRect nút "Hoat dong" (chỉ operator)

        glfw.set_key_callback(self.window, self._on_key)
        glfw.set_char_callback(self.window, self._on_char)
        glfw.set_mouse_button_callback(self.window, self._on_mouse_button)
        glfw.set_cursor_pos_callback(self.window, self._on_cursor)
        glfw.set_scroll_callback(self.window, self._on_scroll)

    # ------------------------------------------------------------------
    #  Vị trí nút chụp màn hình (góc trên-phải, toạ độ framebuffer)
    # ------------------------------------------------------------------
    def _button_rect(self, fb_w, fb_h):
        x = fb_w - BTN_W - BTN_MARGIN
        y = fb_h - BTN_H - BTN_MARGIN
        return mujoco.MjrRect(x, y, BTN_W, BTN_H)

    def _cursor_on_button(self, cx, cy):
        """cx,cy: toạ độ chuột trong cửa sổ (gốc trên-trái)."""
        return self._cursor_on_rect(cx, cy, self._btn_rect)

    def _cursor_on_rect(self, cx, cy, rect):
        """Kiểm tra con trỏ có nằm trong rect (toạ độ framebuffer) không."""
        if rect is None:
            return False
        win_w, win_h = glfw.get_window_size(self.window)
        fb_w, fb_h = glfw.get_framebuffer_size(self.window)
        if win_w == 0 or win_h == 0:
            return False
        sx, sy = fb_w / win_w, fb_h / win_h
        fx = cx * sx
        fy = fb_h - cy * sy  # đổi sang gốc dưới-trái như framebuffer
        r = rect
        return (r.left <= fx <= r.left + r.width and
                r.bottom <= fy <= r.bottom + r.height)

    # ------------------------------------------------------------------
    #  Callbacks
    # ------------------------------------------------------------------
    def _on_key(self, window, key, scancode, action, mods):
        if action != glfw.PRESS and action != glfw.REPEAT:
            return

        app = self.app

        # Khi đang ở chế độ nhập góc, các phím được dùng để soạn thảo số.
        if app.input_active:
            if key == glfw.KEY_ESCAPE:
                app.input_cancel()
            elif key in (glfw.KEY_ENTER, glfw.KEY_KP_ENTER):
                app.input_confirm_field()
            elif key == glfw.KEY_BACKSPACE:
                app.input_backspace()
            elif key == glfw.KEY_TAB:
                app.input_cycle_joint(-1 if mods & glfw.MOD_SHIFT else +1)
            elif key == glfw.KEY_G:
                app.input_activate()
            elif key == glfw.KEY_T:
                app.input_toggle()
            return

        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_T and action == glfw.PRESS:
            app.input_toggle()
        elif key == glfw.KEY_G and action == glfw.PRESS:
            app.input_activate()
        elif key == glfw.KEY_TAB:
            app.input_cycle_joint(-1 if mods & glfw.MOD_SHIFT else +1)
        elif key == glfw.KEY_P and action == glfw.PRESS:
            self._want_shot = True
        elif key == glfw.KEY_H:
            self.app.sim.reset_arm()
        elif key == glfw.KEY_BACKSPACE:
            self.app.sim.reset()
        elif key == glfw.KEY_SPACE and action == glfw.PRESS:
            sim = self.app.sim
            aid = sim.act_ids["gripper"]
            lo, hi = sim.ctrl_range[aid]
            cur = sim.data.ctrl[aid]
            sim.data.ctrl[aid] = lo if cur > (lo + hi) / 2 else hi

    def _on_char(self, window, codepoint):
        """Nhận ký tự khi soạn góc quay (chỉ giữ số, dấu chấm, dấu trừ)."""
        self.app.input_char(chr(codepoint))

    def _on_mouse_button(self, window, button, action, mods):
        self.button_left = glfw.get_mouse_button(
            window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        self.button_right = glfw.get_mouse_button(
            window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        self.button_middle = glfw.get_mouse_button(
            window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
        self.last_x, self.last_y = glfw.get_cursor_pos(window)

        # Bấm trái vào nút -> yêu cầu chụp màn hình
        if (button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS
                and self._cursor_on_button(self.last_x, self.last_y)):
            self._want_shot = True

        # Bấm trái vào nút "Hoat dong" -> áp dụng tất cả góc đã nhập
        if (button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS
                and self._cursor_on_rect(self.last_x, self.last_y,
                                         self._activate_rect)):
            self.app.input_activate()

    def _on_cursor(self, window, xpos, ypos):
        dx = xpos - self.last_x
        dy = ypos - self.last_y
        self.last_x, self.last_y = xpos, ypos

        if not self.is_operator:
            return  # camera cố định, không xoay
        if not (self.button_left or self.button_right or self.button_middle):
            return

        _, win_h = glfw.get_window_size(window)
        if win_h == 0:
            return
        if self.button_left:
            act = mujoco.mjtMouse.mjMOUSE_ROTATE_V
        elif self.button_right:
            act = mujoco.mjtMouse.mjMOUSE_MOVE_V
        else:
            act = mujoco.mjtMouse.mjMOUSE_ZOOM
        mujoco.mjv_moveCamera(
            self.app.model, act, dx / win_h, dy / win_h,
            self.app.scene, self.cam)

    def _on_scroll(self, window, xoffset, yoffset):
        if not self.is_operator:
            return
        mujoco.mjv_moveCamera(
            self.app.model, mujoco.mjtMouse.mjMOUSE_ZOOM,
            0.0, -0.05 * yoffset, self.app.scene, self.cam)

    # ------------------------------------------------------------------
    #  Vẽ cửa sổ
    # ------------------------------------------------------------------
    def render(self):
        glfw.make_context_current(self.window)
        fb_w, fb_h = glfw.get_framebuffer_size(self.window)
        if fb_w == 0 or fb_h == 0:
            return
        viewport = mujoco.MjrRect(0, 0, fb_w, fb_h)

        app = self.app
        mujoco.mjv_updateScene(
            app.model, app.data, app.opt, None, self.cam,
            mujoco.mjtCatBit.mjCAT_ALL, app.scene)
        mujoco.mjr_render(viewport, app.scene, app.context)

        # Tiêu đề + (với operator) phần trợ giúp
        mujoco.mjr_overlay(
            mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
            viewport, self.title, "", app.context)
        if self.is_operator:
            app.render_help(viewport)
            app.render_activate_button(self, viewport)

        # Chụp màn hình NỘI DUNG (trước khi vẽ nút) nếu được yêu cầu
        if self._want_shot:
            self._want_shot = False
            app.capture(self, fb_w, fb_h)

        # Vẽ nút chụp màn hình
        self._btn_rect = self._button_rect(fb_w, fb_h)
        mujoco.mjr_rectangle(self._btn_rect, 0.15, 0.15, 0.18, 0.85)
        mujoco.mjr_overlay(
            mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
            self._btn_rect, " Chup man hinh (P)", "", app.context)

        glfw.swap_buffers(self.window)


class ThreeViewApp:
    def __init__(self, sim, config: dict):
        self.sim = sim
        self.model = sim.model
        self.data = sim.data
        self.config = config

        self.shot_dir = ROOT / "screenshots"
        self.shot_dir.mkdir(exist_ok=True)

        # Trạng thái hộp nhập góc quay (điều khiển bằng bàn phím)
        self.input_active = False
        self.input_buffer = ""        # chuỗi đang gõ cho khớp đang chọn
        self.sel_joint = 0            # chỉ số khớp đang chọn trong JOINT_NAMES
        # Góc mục tiêu (độ) cho từng khớp, nhập sẵn rồi mới "Hoat dong"
        self.pending_deg = [0.0] * len(JOINT_NAMES)
        self.input_status = ""        # thông báo ngắn hiển thị trong hộp nhập

        self._init_glfw()
        self._init_cameras(config)
        self._init_render()
        self._init_windows(config)

    # ------------------------------------------------------------------
    def _init_glfw(self):
        if not glfw.init():
            raise RuntimeError("Không khởi tạo được GLFW")

    def _init_cameras(self, config):
        op = config.get("operator_camera", {})
        self.cam_operator = mujoco.MjvCamera()
        self.cam_operator.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam_operator.azimuth = float(op.get("azimuth", 135))
        self.cam_operator.elevation = float(op.get("elevation", -25))
        self.cam_operator.distance = float(op.get("distance", 0.9))
        self.cam_operator.lookat = np.array(op.get("look_at", [0.12, 0.0, 0.12]))

        self.cam_wrist = self._fixed_camera("wrist_cam")
        self.cam_stand = self._fixed_camera("stand_cam")

    def _fixed_camera(self, name):
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        cam.fixedcamid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        return cam

    def _init_windows(self, config):
        wcfg = config.get("window", {})
        ow = int(wcfg.get("operator_width", wcfg.get("width", 960)))
        oh = int(wcfg.get("operator_height", 640))
        cw = int(wcfg.get("camera_width", 560))
        ch = int(wcfg.get("camera_height", 440))

        # Cửa sổ đầu tiên (operator) là chủ context được chia sẻ
        self.operator_win = CameraWindow(
            self, "OPERATOR (nguoi dieu khien)", self.cam_operator,
            is_operator=True, width=ow, height=oh, share=None)

        share = self.operator_win.window
        self.windows = [
            self.operator_win,
            CameraWindow(self, "CAMERA 1 - Co tay (Wrist)", self.cam_wrist,
                         width=cw, height=ch, share=share),
            CameraWindow(self, "CAMERA 2 - Gia do (Stand)", self.cam_stand,
                         width=cw, height=ch, share=share),
        ]

    def _init_render(self):
        # Scene/option dùng chung cho mọi cửa sổ. MjrContext được tạo sau
        # (trong _build_context) khi cửa sổ chủ đã sẵn sàng để chia sẻ context.
        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
        self.opt = mujoco.MjvOption()
        self.context = None

    def _build_context(self):
        glfw.make_context_current(self.operator_win.window)
        glfw.swap_interval(1)
        self.context = mujoco.MjrContext(
            self.model, mujoco.mjtFontScale.mjFONTSCALE_150)

    # ------------------------------------------------------------------
    #  Hộp nhập góc quay: nhập sẵn cho mọi khớp, rồi bấm "Hoat dong"
    # ------------------------------------------------------------------
    def input_toggle(self):
        self.input_active = not self.input_active
        if self.input_active:
            # Khởi tạo bằng góc hiện tại để chỉ cần sửa khớp muốn đổi
            self.pending_deg = [math.degrees(self.sim.get_ctrl(n))
                                for n in JOINT_NAMES]
            self.input_buffer = self._fmt(self.pending_deg[self.sel_joint])
            self.input_status = "Nhap goc cho tung khop roi bam HOAT DONG"
        else:
            self.input_buffer = ""
            self.input_status = ""

    def input_cancel(self):
        self.input_active = False
        self.input_buffer = ""
        self.input_status = ""

    def _commit_buffer(self):
        """Lưu chuỗi đang gõ vào pending của khớp đang chọn (nếu hợp lệ)."""
        if self.input_buffer not in ("", "-", "."):
            try:
                self.pending_deg[self.sel_joint] = float(self.input_buffer)
            except ValueError:
                pass
        # Đồng bộ lại ô nhập với giá trị pending đã chuẩn hoá
        self.input_buffer = self._fmt(self.pending_deg[self.sel_joint])

    def input_confirm_field(self):
        """ENTER: chốt khớp hiện tại rồi chuyển sang khớp kế tiếp."""
        self._commit_buffer()
        self.input_cycle_joint(+1)

    def input_cycle_joint(self, step):
        self._commit_buffer()
        self.sel_joint = (self.sel_joint + step) % len(JOINT_NAMES)
        self.input_buffer = self._fmt(self.pending_deg[self.sel_joint])

    def input_char(self, ch):
        """Chỉ chấp nhận ký tự số khi đang ở chế độ nhập."""
        if not self.input_active:
            return
        if ch.isdigit():
            self.input_buffer += ch
        elif ch == "." and "." not in self.input_buffer:
            self.input_buffer += ch
        elif ch == "-" and self.input_buffer == "":
            self.input_buffer += ch

    def input_backspace(self):
        self.input_buffer = self.input_buffer[:-1]

    def input_activate(self):
        """Áp dụng tất cả góc đã nhập cho cánh tay cùng lúc."""
        if not self.input_active:
            return
        self._commit_buffer()
        for i, name in enumerate(JOINT_NAMES):
            self.sim.set_ctrl(name, math.radians(self.pending_deg[i]))
        self.input_status = "Da ap dung goc cho tat ca khop!"

    @staticmethod
    def _fmt(deg):
        return f"{deg:.1f}"

    # ------------------------------------------------------------------
    def render_help(self, viewport):
        text = "\n".join(controls.HELP_LINES)
        mujoco.mjr_overlay(
            mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
            viewport, text, "", self.context)

        # Mục tiêu khớp + khoảng giá trị cho phép (hiển thị theo độ)
        lines = []
        for i, n in enumerate(JOINT_NAMES):
            lo, hi = self.sim.get_ctrl_range(n)
            cur = self.sim.get_ctrl(n)
            marker = ">" if (self.input_active and i == self.sel_joint) else " "
            lines.append(
                f"{marker}{n:<13}: {math.degrees(cur):+7.1f}  "
                f"[{math.degrees(lo):+6.0f} .. {math.degrees(hi):+6.0f}]")
        vals = "\n".join(lines)
        mujoco.mjr_overlay(
            mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT,
            viewport, "MUC TIEU KHOP / KHOANG (do)\n" + vals, "", self.context)

        self._render_input_box(viewport)

    def _render_input_box(self, viewport):
        """Vẽ hộp nhập góc quay (tất cả khớp) ở phía trên cửa sổ operator."""
        if not self.input_active:
            hint = "Nhan T de nhap goc quay cho cac khop"
            lines = [hint]
            box_h = 34
        else:
            lines = ["NHAP GOC QUAY (do) - TAB/ENTER doi khop, ESC huy"]
            for i, n in enumerate(JOINT_NAMES):
                lo, hi = self.sim.get_ctrl_range(n)
                if i == self.sel_joint:
                    field = self.input_buffer + "_"
                    marker = ">"
                else:
                    field = self._fmt(self.pending_deg[i])
                    marker = " "
                lines.append(
                    f"{marker}{n:<13}: {field:<8} "
                    f"[{math.degrees(lo):+.0f}..{math.degrees(hi):+.0f}]")
            if self.input_status:
                lines.append(self.input_status)
            box_h = INPUT_H

        text = "\n".join(lines)
        x = max(BTN_MARGIN, (viewport.width - INPUT_W) // 2)
        y = viewport.height - box_h - BTN_MARGIN
        rect = mujoco.MjrRect(x, y, INPUT_W, box_h)
        if self.input_active:
            mujoco.mjr_rectangle(rect, 0.10, 0.20, 0.30, 0.88)
        else:
            mujoco.mjr_rectangle(rect, 0.15, 0.15, 0.18, 0.80)
        mujoco.mjr_overlay(
            mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
            rect, text, "", self.context)

    def render_activate_button(self, win, viewport):
        """Vẽ nút 'Hoat dong' dưới hộp nhập; trả về rect đã vẽ (hoặc None)."""
        if not self.input_active:
            win._activate_rect = None
            return
        x = max(BTN_MARGIN, (viewport.width - INPUT_W) // 2)
        y = viewport.height - INPUT_H - BTN_MARGIN - BTN_H - 6
        rect = mujoco.MjrRect(x, y, INPUT_W, BTN_H)
        mujoco.mjr_rectangle(rect, 0.12, 0.45, 0.15, 0.95)
        mujoco.mjr_overlay(
            mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
            rect, " >> HOAT DONG (G) - quay tat ca khop <<", "", self.context)
        win._activate_rect = rect

    def capture(self, win: CameraWindow, fb_w, fb_h):
        """Đọc pixel khung hiện tại và lưu thành PNG."""
        rgb = np.empty((fb_h, fb_w, 3), dtype=np.uint8)
        viewport = mujoco.MjrRect(0, 0, fb_w, fb_h)
        mujoco.mjr_readPixels(rgb, None, viewport, self.context)
        rgb = np.flipud(rgb)  # OpenGL gốc dưới-trái -> ảnh gốc trên-trái

        slug = _slugify(win.title)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = self.shot_dir / f"{slug}_{stamp}.png"
        _save_png(path, rgb)
        print(f"[SO-101] Da chup man hinh: {path}")

    # ------------------------------------------------------------------
    def run(self, on_frame=None):
        """Vòng lặp GLFW chính.

        ``on_frame``: callback tuỳ chọn ``fn(app)`` gọi mỗi khung sau khi bước
        mô phỏng (dùng cho driver VLM chạy song song).
        """
        self._build_context()

        dt = self.sim.dt
        steps_per_frame = max(1, round((1.0 / 60.0) / dt))
        windows = [w.window for w in self.windows]
        try:
            while not glfw.window_should_close(self.operator_win.window):
                if not self.input_active:
                    controls.apply_held_keys_any(windows, self.sim,
                                                 steps_per_frame * dt)
                self.sim.step(steps_per_frame)

                if on_frame is not None:
                    on_frame(self)

                for w in self.windows:
                    if not glfw.window_should_close(w.window):
                        w.render()

                glfw.poll_events()
        finally:
            if self.context is not None:
                self.context.free()
            glfw.terminate()


# ----------------------------------------------------------------------
#  Tiện ích
# ----------------------------------------------------------------------
def _slugify(text):
    keep = []
    for ch in text.lower():
        if ch.isalnum():
            keep.append(ch)
        elif ch in " -_":
            keep.append("_")
    slug = "".join(keep).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "view"


def _save_png(path: Path, rgb: np.ndarray):
    try:
        from PIL import Image
        Image.fromarray(rgb).save(path)
    except ImportError:
        import struct, zlib
        h, w, _ = rgb.shape
        raw = bytearray()
        for y in range(h):
            raw.append(0)
            raw.extend(rgb[y].tobytes())

        def chunk(typ, data):
            c = typ + data
            return struct.pack(">I", len(data)) + c + struct.pack(
                ">I", zlib.crc32(c) & 0xffffffff)

        png = b"\x89PNG\r\n\x1a\n"
        png += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        png += chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        png += chunk(b"IEND", b"")
        Path(path).write_bytes(png)
