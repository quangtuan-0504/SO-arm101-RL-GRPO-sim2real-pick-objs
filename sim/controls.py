"""Ánh xạ bàn phím để điều khiển cánh tay (giữ phím để chuyển động liên tục)."""
import glfw

# Tốc độ thay đổi mục tiêu khớp (radian/giây) khi giữ phím
JOINT_RATE = 1.2
GRIPPER_RATE = 2.0

# Mỗi phím -> (tên_khớp, dấu). Giữ phím để di chuyển liên tục.
HELD_KEYS = {
    glfw.KEY_A: ("shoulder_pan", +1),
    glfw.KEY_D: ("shoulder_pan", -1),
    glfw.KEY_W: ("shoulder_lift", +1),
    glfw.KEY_S: ("shoulder_lift", -1),
    glfw.KEY_Q: ("elbow_flex", +1),
    glfw.KEY_E: ("elbow_flex", -1),
    glfw.KEY_R: ("wrist_flex", +1),
    glfw.KEY_F: ("wrist_flex", -1),
    glfw.KEY_Z: ("wrist_roll", +1),
    glfw.KEY_X: ("wrist_roll", -1),
    glfw.KEY_O: ("gripper", +1),   # mở kẹp
    glfw.KEY_C: ("gripper", -1),   # đóng kẹp
}

HELP_LINES = [
    "DIEU KHIEN CANH TAY (giu phim):",
    "  A / D : xoay than (shoulder_pan)",
    "  W / S : vai len/xuong (shoulder_lift)",
    "  Q / E : khuyu tay (elbow_flex)",
    "  R / F : co tay gap (wrist_flex)",
    "  Z / X : co tay xoay (wrist_roll)",
    "  O / C : mo / dong kep (gripper)",
    "  SPACE : mo-dong kep nhanh",
    "  T     : nhap goc quay bang so",
    "  TAB   : chon khop de nhap goc",
    "  H     : ve tu the home",
    "  BACKSPACE : reset vat the",
    "  P     : chup man hinh cua so",
    "  ESC   : thoat",
    "Chuot (cua so Operator): keo=xoay, lan=zoom",
]


def apply_held_keys(window, sim, dt):
    """Đọc trạng thái phím đang giữ và cập nhật điều khiển khớp."""
    for key, (joint, sign) in HELD_KEYS.items():
        if glfw.get_key(window, key) == glfw.PRESS:
            rate = GRIPPER_RATE if joint == "gripper" else JOINT_RATE
            sim.nudge_ctrl(joint, sign * rate * dt)


def apply_held_keys_any(windows, sim, dt):
    """Như apply_held_keys nhưng kiểm tra nhiều cửa sổ.

    Phím được coi là "đang giữ" nếu nó được nhấn ở BẤT KỲ cửa sổ nào đang
    focus; mỗi khớp chỉ được cập nhật một lần để không tăng tốc gấp đôi.
    """
    for key, (joint, sign) in HELD_KEYS.items():
        pressed = any(glfw.get_key(w, key) == glfw.PRESS for w in windows)
        if pressed:
            rate = GRIPPER_RATE if joint == "gripper" else JOINT_RATE
            sim.nudge_ctrl(joint, sign * rate * dt)
