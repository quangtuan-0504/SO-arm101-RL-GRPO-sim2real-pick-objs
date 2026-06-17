"""Dựng cảnh mô phỏng từ file cấu hình bằng MuJoCo MjSpec.

Model robot SO-101 gốc (models/so101/scene.xml) được giữ NGUYÊN, không
chỉnh sửa. Mọi thứ thuộc "môi trường trò chơi" (vật thể, camera, bàn, vị
trí đế) được thêm vào động dựa trên config.
"""
from __future__ import annotations

from pathlib import Path
import math
import numpy as np
import mujoco

# Thư mục gốc của dự án
ROOT = Path(__file__).resolve().parent.parent
BASE_SCENE = ROOT / "models" / "so101" / "scene.xml"

# Tên 6 khớp/cơ cấu chấp hành của SO-101 (theo đúng model gốc)
JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# Điểm gắp (đầu kẹp) trong hệ toạ độ cục bộ của thân "gripper".
# Lấy từ site "gripperframe" trong model gốc so101_new_calib.xml.
GRASP_POINT_LOCAL = np.array([-0.0079, -0.0002, -0.0981])

# Màu mặc định cho từng KHÂU của cánh tay (theo tên body trong model gốc).
# Mỗi khâu một màu tương phản, tách biệt với sàn xanh và các khối
# đỏ/lục/lam để VLM dễ phân biệt khâu và đọc góc khớp. Có thể ghi đè
# trong config.yaml qua khoá "link_colors".
DEFAULT_LINK_COLORS = {
    "base": [0.35, 0.35, 0.38, 1.0],            # đế: xám trung tính
    "shoulder": [1.00, 0.55, 0.00, 1.0],        # shoulder_pan: cam
    "upper_arm": [1.00, 0.85, 0.10, 1.0],       # shoulder_lift: vàng
    "lower_arm": [0.65, 0.20, 0.80, 1.0],       # elbow_flex: tím
    "wrist": [0.10, 0.80, 0.85, 1.0],           # wrist_flex: lục lam (cyan)
    "gripper": [0.95, 0.95, 0.97, 1.0],         # wrist_roll: trắng
    "moving_jaw_so101_v1": [1.00, 0.45, 0.70, 1.0],  # gripper (ngàm động): hồng
}

_SHAPE_MAP = {
    "box": mujoco.mjtGeom.mjGEOM_BOX,
    "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
    "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
}


def _euler_deg_to_quat(euler_deg):
    """Chuyển góc Euler (độ, thứ tự xyz) -> quaternion (w,x,y,z)."""
    r = np.radians(euler_deg)
    quat = np.zeros(4)
    seq = "xyz"
    mujoco.mju_euler2Quat(quat, r, seq)
    return quat


def _look_at_quat(cam_pos, target, up=(0.0, 0.0, 1.0)):
    """Quaternion để camera tại cam_pos nhìn về điểm target.

    Quy ước camera MuJoCo: trục -Z hướng nhìn, +Y hướng lên.
    """
    cam_pos = np.asarray(cam_pos, float)
    target = np.asarray(target, float)
    up = np.asarray(up, float)
    forward = target - cam_pos
    n = np.linalg.norm(forward)
    if n < 1e-9:
        forward = np.array([1.0, 0.0, 0.0])
    else:
        forward /= n
    z_axis = -forward
    x_axis = np.cross(up, z_axis)
    if np.linalg.norm(x_axis) < 1e-6:
        x_axis = np.array([1.0, 0.0, 0.0])
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    mat = np.column_stack([x_axis, y_axis, z_axis]).flatten()
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, mat)
    return quat


def build_spec(config: dict) -> mujoco.MjSpec:
    """Tạo MjSpec hoàn chỉnh từ cấu hình."""
    spec = mujoco.MjSpec.from_file(str(BASE_SCENE))

    _configure_physics(spec)
    _place_arm(spec, config.get("arm", {}))
    _color_links(spec, config.get("link_colors", {}))
    _add_wrist_camera(spec, config.get("wrist_camera", {}))
    _add_stand_camera(spec, config.get("stand_camera", {}))
    _add_table(spec, config.get("table", {}))
    _add_basket(spec, config.get("basket", {}))
    _add_objects(spec, config.get("objects", []))

    return spec


def _configure_physics(spec):
    """Tinh chỉnh solver tiếp xúc để kẹp gắp giữ chặt vật, không bị trượt.

    Đây là phần quan trọng nhất để khắc phục hiện tượng vật tuột khỏi kẹp:
    - impratio cao: làm ma sát "cứng" hơn nhiều, vật không trượt khi bị ép.
    - cone elliptic: mô hình nón ma sát chính xác hơn cho việc kẹp.
    - integrator implicitfast: ổn định hơn Euler khi có tiếp xúc giữ lâu.
    - tăng số vòng lặp solver để tiếp xúc hội tụ chắc chắn.
    """
    opt = spec.option
    opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    # Tỉ lệ trở kháng pháp tuyến/tiếp tuyến: càng cao càng ít trượt khi kẹp.
    opt.impratio = 50.0
    opt.iterations = 150
    opt.ls_iterations = 50
    # Ngưỡng hội tụ chặt hơn cho ràng buộc tiếp xúc.
    opt.tolerance = 1e-10


def _place_arm(spec, arm_cfg):
    base = spec.body("base")
    base.pos = list(arm_cfg.get("base_pos", [0.0, 0.0, 0.0]))
    base.quat = list(_euler_deg_to_quat(arm_cfg.get("base_euler_deg", [0, 0, 0])))


def _color_links(spec, color_cfg):
    """Tô màu từng KHÂU của cánh tay để VLM dễ phân biệt và đọc góc khớp.

    Gán một màu riêng cho mọi geom thuộc cùng một body (khâu). Đặt rgba
    trực tiếp trên geom sẽ ghi đè màu của material gốc (material chỉ được
    dùng khi geom giữ rgba mặc định). Bảng màu lấy từ DEFAULT_LINK_COLORS,
    có thể ghi đè theo tên khâu qua config "link_colors".
    """
    colors = dict(DEFAULT_LINK_COLORS)
    for name, rgba in (color_cfg or {}).items():
        colors[name] = list(rgba)

    for body_name, rgba in colors.items():
        try:
            body = spec.body(body_name)
        except (KeyError, ValueError):
            continue  # bỏ qua nếu model không có khâu này
        if body is None:
            continue
        for geom in body.geoms:
            geom.rgba = list(rgba)


def _add_wrist_camera(spec, cam_cfg):
    """Camera 1 - gắn cứng vào thân gripper, nhìn vào vùng kẹp gắp."""
    back = float(cam_cfg.get("back", 0.08))
    up = float(cam_cfg.get("up", 0.06))
    fovy = float(cam_cfg.get("fovy", 70))

    # Vị trí mount trong hệ cục bộ gripper: lùi theo -y, nâng theo +z
    cam_pos = GRASP_POINT_LOCAL + np.array([0.0, -back, up])
    quat = _look_at_quat(cam_pos, GRASP_POINT_LOCAL)

    grip = spec.body("gripper")
    cam = grip.add_camera()
    cam.name = "wrist_cam"
    cam.pos = list(cam_pos)
    cam.quat = list(quat)
    cam.fovy = fovy


def _add_stand_camera(spec, cam_cfg):
    """Camera 2 - đặt trên giá đỡ tại vị trí cố định trong thế giới."""
    pos = cam_cfg.get("pos", [0.45, 0.40, 0.40])
    look_at = cam_cfg.get("look_at", [0.15, 0.0, 0.06])
    fovy = float(cam_cfg.get("fovy", 55))

    cam = spec.worldbody.add_camera()
    cam.name = "stand_cam"
    cam.pos = list(pos)
    cam.quat = list(_look_at_quat(pos, look_at))
    cam.fovy = fovy


def _add_table(spec, table_cfg):
    if not table_cfg.get("enabled", False):
        return
    body = spec.worldbody.add_body()
    body.name = "table"
    body.pos = list(table_cfg.get("pos", [0.18, 0.0, 0.0]))
    geom = body.add_geom()
    geom.type = mujoco.mjtGeom.mjGEOM_BOX
    geom.size = list(table_cfg.get("size", [0.25, 0.25, 0.01]))
    geom.rgba = list(table_cfg.get("rgba", [0.55, 0.4, 0.25, 1.0]))


def _add_basket(spec, basket_cfg):
    """Thêm một cái giỏ tĩnh để robot thả vật vào.

    Giỏ được dựng từ các hộp (box): 1 đáy + 4 thành. Đây là một thân tĩnh
    (không có freejoint) nên giỏ đứng yên tại vị trí cấu hình. Lòng giỏ rỗng
    để vật rơi vào và được giữ lại bởi 4 thành.
    """
    if not basket_cfg.get("enabled", False):
        return

    pos = list(basket_cfg.get("pos", [0.0, 0.20, 0.0]))
    # inner = nửa kích thước lòng giỏ (x, y) và chiều cao thành (z)
    inner = list(basket_cfg.get("inner_size", [0.06, 0.06, 0.05]))
    wall = float(basket_cfg.get("wall_thickness", 0.005))
    floor_t = float(basket_cfg.get("floor_thickness", 0.005))
    rgba = list(basket_cfg.get("rgba", [0.6, 0.45, 0.2, 1.0]))

    ix, iy, h = inner[0], inner[1], inner[2]

    body = spec.worldbody.add_body()
    body.name = "basket"
    body.pos = pos

    def _add_wall(name, size, wpos):
        g = body.add_geom()
        g.name = f"basket_{name}"
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = list(size)
        g.pos = list(wpos)
        g.rgba = rgba
        g.condim = 3

    # Đáy giỏ: nằm sát mặt sàn, tâm ở độ cao floor_t
    _add_wall("floor", [ix + wall, iy + wall, floor_t], [0.0, 0.0, floor_t])

    # 4 thành: cao h, đặt phía trên đáy
    wall_zc = floor_t + h  # tâm thành theo z
    # Thành theo trục +x và -x (vuông góc trục x)
    _add_wall("wall_xp", [wall, iy + wall, h], [ix + wall, 0.0, wall_zc])
    _add_wall("wall_xn", [wall, iy + wall, h], [-(ix + wall), 0.0, wall_zc])
    # Thành theo trục +y và -y (vuông góc trục y)
    _add_wall("wall_yp", [ix + wall, wall, h], [0.0, iy + wall, wall_zc])
    _add_wall("wall_yn", [ix + wall, wall, h], [0.0, -(iy + wall), wall_zc])


def _add_objects(spec, objects_cfg):
    for i, obj in enumerate(objects_cfg):
        name = obj.get("name", f"object_{i}")
        shape = obj.get("shape", "box").lower()
        if shape not in _SHAPE_MAP:
            raise ValueError(f"Hình dạng không hỗ trợ: {shape} (object {name})")

        body = spec.worldbody.add_body()
        body.name = name
        body.pos = list(obj.get("pos", [0.18, 0.0, 0.02]))

        joint = body.add_freejoint()
        joint.name = f"{name}_free"

        geom = body.add_geom()
        geom.name = f"{name}_geom"
        geom.type = _SHAPE_MAP[shape]
        geom.size = _normalize_size(shape, obj.get("size", [0.02, 0.02, 0.02]))
        geom.rgba = list(obj.get("rgba", [0.8, 0.2, 0.2, 1.0]))
        # condim=6: bật ma sát trượt + xoắn + lăn để vật không xoay/lăn tuột
        # khỏi kẹp. (Với condim=3 mặc định, hai thành phần xoắn/lăn bị bỏ qua.)
        geom.condim = 6
        # friction = [trượt, xoắn, lăn]. Ma sát trượt cao + xoắn/lăn vừa đủ
        # để giữ chặt vật trong kẹp.
        geom.friction = [2.0, 0.1, 0.005]
        # solref/solimp: tiếp xúc cứng hơn, giảm độ lún để vật không bị "ép
        # lọt" rồi bật ra khỏi kẹp. solimp gần 1 = ràng buộc rất cứng.
        geom.solref = [0.005, 1.0]
        geom.solimp = [0.95, 0.99, 0.001, 0.5, 2.0]
        geom.mass = float(obj.get("mass", 0.02))


def _normalize_size(shape, size):
    """MjSpec yêu cầu mảng size 3 phần tử."""
    size = list(size)
    if shape == "sphere":
        return [size[0], 0.0, 0.0]
    if shape == "cylinder":
        return [size[0], size[1], 0.0]
    # box
    while len(size) < 3:
        size.append(size[-1])
    return size[:3]
