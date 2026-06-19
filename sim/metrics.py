"""Đo đạc hình học từ trạng thái MuJoCo — dùng cho reward và phát hiện thành công.

Tất cả đại lượng ở đây lấy TRỰC TIẾP từ toạ độ thật 3D trong mô phỏng (privileged
info). Chúng chỉ được dùng lúc HUẤN LUYỆN để chấm điểm/đo tiến độ; policy (VLM)
vẫn chỉ nhìn ảnh. Đây là setup hợp lệ và phổ biến: privileged reward + vision-only
policy.

Các hàm chính:
  - ``dist``            : khoảng cách Euclid giữa 2 điểm.
  - ``is_grasped``      : vật có đang được kẹp & nhấc lên không (suy từ trạng thái,
                          KHÔNG dựa vào nhãn pha do model tự khai).
  - ``obj_in_basket``   : vật đã nằm gọn trong lòng giỏ chưa.
  - ``measure_state``   : gói toàn bộ đại lượng đo cho 1 vật mục tiêu.
  - ``potential``       : hàm thế năng Φ cho potential-based reward shaping (PBRS).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def dist(a, b) -> float:
    """Khoảng cách Euclid giữa 2 toạ độ 3D."""
    return float(np.linalg.norm(np.asarray(a, float) - np.asarray(b, float)))


# ----------------------------------------------------------------------
#  Thông tin giỏ suy từ config (khớp scene_builder._add_basket)
# ----------------------------------------------------------------------
@dataclass
class BasketGeometry:
    center: np.ndarray      # tâm giỏ (x, y, z=đáy) theo config basket.pos
    inner_x: float          # nửa lòng giỏ theo x
    inner_y: float          # nửa lòng giỏ theo y
    height: float           # chiều cao thành giỏ
    floor_thickness: float

    @property
    def rim_z(self) -> float:
        """Độ cao miệng giỏ (mặt trên thành) tính từ sàn."""
        return self.center[2] + self.floor_thickness + 2.0 * self.height

    @property
    def inner_floor_z(self) -> float:
        """Độ cao mặt trong đáy giỏ."""
        return self.center[2] + self.floor_thickness


def basket_geometry(config: dict) -> BasketGeometry | None:
    """Trích hình học giỏ từ config. Trả None nếu giỏ không bật."""
    b = config.get("basket", {})
    if not b.get("enabled", False):
        return None
    pos = np.asarray(b.get("pos", [0.0, 0.20, 0.0]), float)
    inner = b.get("inner_size", [0.06, 0.06, 0.05])
    return BasketGeometry(
        center=pos,
        inner_x=float(inner[0]),
        inner_y=float(inner[1]),
        height=float(inner[2]),
        floor_thickness=float(b.get("floor_thickness", 0.005)),
    )


# ----------------------------------------------------------------------
#  Phát hiện trạng thái khách quan
# ----------------------------------------------------------------------
def is_grasped(sim, target_name: str, *,
               near_thresh: float = 0.05, lift_thresh: float = 0.03) -> bool:
    """Vật có đang được kẹp và nhấc lên không.

    Suy từ điều kiện khách quan, KHÔNG từ nhãn model:
      - đầu kẹp đủ gần tâm vật (``near_thresh`` mét), VÀ
      - vật đã được nhấc cao hơn vị trí nghỉ ban đầu một khoảng ``lift_thresh``.
    Mốc nghỉ ban đầu lấy từ ``sim`` (z lúc khởi tạo vật).
    """
    obj = sim.body_pos(target_name)
    grasp = sim.grasp_point()
    near = dist(obj, grasp) <= near_thresh
    rest_z = sim.object_rest_z(target_name)
    lifted = (obj[2] - rest_z) >= lift_thresh
    return bool(near and lifted)


def obj_in_basket(sim, target_name: str, basket: BasketGeometry | None,
                  *, margin: float = 0.0) -> bool:
    """Vật đã nằm gọn trong lòng giỏ chưa (xy trong khung, z dưới miệng giỏ)."""
    if basket is None:
        return False
    obj = sim.body_pos(target_name)
    dx = abs(obj[0] - basket.center[0])
    dy = abs(obj[1] - basket.center[1])
    inside_xy = (dx <= basket.inner_x + margin) and (dy <= basket.inner_y + margin)
    below_rim = obj[2] <= basket.rim_z
    above_floor = obj[2] >= basket.center[2]
    return bool(inside_xy and below_rim and above_floor)


# ----------------------------------------------------------------------
#  Gói đo + hàm thế năng cho PBRS
# ----------------------------------------------------------------------
@dataclass
class StateMeasure:
    d_grip_obj: float       # khoảng cách đầu kẹp ↔ vật
    d_obj_basket: float     # khoảng cách vật ↔ tâm giỏ
    grasped: bool
    in_basket: bool
    obj_z: float


def measure_state(sim, target_name: str,
                  basket: BasketGeometry | None) -> StateMeasure:
    """Đo toàn bộ đại lượng cho 1 vật mục tiêu trong trạng thái hiện tại."""
    obj = sim.body_pos(target_name)
    grasp = sim.grasp_point()
    grasped = is_grasped(sim, target_name)
    if basket is not None:
        d_basket = dist(obj, basket.center)
    else:
        d_basket = 0.0
    return StateMeasure(
        d_grip_obj=dist(grasp, obj),
        d_obj_basket=d_basket,
        grasped=grasped,
        in_basket=obj_in_basket(sim, target_name, basket),
        obj_z=float(obj[2]),
    )


# Trọng số mặc định cho hàm thế năng. Có thể chỉnh khi tinh reward.
PHI_WEIGHTS = {
    "approach": 1.0,    # gần vật hơn = tốt (pha approach)
    "grasped": 1.0,     # mốc rời rạc: kẹp được vật
    "carry": 1.0,       # khi đã kẹp: gần giỏ hơn = tốt (pha carry)
    "in_basket": 2.0,   # mốc cuối: thả vào giỏ
}


def potential(m: StateMeasure, weights: dict | None = None) -> float:
    """Hàm thế năng Φ cho PBRS.

    Φ tăng dần khi tiến gần mục tiêu cuối. Việc chuyển từ "thưởng gần vật"
    sang "thưởng gần giỏ" được kích hoạt bởi sự kiện ``grasped`` (đo khách
    quan), KHÔNG bởi nhãn pha do model khai, nên model không thể gian lận.
    """
    w = {**PHI_WEIGHTS, **(weights or {})}
    phi = w["approach"] * (-m.d_grip_obj)
    if m.grasped:
        phi += w["grasped"]
        phi += w["carry"] * (-m.d_obj_basket)
    if m.in_basket:
        phi += w["in_basket"]
    return float(phi)
