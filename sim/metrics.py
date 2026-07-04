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
    d_grip_obj: float       # khoảng cách đầu kẹp ↔ vật (3D)
    d_obj_basket: float     # khoảng cách vật ↔ tâm giỏ
    grasped: bool
    in_basket: bool
    obj_z: float
    # --- thành phần cho phễu tiền-kẹp (pre-grasp pose shaping) ---
    # Tách d_grip_obj thành lệch NGANG và chênh CAO để Φ biết "hướng" tiếp cận,
    # không chỉ biết khoảng cách trần. Mặc định 0.0 để dict cũ vẫn dựng được.
    d_xy: float = 0.0       # khoảng cách ngang (mặt phẳng xy) đầu kẹp ↔ vật
    dz: float = 0.0         # chênh cao đầu kẹp − vật (>0: đầu kẹp ở TRÊN vật)


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
    # Lệch ngang (xy) và chênh cao (z) cho phễu tiền-kẹp.
    d_xy = float(np.hypot(grasp[0] - obj[0], grasp[1] - obj[1]))
    dz = float(grasp[2] - obj[2])
    return StateMeasure(
        d_grip_obj=dist(grasp, obj),
        d_obj_basket=d_basket,
        grasped=grasped,
        in_basket=obj_in_basket(sim, target_name, basket),
        obj_z=float(obj[2]),
        d_xy=d_xy,
        dz=dz,
    )


# Trọng số mặc định cho hàm thế năng. Có thể chỉnh khi tinh reward.
PHI_WEIGHTS = {
    # --- pha APPROACH (chưa kẹp): PHỄU tiền-kẹp, kẹp từ TRÊN xuống ---
    # Bề mặt Φ được nắn thành cái "phễu": gradient tại mỗi điểm dẫn cánh tay
    # ĐI LÊN -> CANH NGANG trên vật -> HẠ THẲNG xuống, thay vì lao ngang vào
    # vật (tránh hất văng vật nằm sát ngàm). Hai cơ chế:
    #   (a) ĐÍCH kéo blend: khi còn lệch ngang -> kéo tới ĐIỂM CHỜ phía trên
    #       vật (nâng lên được thưởng); khi đã canh thẳng -> kéo thẳng tới vật
    #       (hạ xuống được thưởng).
    #   (b) PHẠT va chạm: tiếp cận ngang khi đầu kẹp còn THẤP và chưa canh ->
    #       bị phạt mạnh, ép cánh tay đi "đường trên".
    "approach": 1.0,      # lực kéo tới đích tiền-kẹp (điểm chờ -> dần xuống vật)
    "danger": 4.0,        # phạt tiếp cận ngang khi còn thấp & chưa canh (chống đâm)
    "h_standoff": 0.10,   # (m) độ cao điểm chờ phía trên vật
    "r_funnel": 0.08,     # (m) bán kính "đã canh thẳng" (blend đích & tắt phạt)
    "r_danger": 0.12,     # (m) bán kính vùng dễ va chạm khi đầu kẹp ở thấp
    # --- pha sau khi kẹp ---
    "grasped": 1.0,       # mốc rời rạc: kẹp được vật
    "carry": 1.0,         # khi đã kẹp: gần giỏ hơn = tốt (pha carry)
    "in_basket": 2.0,     # mốc cuối: thả vào giỏ
}


def potential(m: StateMeasure, weights: dict | None = None) -> float:
    """Hàm thế năng Φ cho PBRS (kèm PHỄU tiền-kẹp ở pha approach).

    Φ tăng dần khi tiến gần mục tiêu cuối. Việc chuyển từ pha approach sang
    pha carry được kích hoạt bởi sự kiện ``grasped`` (đo khách quan), KHÔNG
    bởi nhãn pha do model khai, nên model không thể gian lận.

    PHA APPROACH (chưa kẹp) thay khoảng cách trần ``-d_grip_obj`` bằng phễu:
        base   = na · d(đầu kẹp, ĐIỂM CHỜ trên vật) + al · d(đầu kẹp, VẬT)
        danger = relu(r_danger − d_xy) · lowness · na
        Φ_pre  = − approach · base − danger_w · danger
    với na = clip(d_xy/r_funnel, 0, 1)  (1 khi còn lệch ngang, 0 khi đã canh),
        al = 1 − na, lowness = clip(1 − dz/h_standoff, 0, 1).
    Khi còn lệch ngang (na≈1): đích kéo là điểm chờ phía TRÊN vật -> nâng lên
    được thưởng; đồng thời tiếp cận ngang lúc còn thấp bị ``danger`` phạt.
    Khi đã canh thẳng (na≈0): đích kéo là chính vật và phạt tắt -> hạ thẳng
    xuống được thưởng. Φ là hàm thuần của trạng thái -> giữ đảm bảo PBRS.
    """
    w = {**PHI_WEIGHTS, **(weights or {})}
    if not m.grasped:
        r_funnel = max(w["r_funnel"], 1e-6)
        h = max(w["h_standoff"], 1e-6)
        na = min(max(m.d_xy / r_funnel, 0.0), 1.0)   # 1: còn lệch ngang, 0: đã canh
        al = 1.0 - na
        lowness = min(max(1.0 - m.dz / h, 0.0), 1.0)  # 1: sát vật/thấp, 0: ở độ cao chờ
        d_to_standoff = float(np.hypot(m.d_xy, m.dz - h))  # đầu kẹp -> điểm chờ trên vật
        base = na * d_to_standoff + al * m.d_grip_obj
        danger = max(0.0, w["r_danger"] - m.d_xy) * lowness * na
        phi = -w["approach"] * base - w["danger"] * danger
    else:
        phi = w["grasped"] + w["carry"] * (-m.d_obj_basket)
    if m.in_basket:
        phi += w["in_basket"]
    return float(phi)
