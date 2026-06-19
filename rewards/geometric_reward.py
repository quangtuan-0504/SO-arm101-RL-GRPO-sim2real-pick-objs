"""Hàm chấm điểm #2 — GEOMETRIC (tín hiệu chính, PBRS).

Reward dày lấy TRỰC TIẾP từ toạ độ thật MuJoCo: chính xác, miễn phí, tức thời,
không nhiễu. Đây là tín hiệu chính giúp GRPO hội tụ; judge VLM (#3) chỉ bổ sung.

Cơ chế:
  - Reward chính = PBRS:  Φ(s_{i+1}) − Φ(s_i)  (không bóp méo chính sách tối ưu).
  - Mốc rời rạc (bonus) khi đạt ``grasped`` lần đầu và khi ``in_basket``.
  - Phạt nhẹ chống reward-hacking: đập vật văng (vận tốc lớn) hoặc đẩy vật ra
    khỏi vùng thao tác.

Việc chuyển "thưởng gần vật" -> "thưởng gần giỏ" do ``grasped`` (đo khách quan)
kích hoạt, KHÔNG do nhãn pha model tự khai.
"""
from __future__ import annotations

from dataclasses import dataclass

from sim.metrics import StateMeasure, potential


@dataclass
class GeoWeights:
    pbrs_scale: float = 10.0      # khuếch đại chênh lệch thế năng cho dễ học
    grasp_bonus: float = 1.0      # thưởng một lần khi vừa kẹp được vật
    basket_bonus: float = 5.0     # thưởng lớn khi vật vào giỏ (mốc cuối)
    fling_penalty: float = 1.0    # phạt nếu làm vật văng (vận tốc lớn)
    fling_speed: float = 1.0      # ngưỡng tốc độ coi là "đập" (m/s)


def geometric_reward(before: StateMeasure, after: StateMeasure,
                     *, weights: GeoWeights | None = None,
                     obj_speed_after: float = 0.0,
                     phi_weights: dict | None = None) -> float:
    """Điểm hình học cho 1 bước (từ trạng thái before -> after).

    ``obj_speed_after`` là độ lớn vận tốc của vật ở trạng thái sau (m/s), dùng
    để phạt hành vi đập/hất vật. Có thể để 0 nếu chưa đo.
    """
    w = weights or GeoWeights()

    # 1) PBRS: chênh lệch thế năng giữa hai trạng thái.
    reward = w.pbrs_scale * (potential(after, phi_weights) - potential(before, phi_weights))

    # 2) Mốc rời rạc: vừa kẹp được vật (chuyển False -> True).
    if after.grasped and not before.grasped:
        reward += w.grasp_bonus
    # Mất vật sau khi đã kẹp (tuột): phạt đối xứng để khỏi "kẹp rồi buông" ăn gian.
    if before.grasped and not after.grasped and not after.in_basket:
        reward -= w.grasp_bonus

    # 3) Mốc cuối: vật vào giỏ (chuyển False -> True).
    if after.in_basket and not before.in_basket:
        reward += w.basket_bonus

    # 4) Chống reward-hacking: phạt nếu vật bị văng nhanh (đập cho gần giỏ).
    if obj_speed_after > w.fling_speed:
        reward -= w.fling_penalty

    return float(reward)
