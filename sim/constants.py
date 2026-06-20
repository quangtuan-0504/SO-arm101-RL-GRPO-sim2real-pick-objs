"""Hằng số dùng chung — KHÔNG phụ thuộc mujoco.

Tách riêng để phía huấn luyện (server GPU, không cài mujoco) import được
``JOINT_NAMES`` mà không kéo theo ``import mujoco`` của ``scene_builder``.
"""
from __future__ import annotations

# Tên 6 khớp/cơ cấu chấp hành của SO-101 (theo đúng model gốc).
JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
