"""Hàm chấm điểm #1 — FORMAT (gate 0/1).

Kiểm tra đầu ra VLM có đúng cấu trúc JSON yêu cầu không. Trả 1.0 nếu hợp lệ,
0.0 nếu sai. Đầu ra 0 điểm sẽ KHÔNG được gửi cho game và KHÔNG được chấm tiếp
(tiết kiệm chi phí branch + judge).

Dùng lại ``parse_control_json`` trong ``vlm.qwen_client`` để chịu được khi model
bọc JSON trong ```json ... ```.
"""
from __future__ import annotations

from dataclasses import dataclass

from sim.constants import JOINT_NAMES
from vlm.qwen_client import parse_control_json

VALID_PHASES = {"approach", "align", "grasp", "lift", "carry", "release"}
VALID_GRIPPER = {"open", "close", "hold"}


@dataclass
class FormatResult:
    ok: bool
    reason: str
    parsed: dict | None
    angles: dict | None   # {joint_name: float} đủ 6 khớp nếu ok


def check_format(text: str) -> FormatResult:
    """Phân tích & kiểm tra format. Không ném lỗi — gói kết quả vào dataclass."""
    try:
        obj = parse_control_json(text)
    except Exception as e:
        return FormatResult(False, f"parse_error: {e}", None, None)

    if not isinstance(obj, dict):
        return FormatResult(False, "không phải object JSON", None, None)

    # Các khoá bắt buộc
    required = ["phase", "gripper_action", "joint_angles_rad", "done"]
    for k in required:
        if k not in obj:
            return FormatResult(False, f"thiếu khoá {k!r}", obj, None)

    if obj["phase"] not in VALID_PHASES:
        return FormatResult(False, f"phase không hợp lệ: {obj['phase']!r}", obj, None)
    if obj["gripper_action"] not in VALID_GRIPPER:
        return FormatResult(False,
                            f"gripper_action không hợp lệ: {obj['gripper_action']!r}",
                            obj, None)
    if not isinstance(obj["done"], bool):
        return FormatResult(False, "done phải là bool", obj, None)

    angles_raw = obj["joint_angles_rad"]
    if not isinstance(angles_raw, dict):
        return FormatResult(False, "joint_angles_rad phải là object", obj, None)

    angles = {}
    for name in JOINT_NAMES:
        if name not in angles_raw:
            return FormatResult(False, f"thiếu khớp {name!r}", obj, None)
        try:
            angles[name] = float(angles_raw[name])
        except (TypeError, ValueError):
            return FormatResult(False, f"góc khớp {name!r} không phải số", obj, None)

    return FormatResult(True, "ok", obj, angles)


def format_reward(text: str) -> float:
    """Điểm format: 1.0 nếu đúng cấu trúc, 0.0 nếu sai."""
    return 1.0 if check_format(text).ok else 0.0
