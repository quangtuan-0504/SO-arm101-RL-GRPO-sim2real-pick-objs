"""Sinh prompt cho VLM (Qwen-VL) để dự đoán góc quay điều khiển SO-101.

Module này tạo ra cặp (system_prompt, user_prompt) + cấu trúc messages sẵn
sàng gửi cho Qwen-VL. Giới hạn góc của từng khớp được đọc ĐỘNG từ
``Simulation`` để prompt luôn khớp với mô phỏng, tránh lệch thông số.

Cách dùng nhanh
---------------
    from sim.simulation import Simulation
    from vlm.prompt_builder import build_messages
    import yaml

    cfg = yaml.safe_load(open("config.yaml"))
    sim = Simulation(cfg)
    messages = build_messages(
        sim,
        target_desc="khối lập phương màu đỏ (cube_red)",
        wrist_img_b64=...,   # ảnh camera 1 dạng base64 (data URL)
        stand_img_b64=...,   # ảnh camera 2 dạng base64 (data URL)
    )
    # messages -> đưa vào client OpenAI-compatible của Qwen-VL
"""
from __future__ import annotations
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
import yaml
from sim.simulation import Simulation


import json

from sim.scene_builder import JOINT_NAMES

# Mô tả ý nghĩa + chiều dương của từng khớp (để VLM suy luận đúng hướng).
JOINT_DESCRIPTIONS = {
    "shoulder_pan":  "Xoay toàn thân cánh tay quanh trục thẳng đứng (z). "
                     "Giá trị + xoay thân sang một bên, - xoay sang bên kia. "
                     "Dùng để nhắm cánh tay theo phương ngang về phía vật.",
    "shoulder_lift": "Nâng/hạ vai (đoạn cánh tay trên). Giá trị + nâng vai lên, "
                     "- hạ vai xuống gần mặt sàn.",
    "elbow_flex":    "Gập/duỗi khuỷu tay. Điều chỉnh tầm với (xa/gần) và độ cao "
                     "của đầu kẹp.",
    "wrist_flex":    "Gập cổ tay lên/xuống. Dùng để chỉnh phương tiếp cận của "
                     "kẹp (gắp từ trên xuống hay từ ngang vào).",
    "wrist_roll":    "Xoay cổ tay quanh trục kẹp. Dùng để căn hướng của hai má "
                     "kẹp cho khớp với chiều dài/hướng của vật.",
    "gripper":       "Độ mở của kẹp. Giá trị LỚN = mở rộng, giá trị NHỎ (gần "
                     "giới hạn dưới) = đóng/kẹp chặt.",
}


def _joint_table_from_spec(joints: list) -> str:
    """Bảng giới hạn góc (radian) + home từ danh sách spec [{name,lo,hi,home}]."""
    lines = [
        "| # | Tên khớp | Giới hạn (radian) | Home | Mô tả |",
        "|---|----------|-------------------|------|-------|",
    ]
    for i, j in enumerate(joints, 1):
        name = j["name"]
        lines.append(
            f"| {i} | `{name}` | [{j['lo']:.3f}, {j['hi']:.3f}] | "
            f"{j['home']:.2f} | {JOINT_DESCRIPTIONS[name]} |"
        )
    return "\n".join(lines)


def _output_schema_example_from_spec(joints: list) -> str:
    """Ví dụ JSON đầu ra hợp lệ, dùng home pose từ spec làm minh hoạ."""
    angles = {j["name"]: round(float(j["home"]), 3) for j in joints}
    example = {
        "reasoning": "Mô tả ngắn gọn vị trí vật và giỏ trong 2 ảnh, đầu kẹp "
                     "đang ở đâu so với chúng, và bước di chuyển kế tiếp.",
        "target_object": "cube_red",
        "phase": "approach",
        "gripper_action": "open",
        "joint_angles_rad": angles,
        "done": False,
        "confidence": 0.0,
    }
    return json.dumps(example, ensure_ascii=False, indent=2)


def _joint_table(sim) -> str:
    """Bảng giới hạn góc (radian) + tư thế home đọc động từ sim."""
    home = sim.config.get("arm", {}).get("home_pose", {})
    joints = []
    for name in JOINT_NAMES:
        lo, hi = sim.get_ctrl_range(name)
        joints.append({"name": name, "lo": lo, "hi": hi,
                       "home": float(home.get(name, 0.0))})
    return _joint_table_from_spec(joints)


def _output_schema_example(sim) -> str:
    """Ví dụ JSON đầu ra hợp lệ, dùng chính home pose làm minh hoạ."""
    home = sim.config.get("arm", {}).get("home_pose", {})
    joints = [{"name": n, "home": float(home.get(n, 0.0))} for n in JOINT_NAMES]
    return _output_schema_example_from_spec(joints)


def build_system_prompt() -> str:
    """System prompt: vai trò + quy tắc bất biến của VLM."""
    return "\n".join([
        "Bạn là bộ điều khiển thị giác (vision-based controller) cho cánh tay "
        "robot 6 bậc tự do SO-101 trong môi trường mô phỏng MuJoCo. Nhiệm vụ "
        "của bạn là QUAN SÁT hai ảnh camera mô tả hiện trạng, rồi DỰ ĐOÁN góc "
        "quay mục tiêu (radian) cho 6 khớp để cánh tay GẮP đúng vật thể được "
        "yêu cầu rồi MANG VÀ THẢ vật đó VÀO GIỎ.",
        "",
        "Toàn bộ nhiệm vụ là một chu trình gắp-và-thả (pick-and-place):",
        "tiếp cận vật -> căn chỉnh -> kẹp -> nâng lên -> di chuyển tới phía "
        "trên giỏ -> mở kẹp thả vật vào giỏ. Chỉ báo done=true khi vật đã nằm "
        "trong giỏ.",
        "",
        "Nguyên tắc bắt buộc:",
        "1. Chỉ suy luận dựa trên những gì NHÌN THẤY trong 2 ảnh, không bịa "
        "thông tin không quan sát được.",
        "2. Mọi góc đầu ra phải là radian và NẰM TRONG giới hạn cho phép của "
        "từng khớp. Tuyệt đối không vượt biên.",
        "3. Điều khiển theo VÒNG KÍN: mỗi lần chỉ đề xuất bước di chuyển an "
        "toàn, nhỏ vừa phải tiến gần mục tiêu hiện tại của pha.",
        "4. Chỉ ĐÓNG kẹp khi đầu kẹp đã thẳng hàng và đủ gần vật trong ảnh.",
        "5. Sau khi đã kẹp được vật thì GIỮ kẹp đóng (gripper_action=\"hold\") "
        "trong suốt lúc nâng và di chuyển; chỉ MỞ kẹp khi đầu kẹp đã ở PHÍA "
        "TRÊN lòng giỏ để thả vật vào đúng giỏ.",
        "6. Đầu ra DUY NHẤT là một đối tượng JSON hợp lệ, không kèm văn bản nào "
        "khác, không markdown, không khối ```.",
    ])


def build_user_prompt(sim, target_desc: str) -> str:
    """User prompt: mục tiêu cụ thể + mô tả cơ cấu + format đầu ra (từ sim)."""
    home = sim.config.get("arm", {}).get("home_pose", {})
    joints = []
    for name in JOINT_NAMES:
        lo, hi = sim.get_ctrl_range(name)
        joints.append({"name": name, "lo": lo, "hi": hi,
                       "home": float(home.get(name, 0.0))})
    return build_user_prompt_from_spec(joints, target_desc)


def build_user_prompt_from_spec(joints: list, target_desc: str) -> str:
    """User prompt từ danh sách spec khớp [{name,lo,hi,home}] (không cần sim)."""
    joints_csv = ", ".join(f'"{j["name"]}"' for j in joints)
    return "\n".join([
        "# MỤC TIÊU",
        f"Điều khiển cánh tay SO-101 để GẮP **{target_desc}** rồi MANG và THẢ "
        "vật đó VÀO GIỎ. Nhiệm vụ hoàn thành khi vật đã nằm gọn trong giỏ.",
        "",
        "# BỐI CẢNH 2 CAMERA",
        "Bạn nhận được 2 ảnh theo thứ tự:",
        "- Ảnh 1 — CAMERA CỔ TAY (wrist_cam): gắn cứng trên kẹp, nhìn cận cảnh "
        "vùng kẹp gắp. Dùng để căn chính xác đầu kẹp so với vật (trái/phải, "
        "trước/sau, đã thẳng hàng chưa, đã đủ gần để đóng kẹp chưa).",
        "- Ảnh 2 — CAMERA GIÁ ĐỠ (stand_cam): đặt cố định bên cạnh, nhìn ngang "
        "toàn cảnh. Dùng để ước lượng tư thế tổng thể của cánh tay, độ cao đầu "
        "kẹp, khoảng cách thô tới vật VÀ tới giỏ.",
        "",
        "Bạn KHÔNG được cấp toạ độ của vật hay của giỏ. Hãy tự ĐỊNH VỊ chúng "
        "trong không gian dựa hoàn toàn vào những gì quan sát được trong 2 ảnh "
        "(vị trí trái/phải, xa/gần, cao/thấp của vật, của giỏ và của đầu kẹp).",
        "",
        "# CƠ CẤU CHẤP HÀNH (6 khớp position-controlled, đơn vị RADIAN)",
        "Mỗi khớp là một động cơ vị trí: bạn đặt GÓC MỤC TIÊU tuyệt đối, bộ "
        "điều khiển PID nội bộ sẽ đưa khớp tới góc đó. Tư thế \"home\" là tư "
        "thế nghỉ an toàn để tham chiếu.",
        "",
        _joint_table_from_spec(joints),
        "",
        "Quy ước về kẹp (`gripper`): giá trị càng LỚN càng MỞ rộng, càng NHỎ "
        "(về phía giới hạn dưới) càng ĐÓNG chặt.",
        "",
        "# QUY TRÌNH GẮP & THẢ VÀO GIỎ (các pha)",
        '- "approach": di chuyển đầu kẹp lại gần vật, kẹp đang MỞ.',
        '- "align": tinh chỉnh shoulder_pan / wrist_roll / wrist_flex để đầu '
        "kẹp thẳng hàng và bao quanh vật.",
        '- "grasp": đầu kẹp đã ôm vật -> ĐÓNG kẹp (gripper_action="close").',
        '- "lift": nâng shoulder_lift / elbow_flex để nhấc vật lên khỏi sàn, '
        'GIỮ kẹp đóng (gripper_action="hold").',
        '- "carry": xoay shoulder_pan và chỉnh tầm với để mang vật tới PHÍA '
        'TRÊN lòng giỏ, vẫn GIỮ kẹp đóng (gripper_action="hold").',
        '- "release": đầu kẹp đã ở trên lòng giỏ và đủ cao -> MỞ kẹp '
        '(gripper_action="open") cho vật rơi vào giỏ. Khi vật đã vào giỏ, đặt '
        "done=true.",
        "Chọn đúng pha hiện tại dựa trên những gì thấy trong ảnh. Không bỏ qua "
        "pha; chỉ chuyển pha khi điều kiện của pha hiện tại đã đạt.",
        "",
        "# ĐỊNH DẠNG ĐẦU RA (chỉ JSON, không thêm gì khác)",
        "Trả về đúng cấu trúc sau:",
        '- "reasoning": (string) suy luận ngắn gọn từ 2 ảnh: vị trí vật, vị trí '
        "giỏ, vị trí đầu kẹp và bước kế tiếp.",
        '- "target_object": (string) tên/định danh vật đang nhắm tới.',
        '- "phase": một trong ["approach","align","grasp","lift","carry",'
        '"release"].',
        '- "gripper_action": "open" | "close" | "hold".',
        '- "joint_angles_rad": (object) góc mục tiêu radian cho ĐỦ 6 khớp: '
        f"{joints_csv}.",
        '- "done": (bool) true KHI VÀ CHỈ KHI vật đã được thả nằm trong giỏ.',
        '- "confidence": (float 0..1) mức độ chắc chắn.',
        "",
        "Ví dụ một đầu ra hợp lệ:",
        _output_schema_example_from_spec(joints),
        "",
        "Bây giờ hãy phân tích 2 ảnh và xuất JSON điều khiển cho bước tiếp theo.",
    ])


def build_messages(sim, target_desc: str,
                   wrist_img_b64: str, stand_img_b64: str) -> list:
    """Tạo messages OpenAI-compatible (Qwen-VL) gồm 2 ảnh + prompt.

    ``wrist_img_b64`` / ``stand_img_b64`` là data URL, ví dụ:
    ``"data:image/png;base64,<...>"``.
    """
    return _assemble_messages(build_user_prompt(sim, target_desc),
                              wrist_img_b64, stand_img_b64)


def build_messages_from_spec(joints: list, target_desc: str,
                             wrist_img_b64: str, stand_img_b64: str) -> list:
    """Như ``build_messages`` nhưng dùng spec khớp tĩnh (cho server GPU).

    ``joints`` = [{name, lo, hi, home}] (đọc từ ``dataset/robot_spec.json``).
    """
    return _assemble_messages(build_user_prompt_from_spec(joints, target_desc),
                              wrist_img_b64, stand_img_b64)


def _assemble_messages(user_prompt: str, wrist_img_b64: str,
                       stand_img_b64: str) -> list:
    return [
        {"role": "system", "content": build_system_prompt()},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "text", "text": "Ảnh 1 — CAMERA CỔ TAY (wrist_cam):"},
                {"type": "image_url", "image_url": {"url": wrist_img_b64}},
                {"type": "text", "text": "Ảnh 2 — CAMERA GIÁ ĐỠ (stand_cam):"},
                {"type": "image_url", "image_url": {"url": stand_img_b64}},
            ],
        },
    ]

if __name__ == "__main__":

    cfg = yaml.safe_load(open("config.yaml"))
    sim = Simulation(cfg)

    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(sim, target_desc="khối lập phương màu đỏ (cube_red)")

    print("=== SYSTEM PROMPT ===")
    print(system_prompt)
    print("\n=== USER PROMPT ===")
    print(user_prompt)
