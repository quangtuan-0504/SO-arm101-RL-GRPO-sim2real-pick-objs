"""Hàm chấm điểm #3 — JUDGE bằng VLM self-host (vLLM). NÂNG CẤP SAU.

Đây là tuỳ chọn để bật KHI pipeline (format + geometric) đã chạy ổn. Judge dùng
CHÍNH base model self-host bằng vLLM để "nhìn" ảnh trạng thái i và các ảnh trạng
thái i+1 hợp lệ, rồi chấm chất lượng từng nhánh theo thang 0–100.

Lưu ý thiết kế (rút từ kinh nghiệm reward-collapse):
  - Yêu cầu judge so sánh TƯƠNG ĐỐI và chấm PHÂN BIỆT RÕ giữa các nhánh, tránh
    chấm gần bằng nhau làm advantage GRPO ≈ 0 (không học được).
  - Thang tuyệt đối không quan trọng (GRPO chuẩn hoá trong nhóm); tính phân biệt
    trong nhóm mới quan trọng.

Phần gọi model cần GPU/vLLM — chưa chạy trên Mac. ``build_judge_messages`` và
``parse_judge_scores`` là thuần Python, test được.
"""
from __future__ import annotations

import json
import re


def build_judge_messages(goal_desc: str, before_img: str,
                         branch_imgs: list[str]) -> list:
    """Dựng messages chấm điểm: ảnh i + danh sách ảnh i+1 nhánh + hướng dẫn.

    ``before_img`` và ``branch_imgs`` là data URL (PNG base64). Dùng ảnh
    stand_cam (toàn cảnh) là chính; có thể mở rộng kèm wrist_cam nếu cần.
    """
    n = len(branch_imgs)
    instr = "\n".join([
        "Bạn là GIÁM KHẢO đánh giá chất lượng điều khiển cánh tay robot SO-101.",
        f"MỤC TIÊU của màn: {goal_desc}",
        "",
        f"Ảnh ĐẦU là TRẠNG THÁI HIỆN TẠI (trước khi hành động). Sau đó là {n} "
        "ảnh KẾT QUẢ ứng với các phương án hành động khác nhau (đánh số 1..N).",
        "",
        "Hãy chấm mỗi phương án theo thang 0–100 dựa trên:",
        "  - Đầu kẹp/cánh tay tiến GẦN HƠN tới mục tiêu của bước hiện tại "
        "(vật nếu chưa kẹp; giỏ nếu đã kẹp) so với trạng thái hiện tại.",
        "  - Tư thế hợp lý để gắp/thả (căn hướng kẹp, độ cao phù hợp).",
        "  - KHÔNG làm vật văng ra ngoài hay đổ vỡ bố cục.",
        "",
        "QUAN TRỌNG: chấm PHÂN BIỆT RÕ giữa các phương án — phương án tốt hơn "
        "phải có điểm CAO HƠN HẲN, đừng cho điểm gần bằng nhau.",
        "",
        "CHỈ trả JSON đúng dạng:",
        '{"scores": [s1, s2, ...], "best": <số thứ tự phương án tốt nhất>}',
        f"với đúng {n} phần tử trong \"scores\" (mỗi phần tử 0–100).",
    ])

    content = [{"type": "text", "text": instr},
               {"type": "text", "text": "Ảnh TRẠNG THÁI HIỆN TẠI:"},
               {"type": "image_url", "image_url": {"url": before_img}}]
    for i, img in enumerate(branch_imgs, 1):
        content.append({"type": "text", "text": f"Phương án {i}:"})
        content.append({"type": "image_url", "image_url": {"url": img}})

    return [{"role": "user", "content": content}]


def parse_judge_scores(text: str, n: int) -> list[float]:
    """Trích danh sách điểm 0–100 từ phản hồi judge. Thiếu -> 0; chuẩn về [0,1]."""
    scores = [0.0] * n
    try:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        obj = json.loads(m.group(0)) if m else {}
        raw = obj.get("scores", [])
        for i in range(min(n, len(raw))):
            scores[i] = max(0.0, min(100.0, float(raw[i]))) / 100.0
    except Exception:
        pass
    return scores


class JudgeReward:
    """Chấm điểm bằng VLM self-host (vLLM OpenAI-compatible). GPU-only."""

    def __init__(self, base_url: str, model: str, *, api_key: str = "EMPTY",
                 weight: float = 1.0, timeout: float = 120.0):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.model = model
        self.weight = weight

    def score(self, goal_desc: str, before_img: str,
              branch_imgs: list[str]) -> list[float]:
        """Trả list điểm (đã chuẩn [0,1] * weight) cho từng nhánh hợp lệ."""
        if not branch_imgs:
            return []
        messages = build_judge_messages(goal_desc, before_img, branch_imgs)
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=0.0, max_tokens=256)
        text = resp.choices[0].message.content or ""
        norm = parse_judge_scores(text, len(branch_imgs))
        return [self.weight * s for s in norm]
