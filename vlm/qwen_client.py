"""Client gọi Qwen-VL qua endpoint OpenAI-compatible (chỉ dùng stdlib).

Hỗ trợ DashScope compatible-mode hoặc bất kỳ server nào nói "chat/completions":
    base_url = https://dashscope.aliyuncs.com/compatible-mode/v1
    model    = qwen-vl-max-latest | qwen2.5-vl-72b-instruct | ...

Ngoài ra có MockQwenClient để chạy thử vòng lặp end-to-end mà không cần mạng
hoặc API key.
"""
from __future__ import annotations

import json
import re
import urllib.request
import urllib.error

from sim.scene_builder import JOINT_NAMES


# ----------------------------------------------------------------------
#  Parse JSON từ phần trả lời của model (chịu được khi bị bọc ```json)
# ----------------------------------------------------------------------
def parse_control_json(text: str) -> dict:
    """Trích đối tượng JSON đầu tiên trong ``text`` và parse thành dict."""
    if not text:
        raise ValueError("Phản hồi rỗng từ model")

    cleaned = text.strip()
    # Bỏ rào ```json ... ``` nếu có
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()

    # Lấy khối {...} cân bằng đầu tiên
    start = cleaned.find("{")
    if start == -1:
        raise ValueError(f"Không tìm thấy JSON trong phản hồi: {text[:200]!r}")
    depth = 0
    for i in range(start, len(cleaned)):
        if cleaned[i] == "{":
            depth += 1
        elif cleaned[i] == "}":
            depth -= 1
            if depth == 0:
                blob = cleaned[start:i + 1]
                return json.loads(blob)
    raise ValueError("JSON không cân bằng dấu ngoặc")


class QwenVLClient:
    """Gọi Qwen-VL qua HTTP (OpenAI-compatible /chat/completions)."""

    def __init__(self, api_key: str, base_url: str,
                 model: str = "qwen-vl-max-latest",
                 temperature: float = 0.1, max_tokens: int = 1024,
                 timeout: float = 90.0):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def predict(self, messages: list) -> dict:
        """Gửi messages, trả về dict đã parse từ JSON model sinh ra."""
        raw = self._raw_completion(messages)
        return parse_control_json(raw)

    def _raw_completion(self, messages: list) -> str:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=body, method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "ignore")
            raise RuntimeError(f"Qwen API lỗi {e.code}: {detail}") from e
        return data["choices"][0]["message"]["content"]


class MockQwenClient:
    """Client giả lập: không gọi mạng, dùng để kiểm thử vòng lặp.

    Sinh phản hồi đơn giản theo "kịch bản": tiến dần các khớp về home rồi
    đóng kẹp, đánh dấu done sau ``done_after`` lần gọi. Mục đích chỉ để xác
    nhận đường ống render -> parse -> set_ctrl -> step hoạt động.
    """

    def __init__(self, sim, done_after: int = 3):
        self.sim = sim
        self.done_after = done_after
        self.calls = 0

    def predict(self, messages: list) -> dict:
        self.calls += 1
        home = self.sim.config.get("arm", {}).get("home_pose", {})
        angles = {n: round(float(home.get(n, 0.0)), 3) for n in JOINT_NAMES}

        if self.calls < self.done_after:
            phase, grip, done = "approach", "open", False
        elif self.calls == self.done_after:
            phase, grip, done = "grasp", "close", False
            angles["gripper"] = -0.1  # đóng kẹp
        else:
            phase, grip, done = "lift", "hold", True
            angles["gripper"] = -0.1
            angles["shoulder_lift"] = 0.2  # nâng lên

        return {
            "reasoning": f"[MOCK] lần gọi {self.calls}",
            "target_object": "mock",
            "phase": phase,
            "gripper_action": grip,
            "joint_angles_rad": angles,
            "done": done,
            "confidence": 0.5,
        }
