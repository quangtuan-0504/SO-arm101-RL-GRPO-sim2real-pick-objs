"""Client gọi Qwen-VL (qwen3.6-flash) qua OpenAI-compatible SDK của DashScope.

Dùng OpenAI Python SDK (Cách 1 trong tài liệu QwenCloud) vì qwen3.6-flash là
model "deep thinking": cần ``enable_thinking=True`` kèm ``stream=True`` và trả
``reasoning_content`` tách khỏi ``content``. SDK xử lý SSE streaming sẵn, ít
lỗi hơn tự parse stream bằng tay.

Cài đặt:  pip install openai
Endpoint mặc định: https://dashscope-intl.aliyuncs.com/compatible-mode/v1
"""
from __future__ import annotations

import json
import re

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
                return json.loads(cleaned[start:i + 1])
    raise ValueError("JSON không cân bằng dấu ngoặc")


class QwenVLClient:
    """Gọi qwen3.6-flash qua OpenAI SDK (DashScope compatible-mode)."""

    def __init__(self, api_key: str, base_url: str,
                 model: str = "qwen3.6-flash",
                 enable_thinking: bool = True,
                 temperature: float = 0.1, max_tokens: int = 2048,
                 timeout: float = 120.0, verbose: bool = False):
        # Import tại đây để chỉ bắt buộc cài openai khi dùng client thật.
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.model = model
        self.enable_thinking = enable_thinking
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.verbose = verbose

    def predict(self, messages: list) -> dict:
        """Gửi messages, trả về dict đã parse từ JSON model sinh ra."""
        return parse_control_json(self._complete(messages))

    def _complete(self, messages: list) -> str:
        """Trả về phần 'content' (đáp án) đã ghép từ stream."""
        if self.enable_thinking:
            return self._complete_stream(messages)
        return self._complete_once(messages)

    def _complete_once(self, messages: list) -> str:
        """Gọi không streaming (khi tắt thinking)."""
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            extra_body={"enable_thinking": False},
        )
        return completion.choices[0].message.content or ""

    def _complete_stream(self, messages: list) -> str:
        """Gọi streaming với thinking; gom 'content', bỏ qua reasoning."""
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            extra_body={"enable_thinking": True},
            stream=True,
        )
        content_parts: list[str] = []
        answering = False
        for chunk in completion:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            reasoning = getattr(delta, "reasoning_content", None)
            if self.verbose and reasoning:
                print(reasoning, end="", flush=True)
            if getattr(delta, "content", None):
                if self.verbose and not answering:
                    print("\n--- answer ---")
                    answering = True
                if self.verbose:
                    print(delta.content, end="", flush=True)
                content_parts.append(delta.content)
        if self.verbose:
            print()
        return "".join(content_parts)


class MockQwenClient:
    """Client giả lập: không gọi mạng, dùng để kiểm thử vòng lặp.

    Sinh phản hồi theo "kịch bản": giữ tư thế home rồi đóng kẹp và nâng lên,
    đánh dấu done sau ``done_after`` lần gọi. Chỉ để xác nhận đường ống
    render -> parse -> set_ctrl -> step hoạt động.
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
            angles["gripper"] = -0.1
        else:
            phase, grip, done = "lift", "hold", True
            angles["gripper"] = -0.1
            angles["shoulder_lift"] = 0.2

        return {
            "reasoning": f"[MOCK] lần gọi {self.calls}",
            "target_object": "mock",
            "phase": phase,
            "gripper_action": grip,
            "joint_angles_rad": angles,
            "done": done,
            "confidence": 0.5,
        }
