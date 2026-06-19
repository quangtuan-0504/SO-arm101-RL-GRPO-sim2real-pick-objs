"""Policy VLM thật (sinh G completions) — CHẠY TRÊN GPU, chưa test được trên Mac.

Dùng để:
  - thay ``MockPolicy`` trong ``train.rollout`` để chạy thử rollout với model thật
    (sanity check trước khi bật GRPO),
  - hoặc làm tham chiếu cho cách model sinh nhiều completion cho cùng 1 prompt.

Hỗ trợ 2 backend:
  - "hf"   : transformers ``.generate`` (đơn giản, chậm, không cần server).
  - "vllm" : gọi vLLM OpenAI-compatible server (nhanh, khuyến nghị khi train).

LƯU Ý: import nặng (torch/transformers/openai) đặt trong __init__ để Mac không
phải cài. Mọi thứ ở đây cần GPU + model weights.
"""
from __future__ import annotations


class HFVLMPolicy:
    """Sinh G completions bằng transformers (Qwen2.5-VL hoặc tương đương)."""

    def __init__(self, model_id: str, *, temperature: float = 1.0,
                 max_new_tokens: int = 512, device: str = "cuda"):
        import torch
        from transformers import AutoProcessor, AutoModelForVision2Seq

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map=device)
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    def generate(self, messages: list, n: int) -> list[str]:
        # messages theo chuẩn OpenAI-vision (text + image_url data URL).
        # Cần chuyển data URL -> PIL Image cho processor của transformers.
        from train.image_utils import data_url_to_pil

        images = [data_url_to_pil(c["image_url"]["url"])
                  for m in messages if isinstance(m["content"], list)
                  for c in m["content"] if c.get("type") == "image_url"]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=images,
                                return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **inputs, do_sample=True, temperature=self.temperature,
            max_new_tokens=self.max_new_tokens, num_return_sequences=n)
        gen = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(gen, skip_special_tokens=True)


class VLLMPolicy:
    """Sinh G completions qua vLLM OpenAI-compatible server (khuyến nghị)."""

    def __init__(self, base_url: str, model: str, *,
                 api_key: str = "EMPTY", temperature: float = 1.0,
                 max_tokens: int = 512, timeout: float = 120.0):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, messages: list, n: int) -> list[str]:
        # vLLM hỗ trợ n>1 để sinh nhiều completion cùng prompt trong 1 request.
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages, n=n,
            temperature=self.temperature, max_tokens=self.max_tokens)
        return [c.message.content or "" for c in resp.choices]
