"""HTTP client tới game service (Mac mini) từ server huấn luyện (Vast AI).

Bọc các endpoint /reset /branch /commit /step /close thành API Python tiện dùng.
Ảnh nhận về là data URL base64 (PNG) — đưa thẳng vào messages cho VLM được.

Class này KHÔNG cần GPU, test được với game service chạy local.
"""
from __future__ import annotations

from dataclasses import dataclass

import requests


@dataclass
class StepObs:
    """Quan sát một trạng thái (ảnh + số đo + tiến độ màn)."""
    wrist_image: str            # data URL
    stand_image: str            # data URL
    ctrl_angles: dict
    measure: dict               # {d_grip_obj, d_obj_basket, grasped, in_basket, obj_z}
    step_idx: int
    max_steps: int
    done: bool
    instruction_vi: str
    target_object_name: str

    @classmethod
    def from_json(cls, d: dict) -> "StepObs":
        return cls(**d)


@dataclass
class Branch:
    """Một nhánh i+1 đã chạy thử trong game."""
    branch_id: str
    applied_angles: dict
    wrist_image: str
    stand_image: str
    measure: dict
    obj_speed: float
    done: bool

    @classmethod
    def from_json(cls, d: dict) -> "Branch":
        return cls(**d)


class EnvClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8000",
                 timeout: float = 180.0, connect_timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        # (connect, read): connect nhanh để phát hiện tunnel hỏng; read dài vì
        # /reset, /branch phải render (và truyền ảnh qua SSH tunnel).
        self.timeout = (connect_timeout, timeout)
        self.session_id: str | None = None

    def _post(self, path: str, payload: dict) -> dict:
        r = requests.post(f"{self.base_url}{path}", json=payload,
                          timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def health(self) -> dict:
        return requests.get(f"{self.base_url}/health",
                            timeout=(self.timeout[0], 15.0)).json()

    def levels(self) -> list:
        return requests.get(f"{self.base_url}/levels", timeout=self.timeout).json()

    def reset(self, level_id: str, *, img_w: int = 480, img_h: int = 360,
              settle_seconds: float = 0.6) -> StepObs:
        r = self._post("/reset", {"level_id": level_id, "img_w": img_w,
                                  "img_h": img_h, "settle_seconds": settle_seconds})
        self.session_id = r["session_id"]
        return StepObs.from_json(r["obs"])

    def branch(self, actions: list[dict]) -> list[Branch]:
        assert self.session_id, "chưa reset"
        r = self._post("/branch", {"session_id": self.session_id,
                                   "actions": actions})
        return [Branch.from_json(b) for b in r["branches"]]

    def commit(self, branch_id: str) -> StepObs:
        assert self.session_id, "chưa reset"
        r = self._post("/commit", {"session_id": self.session_id,
                                   "branch_id": branch_id})
        return StepObs.from_json(r["obs"])

    def step(self, angles: dict) -> StepObs:
        assert self.session_id, "chưa reset"
        r = self._post("/step", {"session_id": self.session_id, "angles": angles})
        return StepObs.from_json(r["obs"])

    def close(self):
        if self.session_id:
            self._post("/close", {"session_id": self.session_id})
            self.session_id = None
