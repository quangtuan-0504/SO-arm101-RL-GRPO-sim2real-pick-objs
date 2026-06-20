"""Ghi metric huấn luyện ra JSONL để trực quan hoá tiến trình GRPO.

Mỗi lần chấm một step môi trường (score_and_advance), reward function gọi
``MetricsLogger.log_step(...)`` với một bản ghi gọn. File JSONL này sau đó được
``train/plot_metrics.py`` đọc để vẽ biểu đồ, hoặc bạn tự phân tích bằng pandas.

Mục tiêu: trả lời câu hỏi "mục tiêu huấn luyện có khả năng đạt tới đâu" qua các
tín hiệu:
  - success_rate : tỉ lệ episode KẾT THÚC với vật ĐÃ vào giỏ (done=true).
  - grasp_rate   : tỉ lệ winner đã KẸP được vật.
  - format_rate  : tỉ lệ completion ĐÚNG format JSON (gate đầu tiên phải qua).
  - reward (mean/max), geometric reward, khoảng cách kẹp->vật / vật->giỏ.

KHÔNG cần GPU; chỉ ghi text. An toàn khi chạy song song nhiều episode (mỗi dòng
độc lập, append + flush).
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path


class MetricsLogger:
    """Append-only JSONL writer cho metric per-step."""

    def __init__(self, path: str):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.path = str(p)
        self._fp = open(self.path, "a", encoding="utf-8", buffering=1)
        self._lock = threading.Lock()
        self._global_step = 0

    def log_step(self, *, episode_id: str, level_id: str, step_idx: int,
                 group_size: int, n_valid: int,
                 rewards: list[float], geometric_rewards: list[float],
                 format_rewards: list[float], winner_idx: int | None,
                 finished: bool, success: bool,
                 winner_measure: dict | None) -> dict:
        """Ghi một dòng metric cho 1 step. Trả về dict đã ghi (tiện in gọn)."""
        n = max(1, group_size)
        wm = winner_measure or {}
        rec = {
            "global_step": self._global_step,
            "t": round(time.time(), 3),
            "episode_id": episode_id,
            "level_id": level_id,
            "step_idx": step_idx,
            "group_size": group_size,
            "n_valid": n_valid,
            "format_rate": n_valid / n,
            "reward_mean": _mean(rewards),
            "reward_max": max(rewards) if rewards else 0.0,
            "geo_mean": _mean(geometric_rewards),
            "geo_max": max(geometric_rewards) if geometric_rewards else 0.0,
            "winner_reward": (rewards[winner_idx]
                              if winner_idx is not None and rewards else 0.0),
            "finished": bool(finished),
            "success": bool(success),
            # tín hiệu hình học của winner (nếu có commit)
            "grasped": bool(wm.get("grasped", False)),
            "in_basket": bool(wm.get("in_basket", False)),
            "d_grip_obj": wm.get("d_grip_obj"),
            "d_obj_basket": wm.get("d_obj_basket"),
            "obj_z": wm.get("obj_z"),
        }
        with self._lock:
            self._global_step += 1
            self._fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._fp.flush()
        return rec

    def close(self):
        try:
            self._fp.close()
        except Exception:
            pass


def _mean(xs) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0
