"""Game service (FastAPI) — chạy trên Mac mini.

Vai trò: nhận lệnh góc quay từ service huấn luyện -> step mô phỏng -> chụp ảnh
-> trả ảnh + số đo. KHÔNG build prompt, KHÔNG chấm điểm (việc đó ở server GPU).

Luồng GRPO mỗi step:
    POST /reset  {level_id}                    -> session_id + quan sát i
    POST /branch {session_id, actions:[g lệnh]} -> g nhánh i+1 (ảnh + số đo)
    POST /commit {session_id, branch_id}        -> tiến sang i+1 đã chọn

An toàn thread: MuJoCo render trên macOS gắn với MỘT thread. Mọi thao tác sim
được dồn về một ThreadPoolExecutor(max_workers=1) để đảm bảo cùng một thread.

Chạy:
    ./.venv/bin/python -m game.service               # cổng 8000
    ./.venv/bin/uvicorn game.service:app --port 8000
"""
from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from dataset.generator import load_levels, DEFAULT_OUT
from game.env import GameEnv, Observation, BranchResult
from sim.metrics import StateMeasure
from vlm.render_utils import rgb_to_data_url

ROOT = Path(__file__).resolve().parent.parent
BASE_CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))

# ----------------------------------------------------------------------
#  Quản lý phiên (mỗi màn đang chơi = 1 session). Mọi thao tác sim dồn về
#  MỘT worker thread để render OpenGL ổn định trên macOS.
# ----------------------------------------------------------------------
_executor = ThreadPoolExecutor(max_workers=1)


class Session:
    def __init__(self, env: GameEnv):
        self.env = env
        self.branches: dict[str, BranchResult] = {}


_sessions: dict[str, Session] = {}
_levels: dict[str, dict] = {}


def _load_level_table():
    path = DEFAULT_OUT
    if not path.exists():
        raise RuntimeError(
            f"Chưa có level table tại {path}. Chạy: python -m dataset.generator")
    for lv in load_levels(path):
        _levels[lv["level_id"]] = lv


def _run(fn, *args):
    """Chạy fn trên worker thread duy nhất và chờ kết quả."""
    return _executor.submit(fn, *args).result()


# ----------------------------------------------------------------------
#  Serialize
# ----------------------------------------------------------------------
def _measure_dict(m: StateMeasure) -> dict:
    return {
        "d_grip_obj": m.d_grip_obj,
        "d_obj_basket": m.d_obj_basket,
        "grasped": m.grasped,
        "in_basket": m.in_basket,
        "obj_z": m.obj_z,
    }


def _obs_dict(o: Observation) -> dict:
    return {
        "wrist_image": rgb_to_data_url(o.wrist_rgb),
        "stand_image": rgb_to_data_url(o.stand_rgb),
        "ctrl_angles": o.ctrl_angles,
        "measure": _measure_dict(o.measure),
        "step_idx": o.step_idx,
        "max_steps": o.max_steps,
        "done": o.done,
        "instruction_vi": o.instruction_vi,
        "target_object_name": o.target_object_name,
    }


# ----------------------------------------------------------------------
#  Schema request/response
# ----------------------------------------------------------------------
class ResetReq(BaseModel):
    level_id: str
    img_w: int = 480
    img_h: int = 360
    settle_seconds: float = 0.6


class BranchReq(BaseModel):
    session_id: str
    actions: list[dict]   # mỗi phần tử: {joint_name: angle_rad, ...}


class CommitReq(BaseModel):
    session_id: str
    branch_id: str


class StepReq(BaseModel):
    session_id: str
    angles: dict


class CloseReq(BaseModel):
    session_id: str


# ----------------------------------------------------------------------
app = FastAPI(title="SO-101 Game Service")


@app.on_event("startup")
def _startup():
    _load_level_table()


@app.get("/health")
def health():
    return {"status": "ok", "n_levels": len(_levels),
            "active_sessions": len(_sessions)}


@app.get("/levels")
def list_levels():
    return [
        {"level_id": lv["level_id"], "map_id": lv["map_id"],
         "target_type": lv["target_type"],
         "target_object_name": lv["target_object_name"],
         "instruction_vi": lv["instruction_vi"],
         "n_objects": len(lv["objects"]), "max_steps": lv["max_steps"]}
        for lv in _levels.values()
    ]


@app.post("/reset")
def reset(req: ResetReq):
    level = _levels.get(req.level_id)
    if level is None:
        raise HTTPException(404, f"Không có level {req.level_id!r}")

    def _do():
        env = GameEnv(BASE_CONFIG, level, img_w=req.img_w, img_h=req.img_h,
                      settle_seconds=req.settle_seconds)
        obs = env.reset()
        return env, obs

    env, obs = _run(_do)
    session_id = uuid.uuid4().hex[:12]
    _sessions[session_id] = Session(env)
    return {"session_id": session_id, "obs": _obs_dict(obs)}


@app.post("/branch")
def branch(req: BranchReq):
    sess = _sessions.get(req.session_id)
    if sess is None:
        raise HTTPException(404, "session không tồn tại")
    if not req.actions:
        raise HTTPException(400, "actions rỗng")

    results = _run(sess.env.branch, req.actions)
    sess.branches = {}
    out = []
    for r in results:
        bid = uuid.uuid4().hex[:8]
        sess.branches[bid] = r
        out.append({
            "branch_id": bid,
            "applied_angles": r.applied_angles,
            "wrist_image": rgb_to_data_url(r.wrist_rgb),
            "stand_image": rgb_to_data_url(r.stand_rgb),
            "measure": _measure_dict(r.measure),
            "obj_speed": r.obj_speed,
            "done": r.done,
        })
    return {"branches": out}


@app.post("/commit")
def commit(req: CommitReq):
    sess = _sessions.get(req.session_id)
    if sess is None:
        raise HTTPException(404, "session không tồn tại")
    br = sess.branches.get(req.branch_id)
    if br is None:
        raise HTTPException(404, "branch_id không tồn tại (đã hết hạn?)")
    obs = _run(sess.env.commit, br)
    sess.branches = {}
    return {"obs": _obs_dict(obs)}


@app.post("/step")
def step(req: StepReq):
    """Tiến bước trực tiếp bằng 1 bộ góc (không branch) — tiện debug/điều khiển."""
    sess = _sessions.get(req.session_id)
    if sess is None:
        raise HTTPException(404, "session không tồn tại")
    obs = _run(sess.env.commit_angles, req.angles)
    return {"obs": _obs_dict(obs)}


@app.post("/close")
def close(req: CloseReq):
    sess = _sessions.pop(req.session_id, None)
    if sess is not None:
        _run(sess.env.close)
    return {"ok": True}


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
