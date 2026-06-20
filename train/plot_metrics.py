"""Trực quan hoá tiến trình huấn luyện GRPO từ file metric JSONL.

Đọc file do train/metrics_logger.py sinh ra (mỗi dòng = 1 step môi trường), vẽ
các biểu đồ giúp trả lời "mục tiêu có khả năng đạt tới đâu":

  1) success_rate (rolling)  — tỉ lệ episode hoàn thành (vật vào giỏ).
  2) grasp_rate   (rolling)  — tỉ lệ winner kẹp được vật.
  3) format_rate  (rolling)  — tỉ lệ completion đúng JSON (gate đầu vào).
  4) reward (mean/max, rolling) + geometric reward.
  5) khoảng cách kẹp->vật và vật->giỏ của winner (giảm dần = tiến gần mục tiêu).

Cách dùng:
    python -m train.plot_metrics logs/metrics_2025....jsonl
    python -m train.plot_metrics logs/metrics_....jsonl --out progress.png --window 100

Yêu cầu: matplotlib (pip install matplotlib). Không cần GPU.
Nếu chỉ muốn xem nhanh con số: thêm --summary-only (không vẽ).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_records(path: str) -> list[dict]:
    recs = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return recs


def _rolling(xs: list[float], window: int) -> list[float]:
    """Trung bình trượt đơn giản (cửa sổ <= window, tính từ đầu chuỗi)."""
    from collections import deque
    out: list[float] = []
    buf: "deque[float]" = deque(maxlen=window)
    for x in xs:
        buf.append(x)
        out.append(sum(buf) / len(buf))
    return out


def _finished_only(recs: list[dict]) -> list[dict]:
    return [r for r in recs if r.get("finished")]


def print_summary(recs: list[dict]) -> None:
    n = len(recs)
    if n == 0:
        print("[plot_metrics] file rỗng — chưa có metric nào.")
        return
    fin = _finished_only(recs)
    n_fin = len(fin)
    succ = sum(1 for r in fin if r.get("success"))
    grasp_steps = sum(1 for r in recs if r.get("grasped"))
    print("=" * 60)
    print(f"Tổng step ghi nhận      : {n}")
    print(f"Episode kết thúc        : {n_fin}")
    if n_fin:
        print(f"  - Thành công (vào giỏ): {succ}  ({100*succ/n_fin:.1f}%)")
        print(f"  - Thất bại            : {n_fin - succ}")
    print(f"Step có winner đã kẹp   : {grasp_steps}  ({100*grasp_steps/n:.1f}%)")
    print(f"format_rate trung bình  : {_avg(recs, 'format_rate'):.3f}")
    print(f"reward_mean trung bình  : {_avg(recs, 'reward_mean'):.3f}")
    print(f"reward_max  trung bình  : {_avg(recs, 'reward_max'):.3f}")
    # 100 step gần nhất (xu hướng hiện tại)
    tail = recs[-100:]
    tfin = _finished_only(tail)
    if tfin:
        tsucc = sum(1 for r in tfin if r.get("success"))
        print(f"[100 step gần nhất] success_rate ~ {100*tsucc/len(tfin):.1f}% "
              f"({tsucc}/{len(tfin)} episode)")
    print("=" * 60)


def _avg(recs, key) -> float:
    vals = [r[key] for r in recs if isinstance(r.get(key), (int, float))]
    return sum(vals) / len(vals) if vals else 0.0


def make_plot(recs: list[dict], out_path: str, window: int) -> None:
    try:
        import matplotlib
    except ModuleNotFoundError:
        print("[plot_metrics] Chưa cài matplotlib — bỏ qua vẽ biểu đồ.\n"
              "  Cài: pip install matplotlib  (hoặc dùng --summary-only).")
        return
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = list(range(len(recs)))
    fmt = _rolling([r.get("format_rate", 0.0) for r in recs], window)
    grasp = _rolling([1.0 if r.get("grasped") else 0.0 for r in recs], window)
    rmean = _rolling([r.get("reward_mean", 0.0) for r in recs], window)
    rmax = _rolling([r.get("reward_max", 0.0) for r in recs], window)
    geo = _rolling([r.get("geo_mean", 0.0) for r in recs], window)

    # success_rate tính theo các episode đã kết thúc (rolling theo episode).
    fin = _finished_only(recs)
    fin_x = list(range(len(fin)))
    succ = _rolling([1.0 if r.get("success") else 0.0 for r in fin], window)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"GRPO training progress (rolling window={window})", fontsize=13)

    ax = axes[0][0]
    if fin_x:
        ax.plot(fin_x, succ, color="tab:green")
    ax.set_title("Success rate (episode hoàn thành — vật vào giỏ)")
    ax.set_xlabel("episode đã kết thúc")
    ax.set_ylabel("tỉ lệ")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)

    ax = axes[0][1]
    ax.plot(steps, fmt, label="format_rate", color="tab:blue")
    ax.plot(steps, grasp, label="grasp_rate", color="tab:orange")
    ax.set_title("Format rate & Grasp rate")
    ax.set_xlabel("step")
    ax.set_ylabel("tỉ lệ")
    ax.set_ylim(-0.02, 1.02)
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1][0]
    ax.plot(steps, rmean, label="reward_mean", color="tab:purple")
    ax.plot(steps, rmax, label="reward_max", color="tab:red")
    ax.plot(steps, geo, label="geo_mean", color="tab:gray", alpha=0.7)
    ax.set_title("Reward (trung bình trượt)")
    ax.set_xlabel("step")
    ax.set_ylabel("reward")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1][1]
    dgo = [r.get("d_grip_obj") for r in recs]
    dob = [r.get("d_obj_basket") for r in recs]
    gx = [i for i, v in enumerate(dgo) if isinstance(v, (int, float))]
    gy = _rolling([v for v in dgo if isinstance(v, (int, float))], window)
    bx = [i for i, v in enumerate(dob) if isinstance(v, (int, float))]
    by = _rolling([v for v in dob if isinstance(v, (int, float))], window)
    if gx:
        ax.plot(gx, gy, label="d(kẹp→vật)", color="tab:cyan")
    if bx:
        ax.plot(bx, by, label="d(vật→giỏ)", color="tab:brown")
    ax.set_title("Khoảng cách winner (giảm = tiến gần mục tiêu)")
    ax.set_xlabel("step")
    ax.set_ylabel("khoảng cách")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=120)
    print(f"[plot_metrics] đã lưu biểu đồ: {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Vẽ tiến trình train GRPO từ JSONL")
    ap.add_argument("metrics_file", help="đường dẫn file metrics .jsonl")
    ap.add_argument("--out", default=None,
                    help="ảnh PNG đầu ra (mặc định cạnh file metric).")
    ap.add_argument("--window", type=int, default=50,
                    help="cửa sổ trung bình trượt (step).")
    ap.add_argument("--summary-only", action="store_true",
                    help="chỉ in tóm tắt, không vẽ biểu đồ.")
    args = ap.parse_args()

    recs = load_records(args.metrics_file)
    print_summary(recs)
    if args.summary_only or not recs:
        return
    out = args.out or str(Path(args.metrics_file).with_suffix(".png"))
    make_plot(recs, out, args.window)


if __name__ == "__main__":
    main()
