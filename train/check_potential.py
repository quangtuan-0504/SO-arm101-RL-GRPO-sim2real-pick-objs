"""Kiểm tra hàm thế năng Φ (PBRS) trên một QUỸ ĐẠO LÝ TƯỞNG scripted.

Mục đích (chẩn đoán reward, KHÔNG cần GPU / MuJoCo / render):
    Dựng chuỗi trạng thái của một lần gắp THÀNH CÔNG lý tưởng:
        tiến tới vật (giữ cao, thu hẹp lệch ngang)
          -> canh thẳng trên vật -> hạ thẳng xuống
          -> kẹp -> nhấc -> mang sang giỏ -> thả vào giỏ
    rồi in Φ = potential(s) và phần thưởng geometric_reward(s_i -> s_{i+1})
    ở TỪNG bước, đồng thời ĐÁNH DẤU mọi bước mà Φ GIẢM (ΔΦ < 0).

Vì sao quan trọng:
    GRPO ở đây là single-step PBRS: mỗi bước model chỉ được thưởng theo
    ΔΦ = Φ(sau) − Φ(trước). Nếu trên chính quỹ đạo THÀNH CÔNG mà Φ có đoạn
    ĐI XUỐNG, thì tại đó reward per-step là ÂM -> policy tham lam một-bước bị
    *khuyến khích tránh* đúng động tác cần làm. Đây là nghi phạm số 1 khiến
    grasp-rate = 0%.

    Script này KHÔNG mô phỏng vật lý (không IK). Nó kiểm tra HÌNH DẠNG của Φ
    trên các số đo lý tưởng, dùng chính ``sim.metrics.potential`` và
    ``rewards.geometric_reward.geometric_reward`` mà pipeline dùng thật, nên
    bắt được lỗi shaping độc lập với sim.

Chạy:
    python -m train.check_potential
    python -m train.check_potential --steps 6   # số bước nội suy mỗi đoạn
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

from sim.metrics import StateMeasure, potential, PHI_WEIGHTS
from rewards.geometric_reward import geometric_reward, GeoWeights


# --- hình học lý tưởng (mét). Lấy standoff từ chính PHI_WEIGHTS để đồng bộ. ---
H = float(PHI_WEIGHTS["h_standoff"])   # độ cao điểm chờ phía trên vật
OBJ_REST_Z = 0.02                       # z nghỉ của vật (xấp xỉ config)
LIFT_Z = 0.13                           # z sau khi nhấc
BASKET_D0 = 0.26                        # khoảng cách vật->giỏ lúc vừa kẹp (~config)
BASKET_NEAR = 0.03                      # khoảng cách khi đã ở trên giỏ


@dataclass
class Seg:
    """Một đoạn quỹ đạo: nội suy tuyến tính từ start -> end qua ``n`` bước.

    Bộ giá trị = (d_xy, dz, d_obj_basket, obj_z). Cờ grasped/in_basket giữ cố
    định trong đoạn (đổi trạng thái xảy ra tại ranh giới đoạn).
    """
    label: str
    n: int
    start: tuple
    end: tuple
    grasped: bool
    in_basket: bool


def _mk(d_xy, dz, d_ob, obj_z, grasped, in_basket) -> StateMeasure:
    # d_grip_obj suy nhất quán từ lệch ngang + chênh cao (khi chưa kẹp).
    d_grip_obj = math.hypot(d_xy, dz)
    return StateMeasure(
        d_grip_obj=d_grip_obj, d_obj_basket=d_ob, grasped=grasped,
        in_basket=in_basket, obj_z=obj_z, d_xy=d_xy, dz=dz)


def build_trajectory(steps: int) -> list[tuple[str, StateMeasure]]:
    """Quỹ đạo gắp thành công lý tưởng, tuân theo 'phễu' tiền-kẹp của Φ.

    Chiến lược tiền-kẹp đúng ý Φ: giữ đầu kẹp Ở ĐỘ CAO chờ (dz≈H) trong khi
    thu hẹp lệch ngang -> chỉ khi canh thẳng (d_xy≈0) mới HẠ THẲNG xuống. Đây
    là đường mà 'danger'/funnel được thiết kế để khuyến khích.
    """
    segs = [
        # 1) tiến tới vật: giữ cao (dz=H), thu hẹp lệch ngang 0.20 -> 0.02
        Seg("approach (giu cao, thu hep ngang)", steps,
            (0.20, H, BASKET_D0, OBJ_REST_Z), (0.02, H, BASKET_D0, OBJ_REST_Z),
            grasped=False, in_basket=False),
        # 2) canh thẳng rồi HẠ THẲNG: d_xy 0.02->0, dz H -> sát vật
        Seg("align + descend (ha thang)", steps,
            (0.02, H, BASKET_D0, OBJ_REST_Z), (0.0, 0.015, BASKET_D0, OBJ_REST_Z),
            grasped=False, in_basket=False),
        # 3) KẸP (grasped False->True). Vật bắt đầu được giữ.
        Seg("grasp (kep duoc vat)", 1,
            (0.0, 0.0, BASKET_D0, OBJ_REST_Z), (0.0, 0.0, BASKET_D0, OBJ_REST_Z),
            grasped=True, in_basket=False),
        # 4) nhấc lên: obj_z rest -> LIFT_Z (khoảng cách tới giỏ ~ giữ nguyên)
        Seg("lift (nhac len)", max(2, steps // 2),
            (0.0, 0.0, BASKET_D0, OBJ_REST_Z), (0.0, 0.0, BASKET_D0 * 0.97, LIFT_Z),
            grasped=True, in_basket=False),
        # 5) mang sang giỏ: d_obj_basket giảm mạnh về gần giỏ
        Seg("carry (mang sang gio)", steps,
            (0.0, 0.0, BASKET_D0 * 0.97, LIFT_Z), (0.0, 0.0, BASKET_NEAR, LIFT_Z),
            grasped=True, in_basket=False),
        # 6) hạ vào lòng giỏ (in_basket False->True ở cuối)
        Seg("release into basket (tha vao gio)", max(2, steps // 2),
            (0.0, 0.0, BASKET_NEAR, LIFT_Z), (0.0, 0.0, 0.02, 0.04),
            grasped=True, in_basket=True),
    ]

    traj: list[tuple[str, StateMeasure]] = []
    for seg in segs:
        n = max(1, seg.n)
        for k in range(n):
            # t=0..1 trên đoạn; đoạn 1-state thì lấy end.
            t = 0.0 if n == 1 else k / (n - 1)
            vals = tuple(a + (b - a) * t for a, b in zip(seg.start, seg.end))
            d_xy, dz, d_ob, obj_z = vals
            traj.append((seg.label,
                         _mk(d_xy, dz, d_ob, obj_z, seg.grasped, seg.in_basket)))
    return traj


def main():
    ap = argparse.ArgumentParser(description="Kiem tra Phi (PBRS) tren quy dao ly tuong")
    ap.add_argument("--steps", type=int, default=6,
                    help="So buoc noi suy moi doan (day hon -> lo diem lom Phi ro hon)")
    ap.add_argument("--tol", type=float, default=1e-6,
                    help="Nguong coi la 'Phi giam' (ΔΦ < -tol)")
    args = ap.parse_args()

    gw = GeoWeights()
    traj = build_trajectory(args.steps)

    print(f"PHI_WEIGHTS = {PHI_WEIGHTS}")
    print(f"GeoWeights  = {gw}\n")
    header = (f"{'idx':>3} {'phase':<34} {'d_xy':>6} {'dz':>6} "
              f"{'d_obj_bk':>8} {'grasp':>5} {'basket':>6} "
              f"{'Phi':>8} {'dPhi':>8} {'geo_rwd':>8}")
    print(header)
    print("-" * len(header))

    drops = []          # các bước Φ giảm (ΔΦ < -tol)
    neg_reward = []     # các bước geometric_reward < 0
    cum = 0.0
    prev_phi = None
    prev_m = None
    for i, (label, m) in enumerate(traj):
        phi = potential(m)
        if prev_m is None:
            dphi = 0.0
            gr = 0.0
        else:
            dphi = phi - prev_phi
            gr = geometric_reward(prev_m, m, weights=gw)
            cum += gr
            if dphi < -args.tol:
                drops.append((i, label, dphi))
            if gr < -args.tol:
                neg_reward.append((i, label, gr))
        print(f"{i:>3} {label:<34} {m.d_xy:>6.3f} {m.dz:>6.3f} "
              f"{m.d_obj_basket:>8.3f} {str(m.grasped):>5} {str(m.in_basket):>6} "
              f"{phi:>8.3f} {dphi:>8.3f} {gr:>8.3f}")
        prev_phi, prev_m = phi, m

    print("-" * len(header))
    print(f"Tong geometric_reward tich luy tren quy dao thanh cong: {cum:.3f}\n")

    if not drops:
        print("[PASS] Φ don dieu KHONG giam tren toan quy dao ly tuong.")
    else:
        print(f"[CANH BAO] Co {len(drops)} buoc Φ GIAM (reward per-step AM) "
              f"tren chinh quy dao thanh cong:")
        for i, label, d in drops:
            print(f"   - buoc {i:>3} [{label}] ΔΦ = {d:+.4f}")
        print("\n  => Tai cac buoc nay, PBRS mot-step tra reward AM cho dung "
              "dong tac can lam. Policy tham lam mot-buoc bi khuyen khich TRANH "
              "chung -> nghi pham truc tiep cho grasp-rate = 0%.")
        print("  Goi y: dung MOT ham khoang-cach lien tuc (khong blend 2 dich "
              "khac goc quy chieu). Vi du 'hanh lang chu L': "
              "L = d_xy*(1+k*belowness) + |dz|, de Φ khong co thung lung.")

    if neg_reward:
        print(f"\n[GHI CHU] {len(neg_reward)} buoc co geometric_reward < 0 "
              f"(bao gom ca phat stall/fling neu co).")


if __name__ == "__main__":
    main()
