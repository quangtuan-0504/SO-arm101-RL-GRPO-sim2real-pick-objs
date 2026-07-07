# Hàm reward hiện đang quá rối và mất kiểm soát, ...

Exported: 7/6/2026, 8:02:48 PM

---

**User:**

Hàm reward hiện đang quá rối và mất kiểm soát, tôi cần nắm lại một cách đầy đủ cách mà hàm reward hoạt động ở hiện tại để đưa ra các chiến lược điều chỉnh và tối ưu.

---

**Assistant:**

Tôi sẽ đọc toàn bộ code liên quan đến hàm reward để nắm rõ cách nó hoạt động hiện tại.

Giờ đọc chi tiết logic tổng hợp reward trong `episode_manager.py` và `rollout.py` — nơi format gate + geometric + winner được kết hợp thành total reward.

Tôi đã đọc đủ các thành phần. Kiểm tra nhanh `config.yaml` và cách `grpo_train.py` nạp tham số để nắm giá trị thực đang chạy.

Tôi đã đọc hết chuỗi reward từ đo đạc → thế năng → geometric → tổng hợp trong rollout/episode_manager → wiring vào GRPO. Dưới đây là bức tranh đầy đủ về cách reward đang hoạt động, kèm những điểm khiến nó "rối" và các núm để bạn điều chỉnh.

## Kiến trúc tổng thể: reward chảy qua 4 tầng

```
completion (text JSON)
      │
 [1] FORMAT GATE  ── sai → không branch, gán điểm phạt theo thứ hạng
      │ hợp lệ
 [2] geometric_reward(before → after)   ← tín hiệu CHÍNH
      │   = pbrs_scale·(Φ_after − Φ_before)  + mốc rời rạc  − phạt fling
      │
 [3] anti-stall  ── trừ stall_penalty nếu no-op thật sự
      │
 [4] TỔNG HỢP nhóm  ── sai format = (hợp lệ tệ nhất − margin)
      │
   chọn winner (geo cao nhất) → commit → advance
```

Judge VLM (#3 trong `rewards/`) hiện **chưa nối vào pipeline** — chỉ có class, không được gọi trong `episode_manager`/`rollout`. Nên thực chất reward đang chạy chỉ gồm **format gate + geometric + anti-stall**.

## Tầng lõi: PBRS và hàm thế năng Φ (`sim/metrics.py`)

Đây là trái tim và cũng là chỗ phức tạp nhất. `geometric_reward` mỗi step tính:

```
reward_pbrs = pbrs_scale · (Φ(after) − Φ(before))
```

Φ là hàm thuần trạng thái (điều kiện cốt lõi của PBRS để không bóp méo policy tối ưu). Φ chia làm 2 chế độ, chuyển đổi bằng cờ khách quan `grasped`:

**Chưa kẹp (approach)** — mô hình "hành lang chữ L":
```
belowness = clip((h_standoff − dz) / h_standoff, 0, 1)   # 1 khi đầu kẹp ngang/dưới vật
L = d_xy · (1 + low_penalty · belowness) + |dz|
Φ = − approach · L
```
Ý tưởng: lệch ngang `d_xy` bị phạt nặng hơn khi đầu kẹp còn thấp → ép "nâng lên trước, canh ngang, rồi hạ thẳng xuống", tránh húc văng vật. Đây là bản viết lại để bỏ "thung lũng Φ" từng gây reward âm trên chính quỹ đạo đúng.

**Đã kẹp (carry)**:
```
Φ = grasped + carry · (−d_obj_basket)
```

**Vào giỏ**: cộng thêm `in_basket`.

Trọng số `PHI_WEIGHTS` hiện tại: `approach=1.0, low_penalty=2.0, h_standoff=0.10, grasped=1.0, carry=1.0, in_basket=2.0`.

## Tầng mốc rời rạc + chống hack (trong `geometric_reward`)

Cộng/trừ thẳng vào reward, **không** phải PBRS (nên về lý thuyết có thể lệch policy tối ưu chút, nhưng đánh đổi để có tín hiệu mạnh ở khoảnh khắc quyết định):

- `+grasp_bonus (5.0)` khi `grasped` chuyển False→True
- `−grasp_bonus` khi tuột vật (đã kẹp → mất, chưa vào giỏ) — phạt đối xứng chống "kẹp rồi buông"
- `+basket_bonus (10.0)` khi vào giỏ
- `−fling_penalty (1.0)` nếu tốc độ vật > `fling_speed (1.0 m/s)`

## Tầng anti-stall (trong rollout/episode_manager, không nằm trong geometric)

Phạt `−stall_penalty (0.5)` nếu completion là **no-op thật sự**: cả khớp tay LẪN gripper đều đổi < `stall_eps (1e-2)`. Logic này lặp lại ở cả `rollout.py` và `episode_manager.py`.

## Tầng tổng hợp nhóm + format gate theo thứ hạng

Đây là phần dễ gây "mất kiểm soát" nhất về mặt thang điểm:

```python
total[i] = geo[i] − (stall_penalty nếu stall)        # cho nước hợp lệ
worst_valid = min(total của các nước hợp lệ)
invalid_value = worst_valid − invalid_format_margin  # cho nước sai format
```

Nước sai format luôn thấp hơn mọi nước hợp lệ một khoảng `margin (1.0)` → advantage GRPO của nó luôn âm. **Winner được chọn theo `geo` thuần** (không tính stall/format), rồi commit để advance.

## Các giá trị đang chạy thực tế

`EpisodeManager` được khởi tạo **không truyền `geo_weights`** (dùng `GeoWeights()` mặc định). Cần xác nhận `grpo_train.py` có đọc reward params từ `config.yaml` không — theo grep thì không thấy wiring, nghĩa là mọi trọng số reward đang là **hằng số mặc định trong code**, không chỉnh được qua config.

## Vì sao nó "rối và mất kiểm soát" — chẩn đoán

1. **Thang điểm không cùng đơn vị, cộng chồng nhau.** PBRS mỗi step ~±3 (pbrs_scale·ΔΦ), grasp_bonus=5, basket_bonus=10, fling=1, stall=0.5, margin=1. Các con số này được tinh chỉnh rời rạc qua nhiều lần (xem comment "hạ 10→4") nên không còn quan hệ tỉ lệ rõ ràng. Một step vừa kẹp được có thể nhảy vọt +5±3, lấn át hoàn toàn shaping liên tục.

2. **Winner chọn theo `geo` nhưng reward học lại là `total`.** Winner bỏ qua stall/format, còn gradient GRPO dùng `total`. Hai tiêu chí lệch nhau → nước được commit chưa chắc là nước có advantage cao nhất.

3. **`invalid_value` phụ thuộc `worst_valid` (động).** Điểm của nước sai format trôi theo nhóm mỗi step, khiến baseline nhóm không ổn định — advantage khó diễn giải.

4. **Φ chế độ approach là công thức ghép phi tuyến** (belowness × d_xy + |dz|). Rất nhạy với `h_standoff` và `low_penalty`; sai lệch nhỏ trong đo `dz` (chênh cao đầu kẹp − vật) đổi dấu ΔΦ. Đây là nghi phạm chính của reward âm rải rác.

5. **Trùng lặp logic** giữa `rollout.py` và `episode_manager.py` (anti-stall, tổng hợp reward) → dễ chỉnh một bên quên bên kia.

6. **PBRS lý thuyết cần `γ·Φ' − Φ`**, nhưng code dùng `Φ' − Φ` (γ=1 ngầm). Nếu GRPO single-step coi mỗi step độc lập thì ổn, nhưng cần thống nhất quan điểm.

## Các núm điều chỉnh, xếp theo mức ảnh hưởng

| Núm | File | Tác dụng | Rủi ro khi chỉnh |
|---|---|---|---|
| `pbrs_scale` | geometric_reward | biên độ shaping liên tục | quá cao → nhiễu lấn mốc; quá thấp → tín hiệu mờ ở sát đích |
| `grasp/basket_bonus` | geometric_reward | độ nổi của mốc quyết định | lệch tỉ lệ với PBRS gây spike |
| `low_penalty`, `h_standoff` | PHI_WEIGHTS | hình dạng đường tiếp cận | nghi phạm reward âm; nhạy nhất |
| `carry` vs `approach` | PHI_WEIGHTS | cân bằng 2 pha | mất liên tục Φ tại lúc grasp |
| `invalid_format_margin` | rollout/episode_mgr | tách nước sai format | baseline nhóm động |
| `stall_penalty/eps` | rollout/episode_mgr | chống đứng yên | phạt oan lúc "giữ tay + đóng kẹp" |

## Gợi ý hướng tối ưu (chưa thực thi, chờ bạn chọn)

- **Kiểm chứng trước khi chỉnh:** chạy `train/check_potential.py` — script này in Φ và geometric_reward trên quỹ đạo lý tưởng và đánh dấu mọi bước ΔΦ<0. Đây là cách nhanh nhất để thấy Φ có đơn điệu không trước khi động vào tham số.
- **Gom tất cả trọng số reward vào `config.yaml`** và wire qua `grpo_train.py`, thay vì hằng số rải rác — để thí nghiệm không cần sửa code.
- **Chuẩn hóa thang:** cố định PBRS về khoảng [−1, 1] mỗi step, rồi đặt bonus theo bội số rõ ràng (vd grasp = 3×biên PBRS).
- **Thống nhất winner và reward** (cùng dùng `total`, hoặc cùng `geo`).
- **Gỡ trùng lặp** giữa rollout.py và episode_manager.py thành một hàm `score_group` dùng chung.

Bạn muốn tôi (a) chạy `check_potential.py` để soi hình dạng Φ hiện tại, (b) gom trọng số reward ra `config.yaml`, hay (c) refactor gỡ trùng lặp trước? Mỗi hướng tôi có thể làm ngay.

---
