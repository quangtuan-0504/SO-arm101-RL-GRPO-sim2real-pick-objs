# hãy giúp tôi giải thích đầy đủ cách hoạt động c...

Exported: 7/6/2026, 8:03:46 PM

---

**User:**

hãy giúp tôi giải thích đầy đủ cách hoạt động của cơ chế chấm điểm hiện tại

---

**Assistant:**

Tôi sẽ đọc các file liên quan đến cơ chế chấm điểm để giải thích đầy đủ cho bạn.

Để giải thích đầy đủ, tôi xem thêm cách reward được gọi trong quá trình rollout/training và các thành phần reward khác.

Tôi xem thêm file format_reward để hoàn thiện bức tranh chấm điểm.

Giờ tôi đã có đầy đủ bức tranh. Dưới đây là giải thích toàn bộ cơ chế chấm điểm hiện tại.

## Tổng quan: một step = một mẫu GRPO

Hệ thống dùng **single-step GRPO**. Mỗi bước của môi trường tạo ra 1 mẫu huấn luyện:
- **prompt** = 2 ảnh (cổ tay + đứng) + instruction + bảng khớp
- **group** = G completions do policy (VLM) sinh ra cùng một prompt
- **reward** = điểm cho từng completion, để GRPO tính advantage trong nhóm

Cơ chế chấm điểm có **3 tầng ưu tiên**: FORMAT (gate) → GEOMETRIC (tín hiệu chính) → JUDGE VLM (tùy chọn, chưa bật). Hiện tại tầng 1 và 2 đang chạy.

---

## Tầng 1 — FORMAT: cổng chặn theo thứ hạng

File `rewards/format_reward.py`. Với mỗi completion, `check_format` kiểm tra JSON có hợp lệ không:
- Đủ các khóa bắt buộc: `phase`, `gripper_action`, `joint_angles_rad`, `done`
- `phase` ∈ {approach, align, grasp, lift, carry, release}
- `gripper_action` ∈ {open, close, hold}
- `done` là bool
- `joint_angles_rad` có đủ 6 khớp và mỗi góc là số

Điểm mấu chốt về cách xử lý điểm cho nước **sai format** (trong `rollout.py`):
- Nước sai format **không được gửi cho game** (tiết kiệm chi phí branch/judge).
- Reward của nó **không phải là 0 cố định**, mà được đặt = `(reward của nước hợp lệ tệ nhất trong nhóm − margin)`, với `invalid_format_margin = 1.0`.

Đây là **gate theo thứ hạng**: đảm bảo mọi nước sai format luôn xếp dưới mọi nước hợp lệ, nên advantage GRPO của nó luôn âm → model bị đẩy ra khỏi việc sinh sai format, không bao giờ "học ngược".

---

## Tầng 2 — GEOMETRIC: tín hiệu chính (PBRS)

File `rewards/geometric_reward.py` + `sim/metrics.py`. Đây là tín hiệu học chính, lấy **trực tiếp từ tọa độ 3D thật trong MuJoCo** (privileged info) — chính xác, tức thời, không nhiễu. Lưu ý: policy (VLM) chỉ nhìn ảnh; các tọa độ thật chỉ dùng lúc chấm điểm. Đây là setup hợp lệ: *privileged reward + vision-only policy*.

Reward của một step (từ trạng thái `before` → `after`) gồm 4 thành phần:

### 1) PBRS — thành phần chính (dày, mỗi step)
```
reward = pbrs_scale × (Φ(after) − Φ(before))
```
với `pbrs_scale = 4.0`. Đây là **potential-based reward shaping**: chỉ thưởng theo chênh lệch thế năng Φ giữa hai trạng thái nên **không bóp méo chính sách tối ưu**, chỉ làm tín hiệu dày hơn để dễ hội tụ.

**Hàm thế năng Φ** (trong `metrics.py`, hàm `potential`) chia làm 2 pha, chuyển pha do sự kiện `grasped` (đo khách quan) chứ không do nhãn model tự khai — nên model không gian lận được:

- **Pha APPROACH (chưa kẹp)** — hành lang chữ L, kẹp từ trên xuống:
  ```
  belowness = clip((h − dz) / h, 0, 1)      # 1: đầu kẹp ngang/dưới vật; 0: đủ cao
  L = d_xy × (1 + low_penalty × belowness) + |dz|
  Φ = − approach × L
  ```
  Trong đó `d_xy` là lệch ngang, `dz` là chênh cao đầu kẹp − vật, `h_standoff = 0.10m`, `low_penalty = 2.0`, `approach = 1.0`.

  Ý nghĩa: lệch ngang bị phạt **nặng hơn khi đầu kẹp còn thấp** → cách rẻ nhất để tiến khi ở thấp là **nâng lên trước**, canh ngang phía trên vật, rồi hạ thẳng xuống (`|dz|` kéo xuống). Điều này ép "đường trên" tránh húc văng vật, và vì là một hàm khoảng cách liền, đơn điệu, nên không tạo "thung lũng" Φ gây reward âm trên chính quỹ đạo thành công.

- **Pha CARRY (đã kẹp)**:
  ```
  Φ = grasped + carry × (−d_obj_basket)      # gần giỏ hơn = Φ cao hơn
  ```
  với `grasped = 1.0`, `carry = 1.0`. Nếu đã vào giỏ thì cộng thêm `in_basket = 2.0`.

### 2) Mốc rời rạc — grasp bonus
- Vừa kẹp được vật (`grasped` chuyển False→True): **+5.0** (`grasp_bonus`).
- Phạt đối xứng nếu tuột vật sau khi đã kẹp (mà chưa vào giỏ): **−5.0**. Ngăn trò "kẹp rồi buông" để ăn bonus lặp lại.

### 3) Mốc cuối — basket bonus
- Vật vào giỏ lần đầu (`in_basket` False→True): **+10.0** (`basket_bonus`).

Các mốc rời rạc này nổi rõ trên nền PBRS, tạo tín hiệu mạnh cho các hành động chính xác (đóng kẹp đúng lúc, thả vào giỏ) mà PBRS liên tục thưởng rất mờ khi ở sát đích.

### 4) Chống reward-hacking — fling penalty
- Nếu vận tốc vật sau bước > `fling_speed = 1.0 m/s`: phạt **−1.0** (`fling_penalty`). Ngăn trò đập/hất vật cho văng gần giỏ.

---

## Ghép lại trong rollout (`train/rollout.py`)

Với mỗi nhóm G completions của một step:

1. **Format gate**: phân loại hợp lệ / sai.
2. **Branch** chỉ các lệnh hợp lệ vào game → nhận trạng thái i+1 mỗi nhánh.
3. **Geometric reward** cho từng nhánh hợp lệ: `geometric_reward(before, after, obj_speed_after)`.
4. **Anti-stall penalty**: nếu completion cho ra pose gần y hệt pose hiện tại (cả khớp tay lẫn gripper đều không đổi, ngưỡng `stall_eps = 1e-2`) → phạt **−0.5** (`stall_penalty`). Đây là "no-op thật sự" — tay đứng yên. Lưu ý "giữ tay + đóng kẹp" khi gắp **không** bị phạt oan. Mục đích tạo phương sai trong nhóm để gradient ưu tiên "có di chuyển".
5. **Tổng hợp**:
   ```
   total[i hợp lệ]  = geo_rewards[i] − (stall_penalty nếu bị stall)
   total[i sai fmt] = min(total hợp lệ) − invalid_format_margin
   ```
6. **Chọn winner**: nhánh hợp lệ có `geometric_reward` cao nhất → `commit` → tiến sang step i+1.

---

## Tóm tắt bảng trọng số

| Thành phần | Giá trị | Vai trò |
|---|---|---|
| `pbrs_scale` | 4.0 | Khuếch đại chênh lệch Φ mỗi step (~±3) |
| `grasp_bonus` | 5.0 (±) | Mốc kẹp được vật / phạt tuột |
| `basket_bonus` | 10.0 | Mốc cuối: vào giỏ |
| `fling_penalty` | 1.0 | Phạt đập/hất vật (>1 m/s) |
| `invalid_format_margin` | 1.0 | Đẩy nước sai format xuống dưới mọi nước hợp lệ |
| `stall_penalty` | 0.5 | Phạt tay đứng yên (no-op) |
| Φ.approach / low_penalty / h_standoff | 1.0 / 2.0 / 0.10m | Hành lang chữ L pha tiếp cận |
| Φ.grasped / carry / in_basket | 1.0 / 1.0 / 2.0 | Thế năng pha mang vật |

**Triết lý thiết kế**: PBRS cung cấp tín hiệu dày dẫn đường mà không làm lệch chính sách tối ưu; các mốc rời rạc tạo điểm nhấn mạnh tại các bước quyết định; format gate + anti-stall + fling penalty là các rào chống gian lận. Toàn bộ tiến trình được đo khách quan từ tọa độ thật, không dựa vào nhãn pha do model tự khai — nên model không thể "nói dối để ăn điểm".

Bạn muốn tôi đi sâu vào phần nào (ví dụ vì sao chọn hành lang chữ L, hay cách tinh chỉnh trọng số) không?

---
