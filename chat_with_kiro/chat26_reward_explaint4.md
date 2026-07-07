# hãy giúp tôi giải thích đầy đủ cách hoạt động c...

Exported: 7/6/2026, 9:01:55 PM

---

**User:**

hãy giúp tôi giải thích đầy đủ cách hoạt động của cơ chế chấm điểm hiện tại

---

**Assistant:**

Tôi đã đọc toàn bộ hệ thống chấm điểm. Dưới đây là giải thích đầy đủ về cách cơ chế chấm điểm hiện tại hoạt động.

## Tổng quan kiến trúc

Đây là pipeline huấn luyện GRPO cho VLM điều khiển cánh tay robot SO-101 gắp vật bỏ vào giỏ. Cơ chế chấm điểm dựa trên **single-step GRPO**: mỗi bước của môi trường là một mẫu huấn luyện. Với mỗi bước, policy sinh ra `G` completions (nhóm), mỗi completion được chấm điểm, và GRPO học bằng cách so sánh điểm giữa các completion trong cùng nhóm (advantage nội nhóm).

Có **3 tầng chấm điểm** theo thứ tự ưu tiên (định nghĩa trong `rewards/__init__.py`):

1. **FORMAT** (gate 0/1) — `format_reward.py`
2. **GEOMETRIC** (tín hiệu chính, PBRS) — `geometric_reward.py`
3. **JUDGE** (VLM, bổ sung, chưa bật) — `judge_reward.py`

Điểm mấu chốt trong thiết kế: mọi tiến độ được đo **khách quan từ toạ độ thật 3D trong MuJoCo** (privileged info), *không* dựa vào nhãn pha mà model tự khai — nên model không thể gian lận bằng cách nói dối.

---

## Tầng 1 — FORMAT (cổng chặn cứng)

File `check_format()` phân tích JSON đầu ra của VLM và trả về hợp lệ/không. Nó yêu cầu:

- Là object JSON, có đủ 4 khoá bắt buộc: `phase`, `gripper_action`, `joint_angles_rad`, `done`
- `phase` ∈ {approach, align, grasp, lift, carry, release}
- `gripper_action` ∈ {open, close, hold}
- `done` là bool
- `joint_angles_rad` là object chứa đủ **cả 6 khớp** (theo `JOINT_NAMES`), mỗi góc là số

**Vai trò gate:** completion sai format sẽ *không* được gửi cho game và *không* được chấm tiếp (tiết kiệm chi phí branch + judge). Nhưng cách phạt không phải gán 0 tuyệt đối — mà là **phạt theo thứ hạng** (xem phần tổng hợp bên dưới).

---

## Tầng 2 — GEOMETRIC (tín hiệu chính, PBRS)

Đây là trái tim của cơ chế. Reward cho một bước (before → after) gồm 4 thành phần trong `geometric_reward()`:

### (1) PBRS — Potential-Based Reward Shaping
```
reward = pbrs_scale * (Φ(after) − Φ(before))
```
- `pbrs_scale = 4.0` (đã hạ từ 10 xuống 4 để giảm biên độ swing mỗi step xuống ~±3, tránh lấn át các mốc rời rạc).
- PBRS đảm bảo về mặt lý thuyết là **không bóp méo chính sách tối ưu** — chỉ làm reward "dày" hơn để dễ học.

**Hàm thế năng Φ** (trong `sim/metrics.py::potential`) là phần tinh tế nhất:

**Pha APPROACH (chưa kẹp)** — dùng "hành lang chữ L", ép robot kẹp từ TRÊN xuống:
```
belowness = clip((h − dz) / h, 0, 1)     # 1 khi đầu kẹp ngang/dưới vật; 0 khi đủ cao
L = d_xy · (1 + low_penalty · belowness) + |dz|
Φ = − approach · L
```
Ý tưởng: lệch ngang `d_xy` bị phạt **nặng hơn khi đầu kẹp còn thấp** (`belowness` lớn). Vì thế cách "rẻ" nhất để tiến khi đang ở thấp là **nâng lên trước** → canh ngang trên vật → hạ thẳng xuống. Điều này tránh húc văng vật, và vì Φ là một hàm khoảng-cách-tiến-trình **đơn điệu duy nhất** (không đổi gốc quy chiếu giữa chừng) nên không còn "thung lũng" Φ gây reward âm trên chính quỹ đạo thành công.

**Pha sau khi kẹp:**
```
Φ = grasped + carry · (−d_obj_basket)
```
Tức đã kẹp thì càng gần giỏ càng tốt.

Quan trọng: việc **chuyển từ pha approach sang carry được kích hoạt bởi sự kiện `grasped` đo khách quan**, không do nhãn pha model khai.

### (2) Mốc rời rạc — grasp
```
+grasp_bonus (5.0)  khi grasped chuyển False → True (vừa kẹp được)
−grasp_bonus        khi tuột vật (grasped True → False và chưa vào giỏ)
```
Phạt đối xứng để chống mánh "kẹp rồi buông" ăn điểm liên tục.

### (3) Mốc cuối — basket
```
+basket_bonus (10.0)  khi in_basket chuyển False → True
```

### (4) Chống reward-hacking — fling
```
−fling_penalty (1.0)  nếu obj_speed_after > fling_speed (1.0 m/s)
```
Phạt hành vi "đập/hất" vật cho văng gần giỏ thay vì gắp thật.

**Các trạng thái khách quan** được đo trong `sim/metrics.py`:
- `is_grasped`: đầu kẹp đủ gần tâm vật (≤5cm) VÀ vật đã được nhấc cao hơn vị trí nghỉ (≥3cm).
- `obj_in_basket`: vật nằm trong khung xy của giỏ, dưới miệng và trên đáy giỏ.

---

## Tầng 3 — JUDGE (VLM, chưa bật)

`judge_reward.py` là tuỳ chọn nâng cấp sau. Nó dùng chính base model (self-host bằng vLLM) để "nhìn" ảnh trạng thái trước và các ảnh kết quả của từng nhánh, rồi chấm 0–100 cho mỗi nhánh. Bài học thiết kế ghi trong file: yêu cầu judge **chấm phân biệt rõ** giữa các nhánh (tránh chấm gần bằng nhau làm advantage GRPO ≈ 0, không học được). Phần này cần GPU nên chưa chạy trên Mac.

---

## Cách tổng hợp reward trong một nhóm GRPO (`train/rollout.py`)

Đây là nơi 3 tầng ghép lại thành điểm cuối cho từng completion trong một step:

1. **Format gate:** chạy `check_format` cho cả G completion, đánh dấu `valid_mask`.
2. **Branch:** chỉ gửi các lệnh **hợp lệ** cho game `/branch` → nhận trạng thái i+1.
3. **Geometric:** chấm `geometric_reward(before, after)` cho từng nhánh hợp lệ.
4. **Anti-stall:** phạt `stall_penalty = 0.5` nếu completion ra góc gần y hệt pose hiện tại (cả khớp tay LẪN gripper không đổi > `stall_eps`). Mục đích tạo phương sai để gradient ưu tiên "có di chuyển" hơn "đứng yên". Lưu ý "giữ tay + đóng kẹp" lúc gắp *không* bị phạt oan.
5. **Tổng hợp:**
   ```
   completion hợp lệ:   total = geo_reward − (stall_penalty nếu stall)
   completion sai format: total = (nước hợp lệ tệ nhất) − invalid_format_margin (1.0)
   ```
   Đây chính là **gate theo thứ hạng**: completion sai format luôn có điểm thấp hơn *mọi* nước hợp lệ → advantage GRPO của nó luôn âm → model học "đừng sai format" mà không cần gán 0 cứng.
6. **Chọn winner:** nhánh hợp lệ có `geometric_reward` cao nhất → `/commit` → tiến môi trường sang trạng thái i+1 để đi tiếp.

---

## Tóm tắt luồng tín hiệu

```
VLM sinh G lệnh
   │
   ├─ FORMAT sai  ──► total = worst_valid − margin   (advantage luôn âm)
   │
   └─ FORMAT đúng ──► branch trong MuJoCo
                          │
                          ├─ PBRS: 4.0·ΔΦ  (dày, dẫn đường theo hành lang chữ L)
                          ├─ +5 kẹp được / −5 tuột vật
                          ├─ +10 vào giỏ
                          ├─ −1 nếu văng vật
                          └─ −0.5 nếu đứng yên (stall)
                                 │
                                 ▼
                          total_reward  ──► GRPO advantage nội nhóm
                                 │
                          winner (geo cao nhất) ──► commit sang step kế
```

**Triết lý cốt lõi:** privileged geometric reward là tín hiệu chính (chính xác, miễn phí, tức thời từ toạ độ thật), format chỉ là cổng chặn theo thứ hạng, và judge VLM để dành bổ sung sau. Mọi mốc quan trọng (`grasped`, `in_basket`) đều đo khách quan để model không thể gian lận bằng nhãn tự khai.

Bạn có muốn tôi đi sâu vào phần nào — ví dụ cách các trọng số ảnh hưởng đến hành vi học, hay cách advantage GRPO được tính từ `total_rewards` trong `grpo_train.py`?

---
