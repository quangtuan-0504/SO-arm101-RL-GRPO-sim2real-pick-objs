# hãy giúp tôi giải thích đầy đủ cách hoạt động c...

Exported: 7/6/2026, 8:03:15 PM

---

**User:**

hãy giúp tôi giải thích đầy đủ cách hoạt động của cơ chế chấm điểm hiện tại

---

**Assistant:**

Tôi sẽ đọc các file liên quan đến cơ chế chấm điểm để giải thích đầy đủ.

Tôi đã đọc đủ các file cốt lõi. Dưới đây là giải thích đầy đủ về cơ chế chấm điểm (reward) hiện tại của pipeline GRPO.

## Tổng quan: 3 lớp chấm điểm

Theo thiết kế trong `rewards/__init__.py`, có 3 hàm chấm điểm theo thứ tự ưu tiên:

1. **format_reward** — cổng (gate) kiểm tra cú pháp JSON
2. **geometric_reward** — tín hiệu CHÍNH (PBRS từ toạ độ thật MuJoCo)
3. **judge_reward** — VLM chấm bổ sung (chưa dùng, để nâng cấp sau)

Hiện tại hệ thống chạy bằng lớp 1 + lớp 2. Đơn vị chấm điểm là **mỗi step của môi trường = một nhóm GRPO** (single-step GRPO): cùng một trạng thái, policy sinh ra G lệnh (completions), mỗi lệnh nhận một reward, GRPO học từ chênh lệch reward trong nhóm.

---

## Lớp 1 — Format gate (`rewards/format_reward.py`)

`check_format()` parse text đầu ra của VLM và kiểm tra:
- Parse được JSON (chịu được khi bọc trong ` ```json ... ``` `)
- Có đủ các khoá bắt buộc: `phase`, `gripper_action`, `joint_angles_rad`, `done`
- `phase` ∈ {approach, align, grasp, lift, carry, release}
- `gripper_action` ∈ {open, close, hold}
- `done` là bool
- `joint_angles_rad` có đủ 6 khớp và mỗi góc là số

Điểm mấu chốt: **format KHÔNG cộng/trừ điểm cố định**. Nó là **cổng theo THỨ HẠNG**:
- Lệnh sai format **không** được gửi vào game (tiết kiệm chi phí branch).
- Reward của nó bị đặt = (nước hợp lệ tệ nhất trong nhóm − `invalid_format_margin`, mặc định 1.0).

Nhờ vậy, advantage GRPO của lệnh sai format **luôn âm** so với mọi lệnh hợp lệ → model bị đẩy ra khỏi việc sinh sai cú pháp mà không cần chọn một con số phạt tuyệt đối.

---

## Lớp 2 — Geometric reward (tín hiệu chính, PBRS)

Đây là trái tim của cơ chế. Nó lấy trực tiếp từ toạ độ 3D thật trong MuJoCo (privileged info — chỉ dùng lúc train để chấm, còn policy vẫn chỉ nhìn ảnh). Reward cho một bước `before → after` (`geometric_reward.py`) gồm 4 thành phần:

### (1) PBRS — thành phần liên tục chính
```
reward = pbrs_scale * (Φ(after) − Φ(before))
```
`pbrs_scale = 4.0`. Đây là *potential-based reward shaping*: vì reward là hiệu của một hàm thế năng Φ thuần trạng thái, nó **không bóp méo chính sách tối ưu**, chỉ làm tín hiệu học dày và mượt hơn.

### (2) Mốc rời rạc — kẹp được vật
```
if after.grasped and not before.grasped:  reward += grasp_bonus (5.0)
if before.grasped and not after.grasped and not after.in_basket:  reward −= grasp_bonus
```
Thưởng một lần khi vừa kẹp được, và **phạt đối xứng** khi tuột vật (chống mánh "kẹp rồi buông" để ăn bonus lặp lại).

### (3) Mốc cuối — vào giỏ
```
if after.in_basket and not before.in_basket:  reward += basket_bonus (10.0)
```

### (4) Chống reward-hacking — phạt văng vật
```
if obj_speed_after > fling_speed (1.0 m/s):  reward −= fling_penalty (1.0)
```
Ngăn hành vi đập/hất mạnh vật để nó bay gần giỏ.

---

## Hàm thế năng Φ (`sim/metrics.py::potential`) — phần tinh vi nhất

Φ được thiết kế thành **một hàm khoảng-cách-tiến-trình duy nhất**, đơn điệu dọc quỹ đạo lý tưởng, chia theo hai pha do sự kiện khách quan `grasped` kích hoạt (không do nhãn `phase` model tự khai → không gian lận được).

**Pha APPROACH (chưa kẹp) — "hành lang chữ L":**
```
belowness = clip((h_standoff − dz) / h_standoff, 0, 1)
L = d_xy * (1 + low_penalty * belowness) + |dz|
Φ = −approach * L
```
- `d_xy`: lệch ngang đầu kẹp ↔ vật; `dz`: chênh cao (đầu kẹp trên vật là dương)
- `belowness` = 1 khi đầu kẹp còn thấp/ngang vật (dễ húc văng), = 0 khi đã đủ cao (`h_standoff = 0.10 m`)
- Ý nghĩa: khi đang ở thấp, lệch ngang bị phạt nặng gấp `(1 + low_penalty)` = 3 lần → cách "rẻ" nhất để tăng Φ là **nâng lên trước, rồi canh ngang trên vật, rồi hạ thẳng xuống**. Đây là cách ép robot đi "đường trên" an toàn, thay cho term "danger" rời rạc trước đây.
- Vì không đổi gốc quy chiếu giữa chừng, Φ không tạo "thung lũng" quanh điểm canh thẳng (nguyên nhân gây reward per-step âm trên chính quỹ đạo thành công).

**Pha CARRY (đã kẹp):**
```
Φ = grasped(1.0) + carry * (−d_obj_basket)
```
Càng gần tâm giỏ Φ càng cao. Có bậc thang `grasped` để Φ nhảy lên khi vừa kẹp được.

**Cộng thêm khi vào giỏ:** `Φ += in_basket (2.0)`.

Trọng số Φ mặc định (`PHI_WEIGHTS`): `approach=1.0`, `low_penalty=2.0`, `h_standoff=0.10`, `grasped=1.0`, `carry=1.0`, `in_basket=2.0`.

---

## Cách trạng thái được đo khách quan (`metrics.py`)

- **`is_grasped`**: đầu kẹp đủ gần tâm vật (≤ 5 cm) **VÀ** vật đã được nhấc cao hơn vị trí nghỉ ban đầu ≥ 3 cm. Hoàn toàn suy từ toạ độ, không từ nhãn model.
- **`obj_in_basket`**: xy nằm trong khung lòng giỏ, z dưới miệng giỏ và trên đáy giỏ.

---

## Tổng hợp trong một nhóm GRPO (`train/rollout.py::collect_step`)

Với mỗi step:
1. Build prompt (2 ảnh wrist + stand, instruction, bảng khớp, góc hiện tại).
2. Policy sinh G completions.
3. **Format gate**: đánh dấu hợp lệ/không; chỉ branch các lệnh hợp lệ vào game.
4. Chấm `geometric_reward(before, after)` cho từng nhánh hợp lệ.
5. **Anti-stall**: phạt `stall_penalty = 0.5` cho lệnh no-op thật sự (cả khớp tay LẪN gripper gần như không đổi so với pose hiện tại) — tạo phương sai để GRPO ưu tiên "có di chuyển" hơn "đứng yên". Lưu ý: "giữ tay + đóng kẹp" lúc gắp không bị phạt oan vì gripper có thay đổi.
6. Tổng hợp:
   ```
   total[hợp lệ]   = geo − (stall_penalty nếu stall)
   invalid_value   = min(total các nước hợp lệ) − invalid_format_margin
   total[sai format] = invalid_value
   ```
7. Chọn **winner** = nước hợp lệ có `geo` cao nhất → `/commit` để môi trường tiến sang step kế tiếp.

---

## Nối vào trainer trl (`train/grpo_train.py`)

`env_reward_func(prompts, completions, episode_id, ...)`:
- Gom completions theo `episode_id` (mỗi episode = 1 nhóm G liền nhau trong batch).
- Gọi `manager.score_and_advance(eid, comp)` → trả về mảng reward phẳng đúng thứ tự, đồng thời đã commit winner để advance môi trường.
- Ghi metric JSONL và (tuỳ chọn) lưu ảnh winner.

Ràng buộc quan trọng: vì môi trường advance dựa trên completion, batch phải đọc dataset **lười và không shuffle/prefetch** (`steps_per_generation=1`, `dataloader_num_workers=0`, `shuffle_dataset=False`) để prompt của step sau chỉ được lấy sau khi winner của step trước đã commit. `beta=0` (tắt KL lúc khởi đầu), `temperature=1` để G completions đủ đa dạng cho GRPO có phương sai học.

---

## Tóm tắt triết lý thiết kế

| Cơ chế | Vai trò |
|---|---|
| Format gate (thứ hạng) | Loại cú pháp sai mà không cần con số phạt tuyệt đối |
| PBRS (Φ) | Tín hiệu dày, mượt, không bóp méo optimum; hành lang chữ L ép tiếp cận từ trên |
| Bonus rời rạc (grasp/basket) | Tín hiệu mạnh cho khoảnh khắc chính xác mà PBRS thưởng rất mờ |
| Phạt tuột/văng/stall | Chống các mánh reward-hacking và no-op |
| Đo khách quan (grasped/in_basket) | Chuyển pha dựa trên vật lý thật, model không tự khai để gian lận |

Nếu bạn muốn, tôi có thể vẽ sơ đồ luồng reward, hoặc phân tích một dòng metrics JSONL cụ thể để xem cơ chế này thể hiện thế nào trên một episode thật.

---
