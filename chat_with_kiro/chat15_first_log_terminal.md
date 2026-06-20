# Bên dưới là một đoạn log sau khi tôi chạy lệnh

Exported: 6/20/2026, 8:58:35 PM

---

**User:**

Bên dưới là một đoạn log sau khi tôi chạy lệnh
python -m train.grpo_train     --model Qwen/Qwen3-VL-2B-Instruct     --game-url http://127.0.0.1:8200     --n-parallel 1 --group-size 2 --use-vllm     --vllm-gpu-util 0.45 --max-model-len 4096 --img-w 256 --img-h 192
Vòng lặp huấn luyện đã chạy được nhưng thực sự tôi không biết nó đã chạy đúng chưa
[reward] ep0 lvl=map4_lvl0_cube_red step=5 n_valid=2 r=[1.0, 1.0] finished=False
{'loss': 0.0, 'grad_norm': 0.0, 'learning_rate': 9.933333333333333e-07, 'num_tokens': 185700.0, 'completions/mean_length': 279.0, 'completions/min_length': 260.0, 'completions/max_length': 298.0, 'completions/clipped_ratio': 0.0, 'completions/mean_terminated_length': 279.0, 'completions/min_terminated_length': 260.0, 'completions/max_terminated_length': 298.0, 'rewards/env_reward/mean': 1.0, 'rewards/env_reward/std': 0.0, 'reward': 1.0, 'reward_std': 0.0, 'frac_reward_zero_std': 1.0, 'sampling/sampling_logp_difference/mean': 0.012386453337967396, 'sampling/sampling_logp_difference/max': 0.18529629707336426, 'sampling/importance_sampling_ratio/min': 0.15308794379234314, 'sampling/importance_sampling_ratio/mean': 0.4365749955177307, 'sampling/importance_sampling_ratio/max': 0.7200620174407959, 'entropy': 0.7233550548553467, 'clip_ratio/low_mean': 0.0, 'clip_ratio/low_min': 0.0, 'clip_ratio/high_mean': 0.0, 'clip_ratio/high_max': 0.0, 'clip_ratio/region_mean': 0.0, 'step_time': 5.379032750992337, 'epoch': 0.02}
1%|▋                                                                                                            | 41/6000 [03:14<8:16:16,  5.00s/it]╭────────────────────────────────────────────────────────────────────── Step 41 ──────────────────────────────────────────────────────────────────────╮
│ ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━┓ │
│ ┃ Prompt                                                     ┃ Completion                                                ┃ env_reward ┃ Advantage ┃ │
│ ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━┩ │
│ │ system                                                     │ {                                                         │       1.00 │      0.00 │ │
│ │ Bạn là bộ điều khiển thị giác (vision-based controller)    │   "reasoning": "Trong ảnh 1, đầu kẹp (gripper) đang ở vị  │            │           │ │
│ │ cho cánh tay robot 6 bậc tự do SO-101 trong môi trường mô  │ trí cận cảnh, nhìn thấy một khối lập phương màu đỏ được   │            │           │ │
│ │ phỏng MuJoCo. Nhiệm vụ của bạn là QUAN SÁT hai ảnh camera  │ gắp ở phía trước, gần kẹp. Vị trí kẹp thẳng hàng với vật. │            │           │ │
│ │ mô tả hiện trạng, rồi DỰ ĐOÁN góc quay mục tiêu (radian)   │ Trong ảnh 2, vật màu đỏ nằm ở trung tâm, cao hơn sàn, gần │            │           │ │
│ │ cho 6 khớp để cánh tay GẮP đúng vật thể được yêu cầu rồi   │ với một giỏ ở phía trên và gần gần mặt đất. Vị trí của    │            │           │ │
│ │ MANG VÀ THẢ vật đó VÀO GIỎ.                                │ vật và giỏ trong không gian được xác định dựa trên vị trí │            │           │ │
│ │                                                            │ ở 2 ảnh. Kẹp đang ở trạng thái mở, cần di chuyển đến phía │            │           │ │
│ │ Toàn bộ nhiệm vụ là một chu trình gắp-và-thả               │ trên giỏ để thả vật vào.",                                │            │           │ │
│ │ (pick-and-place):                                          │   "target_object": "cube_red",                            │            │           │ │
│ │ tiếp cận vật -> căn chỉnh -> kẹp -> nâng lên -> di chuyển  │   "phase": "carry",                                       │            │           │ │
│ │ tới phía trên giỏ -> mở kẹp thả vật vào giỏ. Chỉ báo       │   "gripper_action": "hold",                               │            │           │ │
│ │ done=true khi vật đã nằm trong giỏ.                        │   "joint_angles_rad": {                                   │            │           │ │
│ │                                                            │     "shoulder_pan": 0.0,                                  │            │           │ │
│ │ Nguyên tắc bắt buộc:                                       │     "shoulder_lift": -0.8,                                │            │           │ │
│ │ 1. Chỉ suy luận dựa trên những gì NHÌN THẤY trong 2 ảnh,   │     "elbow_flex": 1.2,                                    │            │           │ │
│ │ không bịa thông tin không quan sát được.                   │     "wrist_flex": 0.9,                                    │            │           │ │
│ │ 2. Mọi góc đầu ra phải là radian và NẰM TRONG giới hạn cho │     "wrist_roll": 1.57,                                   │            │           │ │
│ │ phép của từng khớp. Tuyệt đối không vượt biên.             │     "gripper": 1.2                                        │            │           │ │
│ │ 3. Điều khiển theo VÒNG KÍN: mỗi lần chỉ đề xuất bước di   │   },                                                      │            │           │ │
│ │ chuyển an toàn, nhỏ vừa phải tiến gần mục tiêu hiện tại    │   "done": false,                                          │            │           │ │
│ │ của pha.                                                   │   "confidence": 0.95                                      │            │           │ │
│ │ 4. Chỉ ĐÓNG kẹp khi đầu kẹp đã thẳng hàng và đủ gần vật    │ }                                           

---

**Assistant:**

Tôi sẽ đọc các file liên quan để đánh giá xem vòng lặp huấn luyện đã chạy đúng chưa.

Vòng lặp chạy được, nhưng tôi thấy một dấu hiệu rõ ràng là **mô hình thực tế chưa học được gì**. Để tôi xác nhận bằng cách đọc các hàm reward.

Tôi đã có đủ thông tin để chẩn đoán. Để tôi xem nhanh hàm `potential` vì đây là thứ quyết định geometric reward.

Vòng lặp **chạy được về mặt kỹ thuật** (kết nối game service, sinh completion, chấm reward, advance môi trường đều OK), nhưng nhìn vào số liệu thì **mô hình chưa thực sự học gì cả**. Đây là điểm quan trọng nhất.

## Bằng chứng: gradient = 0

Mấy chỉ số này là "khẩu súng bốc khói":

```
loss: 0.0          grad_norm: 0.0
reward: 1.0        reward_std: 0.0
frac_reward_zero_std: 1.0
r=[1.0, 1.0]
```

Cách GRPO hoạt động: trong mỗi nhóm G completion, advantage = (reward − mean) / std. Nếu **mọi completion trong nhóm có reward bằng nhau** thì std = 0 → advantage = 0 → loss = 0 → grad_norm = 0 → **trọng số không được cập nhật**.

`frac_reward_zero_std: 1.0` nghĩa là **100% số nhóm có phương sai reward bằng 0**. Tức là từ đầu tới giờ chưa có một bước nào tạo ra gradient học được. Learning rate có tăng/giảm theo scheduler nhưng vì grad = 0 nên model đứng yên.

## Vì sao reward luôn = 1.0

Tổng reward = `format (gate 0/1)` + `geometric (PBRS)`.

- Cả 2 completion đều đúng JSON → format = 1.0 cho cả hai.
- `r=[1.0, 1.0]` (đã làm tròn 3 số) → **geometric ≈ 0.0 cho cả hai**.

Geometric là PBRS: `10 * (potential(after) − potential(before))`. Nó bằng ~0 khi `potential(after) ≈ potential(before)`, tức trạng thái **gần như không đổi** giữa before và sau khi branch. Hai nguyên nhân có thể:

1. **Hai completion sinh ra góc khớp gần như giống nhau** (group_size=2 quá nhỏ, model giai đoạn đầu ra JSON na ná nhau) → hai branch cho cùng một trạng thái → geo giống nhau và bằng nhau.
2. **Branch không làm cánh tay di chuyển đủ để `d_grip_obj` đổi** → potential không đổi → geo ≈ 0 cho mọi nhánh.

Lưu ý: bản thân việc format cộng hằng số 1.0 KHÔNG gây zero-std (nó bị trừ đi trong mean). Thủ phạm thật sự là **geometric không tạo ra khác biệt giữa các completion**.

## Cần kiểm tra / sửa

Theo thứ tự ưu tiên:

1. **Tăng `--group-size`** lên 8–16. Với 2 mẫu thì xác suất hai mẫu trùng reward rất cao. Nhóm lớn hơn → đa dạng hơn → mới có phương sai để học. Đây là việc dễ nhất, làm trước.

2. **Xác minh branch có thực sự thay đổi trạng thái không.** In thử `before.d_grip_obj` và `after.d_grip_obj` (hoặc `potential`) trong `score_and_advance`. Nếu chúng bằng nhau y hệt giữa các nhánh → có bug ở tầng `env_client.branch` (không apply góc, hoặc trả về measure của trạng thái cũ). Nếu chúng khác nhau chút xíu nhưng làm tròn về 0 → tăng `pbrs_scale` để khuếch đại tín hiệu.

3. **Kiểm tra `importance_sampling_ratio` thấp** (mean ≈ 0.44, min ≈ 0.15). Lý tưởng nên gần 1.0. Với ~279 token và `logp_difference` per-token nhỏ (~0.012) thì tỉ lệ trôi khỏi 1 là bình thường ở mức độ nào đó, nhưng 0.44 hơi thấp. Hiện tại advantage = 0 nên nó chưa ảnh hưởng, nhưng **khi reward bắt đầu có phương sai**, mismatch giữa logprob của vLLM (sinh) và model train có thể gây cập nhật lệch/bất ổn. Đáng theo dõi, thường liên quan đến việc sync LoRA adapter sang vLLM trong chế độ colocate.

Tóm lại: pipeline thông suốt, nhưng đang ở trạng thái "chạy không tải". Bước đầu tiên tôi khuyên là tăng `group-size` rồi log thêm geometric reward trước/sau branch để xem nó có khác 0 và khác nhau giữa các nhánh không.

Bạn muốn tôi thêm log chi tiết vào `score_and_advance` (in `d_grip_obj`/`potential` before-after cho từng nhánh) để xác định nhanh nguyên nhân số 1 hay số 2 không?

---
