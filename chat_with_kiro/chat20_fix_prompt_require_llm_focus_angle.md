# Khi tôi chạy thực nghiệm tôi nhận thấy rằng:

Exported: 6/26/2026, 5:55:19 PM

---

**User:**

Khi tôi chạy thực nghiệm tôi nhận thấy rằng:
Mô hình không chịu sinh ra góc quay khác mà cứ lấy y chang ví dụ trong prompt. Robot không di chuyển. CHỈ tập trung thay đổi câu reasoning.
Làm sao để model nhận ra việc sinh ra góc quay mới là quan trọng.

---

**Assistant:**

Để hiểu rõ vấn đề, tôi cần xem prompt đang dùng, reward function, và pipeline training. Để tôi đọc các file liên quan.

Tôi cần xem cách build prompt và cấu trúc completion. Để tôi tìm các file liên quan.

Found the key file. Let me read the full prompt builder.

Tôi đã thấy nguyên nhân cốt lõi nằm ở phần ví dụ trong prompt. Để xác nhận đầy đủ, tôi xem thêm policy (tham số sampling) và rollout (cách reward gắn vào completion).

Tôi cần xem `StateMeasure` và `potential()` để biết reward có "nhìn thấy" được việc tay máy đứng yên hay không.

Tôi đã đủ dữ liệu để chẩn đoán. Kiểm tra nhanh xem `StepObs` có mang góc khớp hiện tại không (để biết có thể đưa pose hiện tại vào prompt).

Tôi đã có đủ bức tranh. Đây là chẩn đoán và bản sửa.

## Chẩn đoán: 2 nguyên nhân cộng hưởng

**1. Prompt "mời" model copy.** Trong `prompt_builder.py`, phần `_output_schema_example_from_spec` dựng một JSON ví dụ hợp lệ với `joint_angles_rad` = **đúng home pose**. Một VLM instruct chưa được train cho task này không có cơ sở để tự nghĩ ra góc, nên nó bám vào con số cụ thể duy nhất nó thấy = ví dụ = home. Kết quả: target = home = tay không đổi tư thế. Nó chỉ đổi `reasoning` vì đó là phần text tự do.

**2. GRPO không có tín hiệu để sửa.** GRPO học từ *phương sai trong nhóm* G completion: `advantage_i = reward_i − mean(group)`. Nếu cả G completion đều ra cùng góc home:
- tay không di chuyển → `before ≈ after` → PBRS `Φ(after) − Φ(before) ≈ 0` cho tất cả
- mọi total reward bằng nhau (chỉ còn format 1.0) → advantage ≈ 0 → gradient ≈ 0
- token duy nhất khác nhau giữa các completion nằm ở `reasoning`, nên mọi nhiễu reward nhỏ đều dồn về token reasoning. Model "học" được rằng nghịch reasoning là chỗ tạo khác biệt, còn góc thì không bao giờ. Đúng triệu chứng bạn thấy.

Vậy phải đánh cả hai: (a) chặn việc copy trong prompt + cho model neo vào pose hiện tại; (b) tạo phương sai/tín hiệu cho "có di chuyển" để GRPO có gradient trên token góc.

Tôi sẽ sửa 3 chỗ.

Giờ tôi sửa. Bắt đầu với `prompt_builder.py`: thêm pose hiện tại làm điểm neo, đổi ví dụ để không còn là "đáp án copy được", và thêm luật cấm copy.

Now add a system-prompt rule and thread `current_angles` through the user prompt.

Now update the example label and thread `current_angles` through `build_messages*`:

Now wire the current pose into the rollout and add an anti-stall reward term so GRPO gets a gradient that ranks "moved" above "copied current pose".

Now add an anti-stall penalty to the rollout reward. Let me update the constructor and the reward computation.

Now let me check how `grpo_train.py` feeds rewards, to confirm the anti-stall penalty actually reaches the gradient, and check the sampling temperature setting.

The reward path uses `manager.score_and_advance(...).rewards`. Let me confirm that maps to `total_rewards` (which now includes the anti-stall penalty).

Important: the actual GPU training path is `EpisodeManager`, not `EpisodeRollout` (that's the Mac sanity-test path). I need to apply the same two fixes there too.

Let me add the `# 4)` comment back and surface `n_stalled` in info for logging.

Now bump the sampling temperature in the GRPO config so the G completions actually diverge in angles (without exploration, GRPO has nothing to contrast):

Now verify the prompt builder renders correctly and run the existing tests that exercise these paths.

Prompt renders correctly: current pose shown, placeholder example clearly fake with warnings, system rule 7 added. Let me run the standalone tests that exercise these modules.

Done. All edits compile and the anti-stall logic behaves correctly (copying the current pose → −0.5, moving → 0).

## Tóm tắt thay đổi

Vấn đề có 2 nguyên nhân cộng hưởng, nên tôi sửa cả hai.

**Prompt mời model copy** (`vlm/prompt_builder.py`):
- Ví dụ JSON trước đây dùng đúng `joint_angles_rad` = home pose → model bê nguyên. Giờ ví dụ dùng số **placeholder rõ ràng là giả** kèm nhãn "JSON SHAPE ONLY — DO NOT copy them".
- Thêm khối **`# CURRENT ARM POSE`** liệt kê góc hiện tại của 6 khớp làm điểm neo thật, kèm câu: muốn tay di chuyển thì output PHẢI khác các số này.
- Thêm **luật 7** trong system prompt: bắt buộc đổi góc, cấm copy home/example, nói rõ "output = pose hiện tại là sai".

**GRPO không có tín hiệu trên token góc** (`train/episode_manager.py` — đây mới là đường train thật trên GPU; `train/rollout.py` là đường test Mac, tôi sửa đồng bộ cả hai):
- Truyền `ctrl_angles` hiện tại vào prompt mỗi step.
- Thêm **anti-stall penalty**: completion nào ra góc gần y hệt pose hiện tại (max|Δ| < `stall_eps`, bỏ qua `gripper`) bị trừ `stall_penalty` (mặc định 0.5). Điều này tạo phương sai trong nhóm G → advantage khác 0 → gradient đẩy model ra khỏi việc đứng yên. Số completion bị phạt được log qua `n_stalled`.

**Thiếu khám phá** (`train/grpo_train.py`):
- Tăng `temperature` 1.0 → 1.2 và thêm `top_p=0.95`, để G completion thực sự khác nhau về góc. Nếu không có đa dạng, GRPO chẳng có gì để so sánh.

Cơ chế tại sao trước đây kẹt: tất cả G completion copy home → tay không động → PBRS `Φ(after)−Φ(before)=0` cho mọi nhánh → reward bằng nhau → advantage = 0 → gradient = 0, chỉ còn reasoning (text tự do) là khác nhau nên model học nghịch reasoning. Anti-stall + temperature + bỏ ví dụ home phá vỡ đối xứng đó.

Vài lưu ý khi chạy lại:
- `stall_penalty`/`stall_eps` là tham số của `EpisodeManager`; có thể điều chỉnh. Nếu thấy model thrashing (đổi góc lung tung để né phạt mà không tiến bộ), giảm `stall_penalty` — vì PBRS đã thưởng di chuyển có mục đích, anti-stall chỉ cần đủ để vượt nhiễu.
- Theo dõi `n_stalled` trong log: nó phải giảm dần khi model học được rằng phải sinh góc mới.
- `top_p` sẽ tự bị bỏ nếu phiên bản `trl` không hỗ trợ (`_build_config` lọc field và in cảnh báo).

Tôi chưa chạy `test_episode_manager.py`/`test_rollout.py` vì chúng cần game service đang chạy; các thay đổi chữ ký đều là kwargs có default nên tương thích ngược.

---
