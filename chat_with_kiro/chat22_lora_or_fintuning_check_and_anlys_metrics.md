# quá trình huấn luyện nên dùng Lora hay finetuni...

Exported: 7/4/2026, 10:45:08 PM

---

**User:**

quá trình huấn luyện nên dùng Lora hay finetuning toàn bộ tham số

---

**Assistant:**

Để trả lời chính xác cho dự án của bạn, tôi xem qua setup GRPO training hiện tại đã.

Câu hỏi tư vấn, nên tôi trả lời dựa trên setup thực tế trong `grpo_train.py` của bạn (Qwen2.5-VL-3B, GRPO qua TRL, vLLM colocate, GPU đơn ~40GB).

## Khuyến nghị: dùng LoRA

Với chính bối cảnh dự án này, **LoRA là lựa chọn hợp lý hơn nhiều**, không phải full fine-tune. Lý do cụ thể:

**1. Ràng buộc VRAM trong chế độ colocate**

Code của bạn chạy vLLM và trainer **trên cùng GPU** (`vllm_mode="colocate"`, `vllm_gpu_util=0.40`). Nghĩa là:
- ~40% VRAM đã dành cho vLLM (sinh G completions + KV cache cho ảnh).
- Phần còn lại (~60%) phải chứa model weights + gradients + optimizer states.

Full fine-tune một VLM 3B với AdamW cần khoảng:
- weights bf16: ~6GB
- gradients: ~6GB
- optimizer states (m, v fp32): ~24GB
- → ~36GB *chỉ riêng* cho training, chưa kể activations.

Cộng với phần vLLM chiếm, **full fine-tune gần như không vừa GPU 40GB** trong chế độ colocate. Comment trong code của bạn cũng đã nhận ra điều này:

> "LoRA: chỉ train vài % tham số -> optimizer states gần như biến mất, giải phóng VRAM cho bản vLLM + KV cache. Đây là cách giảm bộ nhớ hiệu quả nhất cho GPU 40GB với VLM 3B."

**2. GRPO không cần cập nhật nhiều tham số**

GRPO là giai đoạn RL tinh chỉnh hành vi (policy), không phải dạy kiến thức mới. Bạn chỉ đang định hình lại cách model xuất lệnh điều khiển khớp cho đúng format + tối ưu reward hình học. LoRA rank 16-32 thừa sức cho việc dịch chuyển policy này. Đây cũng là cấu hình chuẩn của hầu hết công thức RLHF/GRPO công khai.

**3. Ổn định hơn, ít collapse hơn**

Full fine-tune một policy 3B với reward signal thưa (single-step branching) dễ làm model "quên" và lệch phân phối mạnh (catastrophic forgetting / reward hacking). LoRA giữ base model đóng băng nên an toàn hơn, dễ rollback.

## Khi nào mới cân nhắc full fine-tune

- Bạn có GPU lớn hơn (A100 80GB / H100, hoặc multi-GPU) **và** tách riêng vLLM sang GPU khác (`vllm_mode="server"` thay vì colocate).
- LoRA đã hội tụ nhưng reward chạm trần và bạn nghi giới hạn dung lượng của adapter (hiếm gặp với task hẹp như điều khiển khớp này).

## Một lưu ý về cấu hình LoRA hiện tại

Trong code bạn đang nhắm target_modules **chỉ các lớp ngôn ngữ** của Qwen (`q/k/v/o_proj`, `gate/up/down_proj`), bỏ qua vision tower:

```python
target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"],
```

Đây là lựa chọn đúng cho task này: nhận thức thị giác (đọc góc khớp, vị trí vật từ 2 camera) đã tốt từ pretrain, bạn chỉ cần điều chỉnh phần reasoning → sinh lệnh. Giữ vision tower đóng băng vừa tiết kiệm VRAM vừa tránh phá hỏng khả năng nhìn.

**Tóm lại:** giữ mặc định LoRA (không bật `--no-lora`). Bắt đầu với `r=16, alpha=32` như đang cấu hình; nếu thấy model học chậm/không đủ sức biểu diễn thì nâng lên `r=32, alpha=64`. Chỉ chuyển full fine-tune khi bạn đổi sang GPU rời cho vLLM.

---

**User:**

============================================================
Tổng step ghi nhận      : 266
Episode kết thúc        : 13
- Thành công (vào giỏ): 0  (0.0%)
- Thất bại            : 13
Step có winner đã kẹp   : 0  (0.0%)
format_rate trung bình  : 0.817
reward_mean trung bình  : 0.259
reward_max  trung bình  : 0.995
[100 step gần nhất] success_rate ~ 0.0% (0/5 episode)
đây là một phần chỉ số của kết quả chạy, tôi không biết rằng liệu pipeline huấn luyện này có thực sự hiệu quả không

---

**Assistant:**

Để đánh giá pipeline có hiệu quả không, tôi cần đọc reward function và cách episode chấm điểm, vì các con số đang mâu thuẫn với nhau.

Tôi đã đọc hết chuỗi reward (format → PBRS → milestone), `EpisodeManager`, và hàm thế năng `potential`. Đây là đánh giá thẳng dựa trên số liệu của bạn.

## Kết luận ngắn: với log này, pipeline CHƯA học được task

Ba con số tự nó mâu thuẫn và chỉ ra vấn đề:

- **grasp 0% / success 0%** trên cả 13 episode → cánh tay **chưa bao giờ kẹp được vật**, tức chưa vượt qua mốc đầu tiên của chuỗi (approach → grasp → carry → basket).
- **reward_max ≈ 0.995 ≈ 1.0** → completion tốt nhất mỗi step gần như chỉ nhận đúng **điểm format (1.0)**, phần hình học (PBRS) đóng góp ~0.
- **reward_mean 0.259 ≪ format_rate 0.817** → trung bình phần geometric + phạt (stall/fling) đang **âm**. Nghĩa là các nước đi trung bình KHÔNG cải thiện thế năng, thậm chí bị phạt đứng yên.

Diễn giải gộp lại: model đang được "thưởng vì xuất JSON đúng cú pháp", **không** phải vì di chuyển tay đúng. Đó là dấu hiệu kinh điển của reward signal hỏng, không phải chỉ thiếu thời gian train.

## Vì sao tín hiệu học bị washout — root cause

**1. Format reward lấn át tín hiệu hình học.**
`total = format(0/1) + geometric`. Format cho phẳng +1.0; còn PBRS một step chỉ là `pbrs_scale * (Φ_after − Φ_before)`. Với một bước nhỏ, delta Φ rất bé. Khi trong nhóm G có lẫn completion sai format (valid=+1 vs invalid=0), **advantage của GRPO bị chi phối bởi chênh lệch format**, không phải chất lượng chuyển động. Kết quả: gradient chủ yếu dạy "viết JSON hợp lệ" (model đã làm được 82%), rồi chững lại, gần như không học motion.

**2. PBRS một-step, G nhánh từ CÙNG một trạng thái.**
Mỗi step, G lệnh phân nhánh từ cùng state, chỉ khác nhau vài độ ở khớp. Tín hiệu học là "trong G nước đi nhỏ na ná nhau, cái nào tăng Φ nhiều nhất". Delta Φ per-step quá nhỏ so với nhiễu → phương sai trong nhóm gần như là nhiễu. Học từ tín hiệu này rất chậm và dễ collapse.

**3. Mốc grasp/basket không bao giờ kích hoạt.**
`grasp_bonus=1.0`, `basket_bonus=5.0` — đây là tín hiệu mạnh nhất, nhưng grasp 0% nên **chưa lần nào được cộng**. Model chỉ thấy gradient của pha approach (cái phễu tiền-kẹp), vốn nhỏ và không tự dẫn tới động tác đóng kẹp thật.

**4. KL beta=0.5 không hề "tắt".**
Comment trong `grpo_train.py` ghi "khởi đầu tắt KL cho đơn giản" nhưng giá trị là `0.5` — đó là neo KL khá mạnh về base model. Base Qwen2.5-VL-3B vốn không biết điều khiển khớp, neo mạnh vào nó sẽ **kìm thăm dò**, giữ policy ở vùng không-kẹp.

**5. Quy mô chạy quá nhỏ.**
266 step (mặc định `total_samples=2000`) là rất ít cho RL. Nên kể cả khi sửa reward, đừng kỳ vọng nhiều ở vài trăm step.

## Cách kiểm chứng trước khi đổ tiền GPU

Trước khi kết luận pipeline đúng/sai, nên tách bạch "reward có đúng không" khỏi "model có học không":

1. **Kiểm tra Φ có đơn điệu trên quỹ đạo lý tưởng chưa.** Viết một script đưa chuỗi pose "tay tiến tới vật → đóng kẹp → nhấc → sang giỏ" (có thể bằng tay/scripted) và in `potential()` từng bước. Nếu Φ không tăng đều suốt quỹ đạo thành công, reward sai → model không thể học.
2. **Vẽ tách thành phần reward theo step**: format / PBRS / bonus / penalty riêng. Bạn sẽ thấy ngay PBRS có đang ~0 không. (`metrics_logger` đã ghi `geometric_rewards`, `format_rewards` — chỉ cần plot tách ra.)
3. **Log phân bố `n_stalled`**: nếu cao, model đang copy pose cũ → cần tăng `stall_penalty` hoặc xử lý ở prompt.
4. **Kiểm tra biên độ khớp model xuất ra** so với `current_angles`: nếu deltas luôn nhỏ, tay gần như đứng yên → vấn đề ở prompt/khởi tạo, không phải reward.

## Đề xuất ưu tiên (theo thứ tự tác động)

1. **Tách format khỏi reward giá trị.** Đừng cộng thẳng format +1.0 vào total. Dùng format như **gate cứng** (sai → reward âm cố định, ví dụ −1, và không branch) còn completion hợp lệ thì **chỉ chấm bằng geometric**. Như vậy advantage trong nhóm phản ánh đúng chất lượng motion thay vì cú pháp.
2. **Tắt hẳn KL lúc đầu** (`beta=0.0`) để cho thăm dò, bật lại nhỏ (0.01–0.05) sau khi đã có grasp.
3. **Tăng độ mạnh tín hiệu approach** hoặc giảm `near_thresh`/điều kiện `is_grasped` cho dễ chạm mốc grasp đầu tiên — cần ít nhất vài % grasp để bonus có cơ hội định hình policy. Hiện `near_thresh=0.05`, `lift_thresh=0.03`.
4. **Tăng quy mô** sau khi xác nhận reward đúng: vài nghìn → vài chục nghìn step, và theo dõi `success_rate` 100-step gần nhất là chỉ số vàng.

Tóm lại: kiến trúc (privileged geometric reward + vision-only policy, single-step GRPO) là hợp lý về nguyên tắc, nhưng **cách trộn format vào reward + KL beta=0.5 + chưa từng chạm mốc grasp** khiến tín hiệu học gần như chỉ còn "viết JSON đúng". Hãy sửa phần reward shaping và xác minh Φ đơn điệu trên quỹ đạo thành công trước, rồi mới chạy dài.

Bạn muốn tôi sửa luôn phần tách format/geometric trong `episode_manager.py` và viết script kiểm tra Φ trên quỹ đạo scripted không?

---
