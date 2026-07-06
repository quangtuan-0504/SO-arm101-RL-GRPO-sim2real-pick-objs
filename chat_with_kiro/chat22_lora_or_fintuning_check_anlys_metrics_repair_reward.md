# quá trình huấn luyện nên dùng Lora hay finetuni...

Exported: 7/6/2026, 11:09:18 AM

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

**User:**

Tách format khỏi reward giá trị. Đừng cộng thẳng format +1.0 vào total. Dùng format như gate cứng (sai → reward âm cố định, ví dụ −1, và không branch) còn completion hợp lệ thì chỉ chấm bằng geometric. 
viết script kiểm tra Φ trên quỹ đạo scripted

---

**Assistant:**

Bắt đầu với việc sửa `episode_manager.py`. Trước tiên kiểm tra chỗ nào tham chiếu `format_weight` để không phá test.

Có hai đường: `episode_manager.py` (dùng bởi grpo_train) và `rollout.py` (vòng lặp thay thế). Tôi xem test để giữ tương thích.

Test chỉ kiểm `geometric_rewards` và `n_valid`, không phụ thuộc thành phần format trong total — nên đổi an toàn. Tôi sửa `episode_manager.py`.

Giờ kiểm tra dòng tính `total` và comment quanh nó.

Cập nhật comment mô tả field để phản ánh ngữ nghĩa mới.

Cập nhật `rollout.py` cho nhất quán vì nó là bản sao cùng logic reward.

Giờ viết script kiểm tra Φ trên quỹ đạo scripted. Tôi cần hiểu API của `Simulation`.

Tôi đã hiểu API. Giờ viết script kiểm tra Φ. Script sẽ dựng một quỹ đạo lý tưởng (tiến tới vật → canh → hạ → kẹp → nhấc → mang sang giỏ → thả) bằng chuỗi `StateMeasure`, in `potential()` và `geometric_reward` từng bước, và **tự đánh dấu các bước Φ giảm** — vì đó chính là điểm chống lại việc học PBRS một-step.

Chạy thử script để xác nhận nó hoạt động và xem Φ có đơn điệu không.

Script chạy tốt và đã lộ ra một khiếm khuyết thật: Φ **giảm ngay tại bước canh thẳng trên vật** (bước 5 và 7) — đúng chỗ cánh tay cần canh trước khi hạ xuống. Kiểm tra import các file đã sửa cho chắc.

Cả hai thay đổi xong và verify sạch. Tóm tắt:

## 1. Format thành gate cứng (bỏ cộng điểm dương)

Trong `train/episode_manager.py` (và đồng bộ luôn `train/rollout.py`):

- Đổi tham số `format_weight` → `invalid_format_penalty` (mặc định `1.0`).
- Completion **hợp lệ**: `fmt = 0.0` → `total = geometric` thuần (không còn +1.0 lấn át tín hiệu).
- Completion **sai format**: `fmt = -invalid_format_penalty`, và **không branch** vào game (logic branch cũ vốn đã chỉ branch `valid_idx`, nên giữ nguyên) → `total = -1.0`.

Kết quả: advantage trong nhóm GRPO giờ phản ánh **chất lượng chuyển động**, còn cú pháp sai bị phạt âm cố định thay vì chỉ "mất phần thưởng". Trường `format_rewards` vẫn được log (nay mang giá trị 0/âm) nên `metrics_logger`/plot không vỡ. Test hiện có chỉ kiểm `geometric_rewards` và `n_valid` nên không bị ảnh hưởng.

## 2. Script kiểm tra Φ: `train/check_potential.py`

Dựng quỹ đạo gắp thành công lý tưởng (tiến → canh → hạ → kẹp → nhấc → mang → thả), in `Φ`, `ΔΦ`, và `geometric_reward` từng bước, tự đánh dấu các bước Φ giảm. Chạy:

```
.venv/bin/python -m train.check_potential --steps 6
```

**Phát hiện quan trọng ngay từ lần chạy đầu:** trên chính quỹ đạo thành công, Φ **giảm ở 2 bước ngay quanh điểm canh thẳng trên vật** (bước 5: ΔΦ=−0.008; bước 7: ΔΦ=−0.005). Nguyên nhân: trong `potential()`, khi đầu kẹp đã canh thẳng (`na→0`) nhưng còn ở độ cao chờ, đích kéo chuyển từ "điểm chờ" sang "chính vật", làm `base` tăng vọt lên ≈ `h_standoff` → tạo một "thung lũng" Φ đúng tại chỗ cần canh trước khi hạ.

Với GRPO single-step PBRS, ΔΦ<0 nghĩa là reward per-step **âm** cho đúng động tác canh-thẳng. Policy tham lam một-bước bị khuyến khích **né** việc canh trên vật → rất khớp với triệu chứng grasp-rate = 0% bạn thấy trong log.

Đây là bằng chứng cụ thể cho thấy vấn đề nằm ở **hình dạng của Φ**, không chỉ ở cách trộn format. Bước tiếp theo nên là làm mượt đoạn chuyển "phao chờ → vật" trong `potential()` (blend liên tục theo độ cao) hoặc giảm trọng số `danger`, rồi chạy lại `check_potential` cho tới khi báo `[PASS]`. Bạn muốn tôi sửa `potential()` để khử thung lũng này không?

---

**User:**

hãy nghiên cứu và đề xuất cách giải quyết vấn đề này

---

**Assistant:**

Vấn đề đã rõ về mặt toán học. Giờ tôi nghiên cứu kỹ nguyên nhân gốc trong `potential()` và thiết kế lại. Trước tiên kiểm tra xem `r_funnel`/`r_danger`/`danger` còn được tham chiếu ở đâu ngoài `metrics.py`.

Các key trọng số chỉ được dùng trong `metrics.py` (và `check_potential` đọc `h_standoff`); `game/service.py` chỉ serialize `d_xy/dz`. Nên có thể thiết kế lại an toàn.

## Nghiên cứu nguyên nhân gốc

Công thức `potential()` hiện tại ở pha approach **blend hai hàm khoảng cách khác gốc quy chiếu**:

```
base = na·d(đầu kẹp → ĐIỂM CHỜ trên vật) + al·d(đầu kẹp → VẬT)
```

Tại điểm canh thẳng ở độ cao chờ `(d_xy=0, dz=h)`:
- `d(→ điểm chờ) = 0`
- `d(→ vật) = h = 0.10`

Hai đích này **không bằng nhau ở biên blend**. Khi `na→0` (canh thẳng dần), `base` bị kéo từ ~0 lên `h`, tạo một **thung lũng Φ ngay tại chỗ cần canh**. Đó là lỗi cấu trúc, không phải chuyện tinh chỉnh trọng số — mọi cặp `(r_funnel, danger)` đều để lại thung lũng vì gốc quy chiếu đổi đột ngột.

## Đề xuất: thay bằng "hành lang chữ L" — một hàm Φ liền, đơn điệu

Thay vì blend hai đích, dùng **một** hàm khoảng-cách-tiến-trình duy nhất mã hoá đúng đường đi mong muốn (nâng lên → canh ngang → hạ thẳng):

```
belowness = clip((h − dz)/h, 0, 1)          # 1: đầu kẹp ngang/dưới vật; 0: ≥ độ cao chờ
L = d_xy · (1 + k · belowness) + |dz|
Φ_pre = − approach · L
```

Ý tưởng: **lệch ngang bị phạt nặng hơn khi đầu kẹp còn thấp** (`1 + k·belowness`). Hệ quả:
- Khi thấp & lệch ngang: cách rẻ nhất để giảm `L` là **nâng lên trước** (giảm `belowness` làm rẻ đi số hạng ngang lớn) → tự động đi "đường trên", không cần term `danger` rời rạc.
- Khi đã canh thẳng (`d_xy≈0`): `L = |dz|` → **hạ xuống được thưởng**.
- Không còn đổi gốc quy chiếu → **không có thung lũng**.

Về mặt lý thuyết vẫn là hàm thuần trạng thái nên **giữ nguyên đảm bảo PBRS** (không bóp méo chính sách tối ưu). Thay `danger, r_funnel, r_danger` bằng một tham số duy nhất `low_penalty (k)`.

Tôi triển khai luôn rồi verify bằng chính `check_potential.py`.

`[PASS]` — Φ giờ đơn điệu suốt quỹ đạo. Chạy dày hơn để chắc không còn thung lũng ẩn.

Xác nhận thêm tính chất chống va chạm (ưu tiên "nâng lên trước" hơn "tiến ngang khi còn thấp") của Φ mới:

Chống va chạm vẫn đúng: nâng lên (ΔΦ=+0.264) được ưu tiên hơn tiến ngang khi còn thấp (ΔΦ=+0.186), mà không cần term `danger` rời rạc. Cập nhật lại gợi ý trong `check_potential.py` cho khớp thiết kế mới (không còn `danger`).

Thử chạy test reward hiện có để chắc thay đổi không phá gì (test tự dựng `GameEnv`, không cần server).

Xong. Tổng kết nghiên cứu và giải pháp:

## Nguyên nhân gốc

`potential()` cũ ở pha approach **blend hai hàm khoảng cách khác gốc quy chiếu** (`d→điểm chờ` và `d→vật`). Tại điểm canh thẳng ở độ cao chờ, hai đích lệch nhau đúng bằng `h_standoff`, nên khi `na→0` giá trị `base` bị kéo vọt lên → tạo **thung lũng Φ ngay tại chỗ cần canh**. Với PBRS một-step, thung lũng này = reward âm cho đúng động tác canh-thẳng → policy tham lam bị dạy *tránh* canh trên vật, khớp với grasp-rate 0%.

## Giải pháp đã triển khai: "hành lang chữ L"

Thay bằng **một** hàm khoảng-cách-tiến-trình liền, đơn điệu (trong `sim/metrics.py`):

```
belowness = clip((h − dz)/h, 0, 1)
L = d_xy·(1 + low_penalty·belowness) + |dz|
Φ_pre = −approach·L
```

- Lệch ngang bị phạt nặng hơn khi đầu kẹp còn thấp → tự động "nâng lên trước, tiến ngang sau" mà **không cần term `danger` rời rạc**.
- Khi đã canh thẳng, `|dz|` kéo hạ thẳng xuống.
- Không đổi gốc quy chiếu → **không còn thung lũng**; vẫn là hàm thuần trạng thái nên **giữ đảm bảo PBRS**.

Gọn hơn: thay 3 tham số `danger, r_funnel, r_danger` bằng một `low_penalty (k=3.0)`.

## Kiểm chứng

- `check_potential.py`: từ `[CANH BAO] 2 bước Φ giảm` → **`[PASS]` Φ đơn điệu** suốt quỹ đạo (kiểm cả với `--steps 30`).
- Chống va chạm còn nguyên: từ điểm thấp+lệch, "nâng lên" (ΔΦ=+0.264) được ưu tiên hơn "tiến ngang khi thấp" (ΔΦ=+0.186).
- `tests/test_env_rewards` vẫn PASS, geometric reward vẫn phân biệt tốt giữa các nhánh.

## Lưu ý tinh chỉnh khi chạy thật

- `low_penalty` càng lớn càng ép "đi đường trên" mạnh (thử 2–5). `h_standoff` nên ≈ chiều cao vật + vài cm.
- Đáy hành lang (`dz→0`, `d_xy→0`) phải nằm trong vùng `near_thresh=0.05`/`lift_thresh=0.03` mà `is_grasped` công nhận, nếu không tay tới sát vật vẫn không được tính grasped.
- Có thể override các trọng số này qua `phi_weights` truyền xuống `geometric_reward` mà không sửa code.

Kết hợp với thay đổi format-gate trước đó, giờ tín hiệu học phản ánh đúng chất lượng chuyển động **và** không có anti-incentive quanh điểm canh. Bước tiếp theo nên chạy lại một đợt train ngắn để xem grasp-rate có nhích khỏi 0 không.

---
