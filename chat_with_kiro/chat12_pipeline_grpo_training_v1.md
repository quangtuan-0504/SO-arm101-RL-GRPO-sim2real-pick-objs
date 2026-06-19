# Tôi muốn huấn luyện VLM học cách điều khiển cán...

Exported: 6/19/2026, 12:29:08 PM

---

**User:**

Tôi muốn huấn luyện VLM học cách điều khiển cánh tay robot bằng GRPO. Quy trình huấn luyện như sau:
- Đầu tiên là Dataset, không giống như các dataset thông thường đã tồn tại sẵn các data point. Các data point trong dự án này chưa hề tồn tại sẵn mà sẽ được lấy từ việc model tương tác với môi trường trong quá trình huấn luyện model. 
Dataset của tôi sẽ bao gồm các map game. Mỗi map sẽ có sự khác biệt về vị trí, số lượng các object. Tôi muốn có khoảng 5 map game. Mỗi map sẽ có ít nhất 1 object mỗi loại và có tối đa 5 object. Vị trí các object là khác nhau giữa mỗi map. (Tôi đang thắc mắc vị trí camera nên để cố định trong mọi map, hay tuỳ chỉnh theo từng map để cho dù đặt cam ở đâu model cũng dự đoán góc quay được. Với giai đoạn đầu, nên cố định camera trong toàn bộ map để bài toán đơn giản và reward ổn định hơn)
- Mỗi map sẽ có 3 màn game, tổng có 15 màn game, mỗi màn game yêu cầu gắp một loại đối tượng (không biết có nên đa dạng hoá yêu cầu mỗi màn game không nữa, ví dụ yêu cầu gắp obj loại A và B, thôi cứ làm đơn giản trước). Mỗi màn game chỉ được phép chạy tối đa 20 step.
- Mỗi step chạy sẽ ứng với 1 data point được tạo ra.
- Sẽ có một bảng lưu trữ id của từng màn game, cấu hình và yêu cầu đối với màn game đó
- Server  Linux có GPU để chạy huấn luyện được tôi thuê trên Vast AI. Server này được dùng để chạy quá trình huấn luyện model: Dataloader lấy dữ liệu, quá trình forward và cập nhật bộ tham số, gọi service tính toán điểm thưởng, gọi game để tương tác với môi trường.
- Game chạy trên mac mini của tôi chỉ đóng vai trò là nhận lệnh chứa các góc quay điều khiển, sau đó thực hiện chụp ảnh sau khi thực hiện lệnh điều khiển và gửi lại cho server huấn luyện.
- Đầu tiên Dataloader sẽ truy cập vào mỗi màn game, gửi request yêu cầu khởi động màn game này, game chụp hình góc nhìn của 2 camera hiện tại gửi lại cho dataloader( tôi đang phân vân không biết nên build prompt trong game rồi gửi cho service huấn luyện hay là gửi ảnh thôi và service huấn luyện nhận ảnh rồi tự build prompt) , dataloader xây input prompt cho vào VLM. VLM sinh ra 4 lệnh điều khiển gửi 4 lệnh này cho game. Từ trạng thái i hiện tại, game chạy thử cả 4 lệnh cho ra 4 trạng thái i+1 mới, chụp hình 4 trạng thái mới này, gửi lại cho service huấn luyện, service huấn luyện gửi trạng thái i và 4 trạng thái i+1 tới service reward, kêu chấm điểm 4 trạng thái theo các tiêu chí đề ra, service chấm điểm chấm xong gửi điểm lại cho service huấn luyện, service huấn luyện dùng điểm số này để cập nhật model. cập nhật xong đến step tiếp theo. chọn 1 trong 4 trạng thái i+1 vừa rồi (có thể chọn nhờ vào điểm số). Yêu cầu game giữ trạng thái được chọn để chơi tiếp, Dataloader lấy trạng thái được chọn build input prompt và gửi tới VLM. VLM lại sinh ra 4 lệnh điều khiển và tiếp tục vòng lặp.
- Có thể thiết kế để chơi 2 màn chơi song song để huấn luyện cho nhanh. 
- Sử dụng thư viện trl để chạy GRPO. 
- Về hàm chấm điểm thiết kế 2 hàm chấm điểm. Hàm thứ 1 chấm format đầu ra, cho điểm 0 or 1, 1 nếu đầu ra đúng format yêu cầu, 0 nếu sai format yêu cầu. Đầu ra nào bị 0 điểm ở hàm chấm 1 sẽ không cần hàm 2 chấm nữa và sẽ nhận về 0 điểm luôn (đương nhiên là đầu ra sai format thì cũng không gửi đầu ra đó cho game để lấy trạng thái mới). Hàm chấm điểm thứ 2 dựa trên các hình ảnh đầu ra, sử dụng cùng 1 VLM để self host với vLLM để thực hiện đọc ảnh và chấm điểm. Thiết kế prompt chấm điểm, prompt chấm điểm gồm ảnh trạng thái i và ảnh của các trạng thái i+1 hợp lệ, cùng với mô tả mục tiêu và hướng dẫn chấm điểm, format đầu ra. Thang điểm từ 0-100.

---

**Assistant:**

Đây là một thiết kế khá lớn. Trước khi đưa ra góp ý và kế hoạch, tôi cần đọc code hiện có để nắm rõ những gì đã tồn tại (game, camera, VLM client, prompt builder).

Giờ tôi cần hiểu API mô phỏng (reset, set trạng thái, ngẫu nhiên hoá vật thể) để biết cái gì đã có và cái gì cần xây thêm cho dataset map game.

Tôi đã đọc xong toàn bộ pipeline hiện có (sim, scene_builder, prompt_builder, qwen_client, control_loop, worker). Trước khi góp ý kiến trúc GRPO, tôi muốn kiểm tra trạng thái hiện tại của `trl` về hỗ trợ GRPO cho VLM và môi trường nhiều bước, vì đây là yếu tố quyết định và phụ thuộc phiên bản.

Tôi đã nắm đủ code hiện có và vừa kiểm tra trạng thái `trl` (hiện ~v0.27 hỗ trợ GRPO cho VLM + vLLM). Dưới đây là góp ý kiến trúc, trả lời các câu hỏi mở của bạn, và một điểm kỹ thuật quan trọng cần xử lý trước khi viết code.

## Trả lời nhanh các câu hỏi mở của bạn

1. **Camera cố định hay thay đổi theo map?** → Giai đoạn đầu **cố định** như bạn nghĩ. Lý do đúng: reward ổn định, loại bỏ một biến nhiễu lớn, và quan trọng nhất là `stand_cam` của bạn đang đặt ở toạ độ thế giới cố định (`scene_builder._add_stand_camera`), còn `wrist_cam` gắn cứng vào gripper — nên giữ nguyên là tự nhiên. Việc tổng quát hoá góc camera để lại cho giai đoạn 2 (domain randomization).

2. **Đa dạng hoá yêu cầu mỗi màn (gắp A và B)?** → Làm **đơn giản trước**: mỗi màn 1 loại. Lý do reward: nhiệm vụ đa-object cần credit assignment phức tạp hơn nhiều, và GRPO vốn đã khó với reward thưa.

3. **Build prompt trong game hay trên server huấn luyện?** → **Build trên server huấn luyện, game chỉ gửi ảnh + metadata**. Đây là lựa chọn rõ ràng vì:
   - Prompt gắn chặt với model/tokenizer/reward design — tất cả nằm trên server.
   - `prompt_builder.py` đã ở repo huấn luyện rồi.
   - Khi bạn sửa prompt (sẽ sửa rất nhiều lần), không phải deploy lại Mac mini.
   - Game nên "dumb" và ổn định: nhận góc → step → chụp ảnh → trả ảnh + metadata (góc khớp hiện tại, level instruction). Server tự dựng prompt.

## Vấn đề kỹ thuật quan trọng nhất: GRPO vốn là single-turn

Đây là điểm tôi cần bạn lưu ý trước khi code. Tài liệu hiện tại xác nhận rõ: *"GRPO and its variants are inherently single-turn: they optimize from terminal rewards on isolated prompt-response pairs"* ([Feedback-Aware GRPO, 2026](https://arxiv.org/abs/2511.07833) — nội dung đã diễn giải lại). 

Vòng lặp của bạn lại là **multi-step** (20 step/màn, mỗi step chọn 1 trong 4 trạng thái để đi tiếp). Có hai cách ánh xạ:

**Cách A — Single-step GRPO (khuyến nghị cho giai đoạn đầu):** Coi mỗi step là một mẫu GRPO độc lập. Reward của mỗi completion = điểm của trạng thái i+1 ngay sau đó (do judge VLM chấm). Đây chính xác là thứ bạn mô tả. Ưu điểm: chạy được ngay với `trl`, reward dày (dense). Nhược điểm: myopic — không học được credit dài hạn (một bước "hi sinh" giờ để gắp tốt hơn sau sẽ không được thưởng). Với bài pick-and-place có judge chấm tiến độ từng bước thì chấp nhận được.

**Cách B — Multi-turn/trajectory GRPO:** Tính advantage trên cả quỹ đạo. Mạnh hơn nhưng `trl` chưa hỗ trợ ổn định, phải tự custom loss. Để dành giai đoạn sau.

Tôi đề xuất **Cách A**, và nó ánh xạ rất đẹp vào `trl`:

```
Batch GRPO = step hiện tại của N màn chạy song song   (N = số màn song song)
  └── mỗi prompt → group G=4 completions               (4 lệnh VLM sinh ra)
        └── reward_func nhận cả 4 cùng lúc → branch 4 nhánh trong game → chấm điểm
```

Điều mấu chốt: hàm reward của `trl` nhận **toàn bộ G completions của một prompt cùng lúc**, đúng thứ bạn cần để branch 4 nhánh từ cùng trạng thái i. Việc "chơi song song 2 màn" của bạn chính là **batch size** của GRPO — rất tự nhiên.

## Một thứ còn THIẾU trong code: snapshot/restore trạng thái để branch

Để từ trạng thái i chạy thử **cả 4 lệnh** ra 4 trạng thái i+1 khác nhau, game phải **lưu và khôi phục** trạng thái MuJoCo. Hiện `Simulation` chỉ có `reset()` / `reset_arm()`, chưa có save/restore tuỳ ý. Đây là phần cốt lõi cần thêm:

```python
# sim/simulation.py — cần bổ sung
def get_state(self) -> dict:
    return {
        "qpos": self.data.qpos.copy(),
        "qvel": self.data.qvel.copy(),
        "ctrl": self.data.ctrl.copy(),
        "time": self.data.time,
    }

def set_state(self, s: dict):
    self.data.qpos[:] = s["qpos"]
    self.data.qvel[:] = s["qvel"]
    self.data.ctrl[:] = s["ctrl"]
    self.data.time = s["time"]
    mujoco.mj_forward(self.model, self.data)
```

Game lưu snapshot theo `state_token`; khi branch, nó `set_state(token)` rồi áp lệnh j cho từng nhánh.

## Kiến trúc 3 service đề xuất

```
┌─ Mac mini ──────────────┐   ┌─ Vast AI GPU ───────────────────────────────┐
│ GAME SERVICE (FastAPI)  │   │ TRAINING SERVICE (trl GRPOTrainer)           │
│  POST /reset {level_id} │◄──┤  - IterableDataset: prompt = step h/màn       │
│   → token, 2 ảnh, meta  │   │  - GRPOTrainer + vLLM sinh 4 completions      │
│  POST /step             │   │  - reward_funcs = [fmt_reward, judge_reward]  │
│   {token, angles}       │   │                                               │
│   → token', 2 ảnh, meta │   │  REWARD/JUDGE (vLLM self-host cùng GPU)        │
│  get_state/set_state    │   │   - judge VLM chấm ảnh i vs i+1 (0..100)      │
│  để branch 4 nhánh      │   └───────────────────────────────────────────────┘
└─────────────────────────┘
```

- **Game (Mac):** stateless về logic, stateful về snapshot. Trả ảnh PNG (đã có `render_utils`) + metadata (góc khớp hiện tại, instruction). Không build prompt.
- **Training (Vast):** dựng prompt, GRPO, orchestrate. Sau mỗi step, chọn winner (theo điểm) → gọi game giữ trạng thái đó → sinh prompt step kế.
- **Judge (vLLM):** chạy chung GPU server, self-host cùng base model để chấm.

## Bảng lưu màn game (level table)

Đề xuất schema (SQLite/JSON), khớp `config.yaml` hiện tại:

```
levels(
  level_id, map_id, target_type, target_object_name,
  instruction_text, camera_config_ref, max_steps=20,
  objects_json   -- danh sách object (name/shape/size/pos/rgba/mass)
)
```
5 map × 3 màn = 15 level. Mỗi map sinh `objects` ngẫu nhiên (1–5 object, ≥1 mỗi loại, vị trí khác nhau) — có thể viết một generator dùng lại `scene_builder`.

## Lưu ý về reward (từ thực nghiệm gần đây)

- **Hàm 1 (format, 0/1) làm gate** đúng như bạn nói: 0 điểm → bỏ qua, không gửi game, không gọi judge. Tốt cho tiết kiệm chi phí.
- **Cảnh báo "reward collapse":** nếu judge chấm 4 nhánh gần bằng nhau → advantage chuẩn hoá ≈ 0 → không học ([RC-GRPO, 2026](https://arxiv.org/html/2602.03025v1), diễn giải lại). Khắc phục: yêu cầu judge **so sánh tương đối và chấm phân biệt rõ** giữa các nhánh, hoặc thêm thành phần reward dày (khoảng cách kẹp–vật giảm, độ cao khi nâng…) lấy trực tiếp từ MuJoCo (rẻ, khách quan, không cần VLM).
- Thang 0–100 của judge ổn vì GRPO chuẩn hoá trong group, scale tuyệt đối không quan trọng — **tính phân biệt trong nhóm** mới quan trọng.

## Kế hoạch theo giai đoạn

1. Thêm `get_state/set_state` vào `Simulation` + map/level generator + level table.
2. Game service (FastAPI) trên Mac: `/reset`, `/step` (có branch), trả ảnh + meta.
3. Reward service: `format_reward` (cục bộ) + `judge_reward` (vLLM) + prompt chấm điểm.
4. Training service: IterableDataset dynamic + GRPOTrainer (single-step) + vòng outer advance môi trường.
5. Chạy 1 màn → 2 màn song song → scale.

---

Đây là phân tích và đề xuất, tôi **chưa sửa code gì cả**. Bạn muốn bắt đầu từ đâu? Tôi nghĩ nên khởi động từ **bước 1 (snapshot/restore + level generator)** vì nó là nền tảng cho mọi thứ và kiểm thử được ngay trên Mac mà chưa cần GPU. Bạn xác nhận hướng này, hay muốn điều chỉnh chỗ nào trong kiến trúc trước?

---

**User:**

Tôi có nên thêm một hàm chấm điểm dựa vào vị trí vật thể cần gắp or giỏ so với kẹp gắp ở từng giai đoạn không? ví dụ giai đoạn approach thì đầu gắp càng gần object mục tiêu điểm càng cao, carry thì càng gần giỏ điểm càng cao 

---

**Assistant:**

Có, **nên thêm** — và thực ra đây nên là reward *chính* để pipeline hội tụ được, còn judge VLM đóng vai trò bổ sung. Lý do: điểm hình học lấy thẳng từ MuJoCo là **chính xác, miễn phí, tức thời và không nhiễu**, trong khi judge VLM thì chậm, tốn GPU và dao động. Nhưng có vài điểm thiết kế quan trọng để nó không phản tác dụng.

## 1. Đừng key reward theo "phase do model tự khai"

Cách bạn phát biểu — "giai đoạn approach thì..., carry thì..." — có một bẫy: nếu reward phụ thuộc vào `phase` mà chính model xuất ra, model sẽ **gian lận** bằng cách luôn khai phase dễ ăn điểm. 

Thay vào đó, hãy suy ra giai đoạn từ **điều kiện trạng thái khách quan** trong sim:
- `grasped` = vật đang được kẹp (tiếp xúc 2 má kẹp + vật nâng khỏi sàn) — đo bằng contact/độ cao z, không phải bằng lời model nói.
- Khi `grasped == False` → đang ở pha approach → thưởng theo khoảng cách kẹp↔vật.
- Khi `grasped == True` → đang ở pha carry → thưởng theo khoảng cách vật↔giỏ.

Gating bằng `grasped` thực tế cũng tạo ra đúng hành vi "approach rồi carry" mà bạn muốn, nhưng model không thể bịa.

## 2. Dùng potential-based shaping (PBRS) để không bóp méo chính sách tối ưu

Đây là cách "đúng chuẩn lý thuyết" để thêm reward dày mà không làm lệch nghiệm tối ưu. Định nghĩa một hàm thế năng Φ cho trạng thái, rồi reward mỗi bước = `Φ(s_{i+1}) − Φ(s_i)`:

```python
def potential(state):
    d_obj    = dist(gripper_site, target_obj)      # kẹp ↔ vật
    d_basket = dist(target_obj, basket_center)     # vật ↔ giỏ
    grasped  = is_grasped(state)                   # đo từ contact + z
    in_basket = obj_in_basket(state)               # vật nằm trong lòng giỏ

    phi  = w1 * (-d_obj)                    # approach: gần vật hơn = tốt
    phi += w2 * grasped                     # mốc: kẹp được vật
    phi += w3 * grasped * (-d_basket)       # carry: gần giỏ hơn (chỉ tính khi đã kẹp)
    phi += w4 * in_basket                   # mốc cuối: thả vào giỏ
    return phi

shaped_reward = potential(s_next) - potential(s_cur)
```

`grasped * (-d_basket)` chính là cơ chế tự động chuyển từ "thưởng gần vật" sang "thưởng gần giỏ" đúng như ý bạn — nhưng được kích hoạt bởi sự kiện kẹp được vật, không bởi nhãn pha tự khai.

## 3. Chống reward hacking

Khoảng cách *đơn thuần* dễ bị lách (rê kẹp lại sát vật mà không kẹp, hoặc đập vật văng tới gần giỏ). Phải kèm:
- `grasped` và `in_basket` là **mốc rời rạc** (bonus lớn) chứ không chỉ khoảng cách liên tục.
- Phạt nhẹ nếu làm vật văng khỏi vùng thao tác, hoặc nếu vận tốc vật quá lớn (đập).
- Yêu cầu `grasped` mới được tính `-d_basket`, nếu không vật bị hất tới gần giỏ vẫn không ăn điểm carry.

## 4. Về "privileged info" — yên tâm

Reward dùng toạ độ thật 3D, còn policy (VLM) chỉ thấy ảnh. Đây là setup **hợp lệ và phổ biến** (privileged reward, vision-only policy): thông tin đặc quyền chỉ dùng lúc *huấn luyện* để chấm điểm, lúc *suy luận/triển khai* model vẫn chỉ nhìn ảnh. Không mâu thuẫn với mục tiêu "VLM học điều khiển bằng thị giác".

## 5. Phối hợp 3 nguồn reward

Đề xuất tổ hợp (trl cộng các reward func):

| Reward | Vai trò | Khi nào tính |
|--------|---------|--------------|
| `format_reward` (0/1) | gate | luôn; =0 thì bỏ qua 2 cái dưới |
| `geometric_reward` (PBRS) | **tín hiệu chính**, hội tụ | mọi bước qua được gate |
| `judge_reward` (0–100, chuẩn hoá) | bổ sung thứ khó đo (căn hướng kẹp, chất lượng tư thế) | tuỳ chọn, có thể bật sau |

**Lộ trình thực tế:** giai đoạn đầu chạy với `format + geometric` trước để xác nhận toàn bộ pipeline GRPO **thực sự học được** (rẻ, ổn định, debug nhanh). Khi đã hội tụ, mới layer thêm judge VLM. Nếu bật cả hai từ đầu, lúc không hội tụ bạn sẽ rất khó biết lỗi do reward hình học, do judge, hay do GRPO.

Một hệ quả tốt: các đại lượng `grasped`, `in_basket`, `d_obj`, `d_basket` cũng chính là **điều kiện kết thúc màn** và **metric đánh giá** — viết một lần dùng cho cả reward lẫn success detection.

Muốn tôi triển khai luôn các hàm đo này (`dist`, `is_grasped`, `obj_in_basket`) dựa trên site `gripperframe` và geom vật/giỏ trong model hiện tại không? Đây là phần kiểm thử được ngay trên Mac, không cần GPU.

---
