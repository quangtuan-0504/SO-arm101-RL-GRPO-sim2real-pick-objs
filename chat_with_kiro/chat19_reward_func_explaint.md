# giải thích lại giúp tôi cách hoạt động của hàm ...

Exported: 6/26/2026, 4:29:08 PM

---

**User:**

giải thích lại giúp tôi cách hoạt động của hàm chấm điểm

---

**Assistant:**

The actual scoring logic lives in `score_and_advance`. Let me read the episode manager.

Let me read the two reward components to explain them fully.

Let me check the `potential` function to complete the picture.

Đây là toàn bộ cách hàm chấm điểm hoạt động trong pipeline GRPO. Hàm tổng thể nằm ở `EpisodeManager.score_and_advance`, gồm **2 thành phần điểm cộng lại**: format gate + geometric reward.

## Tổng quan luồng

Với mỗi trạng thái (step) của một episode, VLM sinh ra G completions (G lệnh điều khiển khác nhau cho cùng một ảnh). Hàm chấm điểm sẽ:

```
total[i] = format_reward[i] + geometric_reward[i]
```

rồi chọn completion có điểm hình học cao nhất làm **winner**, commit nó để môi trường tiến sang step kế tiếp.

## 1. Format gate (0/1) — `rewards/format_reward.py`

Đây là "cửa lọc" rẻ tiền chạy trước. `check_format(text)` kiểm tra output của VLM có đúng cấu trúc JSON yêu cầu không:

- Có đủ 4 khoá bắt buộc: `phase`, `gripper_action`, `joint_angles_rad`, `done`
- `phase` thuộc tập hợp lệ (approach/align/grasp/lift/carry/release)
- `gripper_action` thuộc {open, close, hold}
- `done` là bool
- `joint_angles_rad` có đủ 6 khớp và mỗi góc là số

→ Đúng hết: **1.0**, sai bất kỳ điểm nào: **0.0**.

Ý nghĩa quan trọng: completion sai format **không được gửi vào game** để branch và **không được chấm hình học** (tiết kiệm chi phí mô phỏng). Trong `score_and_advance`, chỉ những index hợp lệ (`valid_idx`) mới được branch.

## 2. Geometric reward (tín hiệu chính) — `rewards/geometric_reward.py`

Đây là điểm dày, lấy **trực tiếp từ toạ độ thật trong MuJoCo** (chính xác, không nhiễu). Tính từ trạng thái `before` → `after` (sau khi thực thi lệnh). Gồm 4 phần:

**a) PBRS (Potential-Based Reward Shaping)** — phần chủ đạo:
```
reward = pbrs_scale * (Φ(after) − Φ(before))
```
Hàm thế năng Φ (trong `sim/metrics.py::potential`) đo "độ gần mục tiêu cuối":
- Khi **chưa kẹp**: `Φ = -d_grip_obj` → đầu kẹp càng gần vật, Φ càng cao.
- Khi **đã kẹp** (`grasped=True`): cộng bonus + `-d_obj_basket` → chuyển sang thưởng việc đưa vật gần giỏ.
- Khi **vào giỏ**: cộng thêm.

Điểm tinh tế: việc chuyển "thưởng gần vật" → "thưởng gần giỏ" được kích hoạt bởi `grasped` (đo khách quan từ vật lý), **không** bởi nhãn `phase` mà model tự khai → model không thể gian lận bằng cách khai bừa pha. Dùng PBRS đảm bảo không bóp méo chính sách tối ưu.

**b) Mốc rời rạc (bonus):**
- Vừa kẹp được vật (False→True): `+grasp_bonus`
- Đã kẹp rồi mà tuột mất (mà chưa vào giỏ): `−grasp_bonus` (phạt đối xứng để khỏi "kẹp rồi buông" ăn điểm)
- Vật vào giỏ (False→True): `+basket_bonus` (mốc cuối, thưởng lớn = 5.0)

**c) Chống reward-hacking:**
- Nếu vận tốc vật sau bước > ngưỡng `fling_speed`: `−fling_penalty` (phạt hành vi đập/hất vật cho văng gần giỏ).

## 3. Chọn winner & advance — trong `score_and_advance`

```python
winner_idx = max(valid_idx, key=lambda i: geo[i])   # chọn theo điểm HÌNH HỌC
next_obs = ep.env.commit(branch_of[winner_idx].branch_id)
```

- Winner được chọn theo **điểm geometric cao nhất** (không phải total), rồi commit để môi trường tiến thật sự sang trạng thái mới.
- `success = next_obs.done` (vật đã vào giỏ).
- `finished = True` khi xong (done) hoặc hết số step tối đa.
- Trường hợp **cả G completion đều sai format**: không branch được, không advance được → coi như hỏng, `finished=True` và reset episode sang level mới.

## Tóm tắt vai trò trong GRPO

GRPO cần so sánh tương đối G completion trong cùng nhóm để tính advantage. Vì vậy:
- `format_reward` lọc/trừng phạt output sai cú pháp.
- `geometric_reward` tạo ra phổ điểm liên tục giúp phân biệt completion nào điều khiển cánh tay tốt hơn, để model học theo những lệnh đưa hệ tiến gần mục tiêu (vật → kẹp → giỏ).

Bạn có muốn tôi giải thích sâu hơn phần nào không — ví dụ cách PBRS đảm bảo không làm lệch policy tối ưu, hay cách `is_grasped`/`obj_in_basket` phát hiện trạng thái?

---

**User:**

hàm điểm thưởng này có giải quyết được case: ở trạng thái i kẹp đã gắp được vật nhưng ở trạng thái này điểm lại cao cho nên nó không chịu nâng vật lên để bỏ vô giỏ mà qua trạng thái i+1 nó vẫn cố kẹp vật và giữ nguyên vị trí i

---

**Assistant:**

Câu hỏi rất hay, và đây chính là vấn đề mà thiết kế PBRS này được tạo ra để giải quyết. Câu trả lời ngắn gọn: **Có, hàm reward này không khuyến khích "đứng im ôm vật"** — và lý do nằm ở chỗ nó dùng *hiệu thế năng* chứ không phải *thế năng tuyệt đối*.

## Tại sao trạng thái i "điểm cao" không gây ra việc đứng im

Hiểu lầm thường gặp là nghĩ reward = giá trị của trạng thái. Nếu thiết kế kiểu đó (ví dụ mỗi step thưởng `-distance`), thì đúng là agent ở trạng thái tốt sẽ muốn ở lì để "hưởng điểm cao mãi". Nhưng ở đây không phải vậy.

Nhìn lại công thức trong `geometric_reward`:
```python
reward = pbrs_scale * (potential(after) - potential(before))
```

Reward là **chênh lệch** Φ(sau) − Φ(trước), không phải Φ tuyệt đối. Hệ quả:

- **Đứng im ở trạng thái i (giữ vật, không di chuyển):** Φ(after) ≈ Φ(before) → PBRS ≈ **0**. Trạng thái "điểm cao" không sinh ra điểm lặp lại nào cả. Agent **không** được thưởng vì *đang ở* trạng thái tốt, chỉ được thưởng vì *làm trạng thái tốt lên*.
- **Nâng vật/đưa về phía giỏ ở step i+1:** vì khi `grasped=True`, hàm Φ có thành phần `carry * (-d_obj_basket)`, nên giảm khoảng cách vật↔giỏ làm Φ **tăng** → PBRS **dương**.

Trong GRPO, ở step i+1 các completion cạnh tranh nhau:
- completion "ôm vật đứng yên" → geo ≈ 0
- completion "tiến về giỏ" → geo > 0

`winner_idx = max(valid_idx, key=lambda i: geo[i])` sẽ chọn completion tiến về giỏ. Vậy advantage tương đối luôn ưu ái việc *tiến triển*, không ưu ái việc *đứng im*.

## grasp_bonus không phải nguồn điểm để "ôm mãi"

Điểm cao tại trạng thái i mà bạn lo, một phần đến từ `grasp_bonus`. Nhưng nó là **mốc rời rạc theo chuyển dịch trạng thái**, chỉ cộng đúng *một lần* tại thời điểm `grasped` chuyển False→True:
```python
if after.grasped and not before.grasped:
    reward += w.grasp_bonus
```
Sang step i+1, `grasped` đã là True từ trước → **không cộng nữa**. Nên không có cách nào "kẹp được rồi ngồi hưởng grasp_bonus lặp lại".

Và việc "kẹp rồi buông để re-grasp ăn bonus lần nữa" cũng bị chặn bằng phạt đối xứng:
```python
if before.grasped and not after.grasped and not after.in_basket:
    reward -= w.grasp_bonus
```
Một chu kỳ kẹp–buông–kẹp lại tổng bằng 0, không lời.

## Những điều cần lưu ý (case mà thiết kế này KHÔNG tự giải)

Để trung thực, vấn đề bạn nêu được xử lý, nhưng có vài điểm liên quan đáng cảnh báo:

1. **Bế tắc do "không completion nào cải thiện được Φ" (local optimum).** Vì đây là GRPO single-step greedy theo Φ, nếu một bước cần *tạm rời xa giỏ* để né vật cản rồi mới tới được, thì không completion nào cho Φ dương trong bước đó → có thể kẹt. Đây là điểm yếu của shaping greedy, khác với việc "ôm vật ăn điểm".

2. **Plateau không có tín hiệu học.** Nếu cả G completion tại một step cho Φ thay đổi gần như bằng nhau (đều ≈ 0), thì advantage ≈ 0 → step đó không có gradient học, và env vẫn advance theo winner "tốt nhất trong đám tệ". Điều này lãng phí step chứ không *thưởng* cho việc đứng im, nhưng nó là lý do thực tế khiến agent có thể loanh quanh tại chỗ một thời gian trong giai đoạn đầu (do thiếu exploration, không phải do reward sai hướng).

3. **Không có phạt theo thời gian/step.** Hiện không có `time_penalty` (ví dụ −ε mỗi bước). Thêm một khoản phạt nhỏ mỗi step sẽ tạo áp lực *chủ động* hoàn thành sớm, củng cố thêm việc không lề mề. Hiện tại động lực hoàn thành đến từ `basket_bonus` ở cuối + PBRS dương khi tiến về giỏ, là đủ về mặt lý thuyết, nhưng một time penalty nhẹ thường giúp hội tụ nhanh hơn.

Nếu bạn muốn phòng xa case bế tắc/loanh quanh, tôi có thể thêm một `time_penalty` nhỏ vào `GeoWeights` và trừ mỗi bước. Bạn có muốn tôi làm không?

---

**User:**

trong trường hợp vật thể cần gắp ở ngay bên cạnh kẹp gắp, nhưng nếu dịch chuyển lại gần vật thì kẹp gắp sẽ hất văng vật, cánh tay cần phải né tránh và tiếp cận vật theo hướng khác để gắp, hàm điểm thưởng cover trường hợp này chưa

---

**Assistant:**

Câu này chạm đúng vào điểm yếu thật sự của thiết kế hiện tại. Trả lời thẳng: **Chưa cover tốt.** Hàm reward chỉ *trừng phạt hậu quả xấu* (hất văng vật) chứ **không hướng dẫn** cánh tay né và tiếp cận theo hướng đúng. Để giải thích vì sao, cần tách làm hai vấn đề.

## Vấn đề 1: Φ chỉ biết "khoảng cách", không biết "hướng tiếp cận"

Khi chưa kẹp được, thế năng chỉ là:
```python
phi = w["approach"] * (-m.d_grip_obj)
```
Φ thuần tuý là hàm của *khoảng cách đầu kẹp ↔ tâm vật*. Nó **không có khái niệm gì** về:
- góc tiếp cận (từ trên xuống vs từ ngang vào),
- hình học va chạm giữa ngàm kẹp và vật,
- vùng "an toàn" để áp sát.

Nên reward không thể biểu diễn được yêu cầu "phải vòng ra tiếp cận hướng khác". Với reward, mọi hướng làm giảm `d_grip_obj` đều "tốt như nhau" — kể cả hướng đâm thẳng làm văng vật.

## Vấn đề 2: GRPO single-step làm reward trở nên thiển cận (myopic)

Đây mới là gốc rễ. Về lý thuyết, PBRS có tính chất *telescoping*: tổng reward shaping trên cả episode = Φ(cuối) − Φ(đầu), **độc lập với đường đi**. Nghĩa là trong RL nhiều bước chuẩn, việc đi đường vòng (tạm thời tăng `d_grip_obj`) **không bị thiệt** về tổng return — PBRS không làm đổi policy tối ưu.

NHƯNG pipeline này là **GRPO single-step ("Cách A")**: mỗi step được chấm độc lập, không cộng dồn return qua các bước, không có value bootstrapping. Mỗi bước, G completion cạnh tranh nhau theo **reward tức thời của riêng bước đó**. Tức là agent *tham lam tối đa hoá ΔΦ ngay tại bước hiện tại*.

Hệ quả cho case của bạn — vật nằm sát ngàm, muốn gắp phải vòng ra:
- Completion "vòng ra" (tạm thời giữ nguyên hoặc tăng `d_grip_obj`) → ΔΦ ≤ 0 → reward ≈ 0 hoặc âm.
- Completion "áp thẳng vào" → trong khoảnh khắc làm `d_grip_obj` giảm, nhưng vật bị hất → ở trạng thái *after*, vật văng ra xa, `obj_speed` cao → Φ tụt + dính `fling_penalty` → reward âm.

So sánh tương đối: "vòng ra" (≈0) vẫn nhỉnh hơn "áp thẳng" (âm), nên GRPO có thể học *tránh* đâm thẳng. Nhưng nó **không bao giờ nhận tín hiệu dương** cho hành vi vòng-né đúng đắn. Bước đi vòng tốt nhất cũng chỉ được ~0. Agent phải **tự dò ra bằng exploration**, và reward dày trong lúc đó không hề dẫn đường — thậm chí còn hơi nghịch (đường vòng làm Φ giảm tạm thời).

Một lỗ hổng phụ: `fling_penalty` chỉ kích hoạt khi `obj_speed_after > fling_speed` (1.0 m/s). Một cú huých *nhẹ* làm vật trượt đi chậm (< 1 m/s) ra khỏi vị trí thì **không bị phạt** theo cơ chế fling — và `StateMeasure` cũng không có đại lượng nào đo "vật đã bị xê dịch khỏi chỗ ban đầu". Nên việc xáo trộn vật nhẹ nhàng gần như không bị trừng phạt.

## Tóm lại

| Khía cạnh | Hiện trạng |
|---|---|
| Phạt hất văng mạnh | Có (fling_penalty, ngưỡng 1 m/s) |
| Phạt huých nhẹ làm vật xê dịch | **Không** |
| Hướng dẫn hướng tiếp cận đúng | **Không** |
| Khuyến khích đường vòng cần thiết | **Không** (do single-step greedy) |
| Biểu diễn được hình học va chạm | **Không** |

Vật ở sát ngàm là một dạng *local optimum*: bước tiến tham lam nhất (áp thẳng) lại phá trạng thái, còn lối thoát đúng (vòng ra) thì reward không thưởng. Hiện chỉ trông cậy vào exploration ngẫu nhiên + sự cạnh tranh tương đối trong nhóm để né được, chứ không có lực kéo chủ động.

## Hướng khắc phục đề xuất

1. **Phạt xê dịch vật trước khi kẹp** (rẻ và hiệu quả nhất). Thêm đại lượng "khoảng cách vật so với vị trí nghỉ ban đầu" vào `StateMeasure`, rồi trong `geometric_reward` trừ điểm khi `not grasped` mà vật bị dời chỗ. Tín hiệu này dạy trực tiếp: "tiến lại gần mà KHÔNG làm xáo trộn vật = tốt", bất kể tốc độ.
2. **Shaping theo tư thế tiền-kẹp (pre-grasp pose)**: thưởng khi đầu kẹp vào được "nón tiếp cận" phía trên/đúng hướng của vật, thay vì chỉ thưởng khoảng cách trần. Việc này đưa khái niệm *hướng* vào Φ.
3. **Hạ ngưỡng `fling_speed`** hoặc đổi fling penalty thành tỉ lệ với `obj_speed` (phạt mượt) để cú huých nhẹ cũng bị bắt.
4. **Cân nhắc bỏ single-step greedy** sang chấm theo return nhiều bước (hoặc thêm value/GAE) để khôi phục tính telescoping của PBRS — khi đó đường vòng không còn bị thiệt.

Cách (1) ít rủi ro nhất và đánh trúng case bạn mô tả. Bạn muốn tôi triển khai phạt-xê-dịch-vật vào `sim/metrics.py` (`StateMeasure` + đo từ `object_rest_*`) và `rewards/geometric_reward.py` không? Tôi sẽ cần xem `EnvClient`/service có trả về vị trí nghỉ ban đầu của vật để client tính được đại lượng này hay không trước khi sửa.

---

**User:**

làm sao để biết grasped=True

---

**Assistant:**

`grasped` được xác định trong `is_grasped()` ở `sim/metrics.py`. Để giải thích chính xác, tôi cần xem các hàm sim mà nó gọi (`grasp_point`, `body_pos`, `object_rest_z`).

`grasped=True` được suy ra **gián tiếp từ hình học**, không dựa vào việc model khai báo hay vào lực kẹp/contact thật. Logic nằm ở `is_grasped()` trong `sim/metrics.py`:

```python
def is_grasped(sim, target_name, *, near_thresh=0.05, lift_thresh=0.03):
    obj   = sim.body_pos(target_name)     # toạ độ tâm vật (xyz)
    grasp = sim.grasp_point()             # toạ độ điểm gắp (site "gripperframe")
    near  = dist(obj, grasp) <= near_thresh          # (1) đủ gần
    rest_z = sim.object_rest_z(target_name)          # z lúc nghỉ ban đầu
    lifted = (obj[2] - rest_z) >= lift_thresh        # (2) đã được nhấc lên
    return bool(near and lifted)
```

## Hai điều kiện phải đồng thời đúng

**(1) `near` — đầu kẹp đủ sát vật**
- `sim.grasp_point()` trả toạ độ thế giới của site `gripperframe` (điểm giữa hai ngàm kẹp), lấy từ `data.site_xpos` trong MuJoCo.
- `sim.body_pos(target_name)` trả toạ độ tâm body của vật, từ `data.xpos`.
- `dist(...)` là khoảng cách Euclid 3D. Phải ≤ `near_thresh = 0.05 m` (5 cm).

**(2) `lifted` — vật đã được nhấc khỏi mặt nghỉ**
- `object_rest_z` là độ cao z ban đầu của vật, được chốt **một lần lúc khởi tạo sim**: trong `__init__`, sau `reset()`, nó lưu `self._object_rest_z[name] = body_pos(name)[2]` cho từng vật.
- `lifted = (z hiện tại − z nghỉ) ≥ lift_thresh = 0.03 m` (3 cm).

Chỉ khi **vừa sát vật VÀ vật đã bị nâng lên ≥ 3 cm** thì mới tính là `grasped`.

## Vì sao thiết kế kiểu này (logic phía sau)

Đây là cách "đo khách quan" để tránh model gian lận. Nếu chỉ dùng điều kiện gần (1), model có thể chỉ cần rê đầu kẹp tới sát vật là được tính kẹp dù chưa thực sự cầm được. Thêm điều kiện nhấc (2) đảm bảo: vật **đi theo đầu kẹp lên cao** → bằng chứng vật lý rằng nó đang được giữ. Vật nằm yên trên bàn (z ≈ z nghỉ) sẽ không bao giờ qua được điều kiện (2) dù đầu kẹp ở ngay cạnh.

Đây cũng là lý do trong `potential()` việc chuyển từ "thưởng gần vật" sang "thưởng gần giỏ" được gắn với `grasped` — vì nó là tín hiệu vật lý đáng tin, không phải nhãn `phase` do model tự khai.

## Hạn chế cần biết

Đây là **proxy hình học**, không phải phát hiện kẹp thật bằng tiếp xúc (contact force / friction), nên có vài kẽ hở:

1. **Không kiểm tra ngàm có thực sự đóng/đang giữ không.** Nếu một lệnh tình cờ hất vật bay lên cao > 3 cm trong lúc đầu kẹp ở gần (nhưng ngàm mở), `is_grasped` vẫn trả True dù vật chỉ đang bay tự do. Cú văng thoáng qua có thể bị tính nhầm là kẹp.
2. **Phụ thuộc tâm body.** `body_pos` trả tâm vật; với vật to/dài, đầu kẹp có thể chạm mép vật nhưng tâm vẫn > 5 cm → bị tính là chưa gần. Ngược lại vật nhỏ thì dễ thoả.
3. **`rest_z` cố định lúc init.** Nếu vật bị xê dịch/lăn sang chỗ thấp hơn rồi được nhấc lại, mốc so sánh vẫn là z ban đầu — thường ổn, nhưng nếu địa hình không phẳng thì mốc này có thể lệch.
4. **Ngưỡng cứng** (5 cm / 3 cm) không theo kích thước vật; nên cân nhắc tỉ lệ theo bán kính vật nếu palette có vật chênh lệch kích thước nhiều.

Nếu muốn chắc chắn hơn, có thể bổ sung kiểm tra **contact thật giữa hai ngàm và vật** từ `data.contact` của MuJoCo (cả hai má kẹp đều chạm vật) kết hợp với điều kiện nhấc hiện tại. Bạn có muốn tôi xem `data.contact` và thêm một hàm `is_grasped` dựa trên tiếp xúc thực để siết lại các kẽ hở trên không?

---

**User:**

Tôi thấy cách này khá triển vọng Shaping theo tư thế tiền-kẹp (pre-grasp pose), bạn nói rõ thêm về cách này giúp tôi trước khi thực hiện

---

**Assistant:**

Trước khi viết code, để tôi xem những đại lượng hình học nào hiện có sẵn (hướng/tư thế của đầu kẹp, trục tiếp cận) để phần giải thích bám sát đúng những gì pipeline có thể lấy được.

Tôi đã xem đủ phần hình học khả dụng. Dưới đây là giải thích đầy đủ ý tưởng "shaping theo tư thế tiền-kẹp" trước khi ta đụng vào code.

## Vấn đề mà nó giải quyết

Hiện tại, khi chưa kẹp:
```python
phi = w["approach"] * (-m.d_grip_obj)
```
Φ chỉ là hàm của **một con số**: khoảng cách thẳng đầu kẹp ↔ tâm vật. Gradient của nó luôn chỉ thẳng vào tâm vật — kể cả khi hướng đó đâm ngang làm văng vật. Đây chính là lý do case "vật sát ngàm" bị kẹt: hướng làm tăng Φ nhanh nhất lại là hướng phá.

Ý tưởng pre-grasp pose shaping: **thay "khoảng cách thẳng tới vật" bằng "khoảng cách tới một *tư thế tiền-kẹp tốt*"**. Tức là đổi đích của Φ từ "tâm vật" sang "điểm chờ phía trên vật, với đầu kẹp định hướng đúng". Khi đó gradient greedy sẽ tự nhiên dẫn cánh tay **đi lên → canh ngang phía trên vật → hạ xuống thẳng đứng**, thay vì lao ngang.

## Định nghĩa "tư thế tiền-kẹp" (pre-grasp pose)

Gồm hai phần:

**a) Vị trí — điểm chờ (standoff point)**
Một điểm nằm ngay phía trên tâm vật một khoảng `h_standoff`:
```
p_pre = (o_x, o_y, o_z + h_standoff)
```
với `o` = `body_pos(target)`. Đây là nơi cánh tay nên tới trước khi hạ xuống.

**b) Hướng — trục tiếp cận thẳng đứng**
Đầu kẹp nên áp xuống vật theo phương từ trên xuống. Định nghĩa vector từ vật tới đầu kẹp:
```
v = g - o      (g = grasp_point())
```
Khi đầu kẹp ở ngay trên vật, `v` gần như cùng phương `+ẑ` (trục z thế giới). Khi tiếp cận từ ngang, `v` gần phương ngang. Ta đo độ "thẳng đứng" bằng tích vô hướng chuẩn hoá:
```
align = (v · ẑ) / |v|        # = 1 nếu thẳng trên, ≈0 nếu ngang, <0 nếu từ dưới
```

## Cách dựng thế năng funnel (cái "phễu")

Tách `v` thành hai thành phần thay vì dùng một khoảng cách trần:
- `d_xy` = khoảng cách **ngang** giữa đầu kẹp và vật = `sqrt((g_x−o_x)² + (g_y−o_y)²)`
- `dz`   = chênh cao = `g_z − o_z`

Thế năng tiền-kẹp đề xuất (thay cho `-d_grip_obj`):
```
Φ_pre =  − w_xy · d_xy                      (1) luôn kéo đầu kẹp canh thẳng trên vật
         − w_z  · |dz − h_standoff| · gate  (2) chỉ kéo xuống đúng độ cao khi đã canh ngang
```
trong đó `gate = clip(1 − d_xy / r_funnel, 0, 1)`: một hệ số "mở phễu".

Ý nghĩa hành vi:
- Khi đầu kẹp còn lệch ngang nhiều (`d_xy` lớn → `gate ≈ 0`): term (2) gần như tắt, chỉ còn term (1) kéo cánh tay **canh cho thẳng phía trên vật**, không thưởng việc hạ thấp. Cánh tay không có động lực đâm ngang.
- Khi đã canh gần thẳng trên (`d_xy` nhỏ → `gate ≈ 1`): term (2) bật mạnh, kéo đầu kẹp **hạ xuống đúng độ cao kẹp**. Lúc này hạ thẳng đứng → ít va chạm ngang.

Đây chính là cái "phễu": miệng phễu rộng ở trên, thắt dần xuống điểm kẹp. Gradient greedy đi theo thành phễu = đi lên rồi vòng xuống đúng hướng.

Phương án có thêm hướng (tốn thêm chút việc): cộng một term thưởng `align` cao và/hoặc thưởng trục ngàm kẹp thực sự chĩa xuống vật (cần lấy ma trận xoay của site). Nhưng bản chỉ-dùng-vị-trí ở trên đã đủ giải quyết case của bạn mà không cần đo orientation.

## Tích hợp vào `potential()` thế nào

Quan trọng: PBRS chỉ giữ được tính chất "không làm lệch policy tối ưu" nếu Φ vẫn là **hàm thuần của trạng thái** (Markov). Cả `d_xy`, `dz`, `align` đều suy từ `g` và `o` ở trạng thái hiện tại → thoả mãn. Ta chỉ thay nhánh "chưa kẹp" trong `potential()`:

```python
def potential(m, weights=None):
    w = {...}
    if not m.grasped:
        phi = -(w_xy * m.d_xy) - w_z * abs(m.dz - h_standoff) * gate(m.d_xy)
    else:
        phi = w["grasped"] + w["carry"] * (-m.d_obj_basket)
    if m.in_basket:
        phi += w["in_basket"]
    return phi
```

Phần đã-kẹp và phần in_basket giữ nguyên. Chỉ pha approach được "phễu hoá".

Đây cũng là lý do cách này hợp với pipeline **GRPO single-step** của bạn: vì là greedy theo ΔΦ mỗi bước, ta không thể trông cậy telescoping nhiều bước; thay vào đó ta **nắn lại bề mặt Φ** để hướng dốc nhất tại mỗi điểm trỏ đúng đường lên-và-vòng. Reward dày giờ *dẫn đường* thay vì *cản* đường vòng.

## Đại lượng mới cần thêm

Vào `StateMeasure` (trong `sim/metrics.py`):
- `d_xy` : khoảng cách ngang đầu kẹp ↔ vật.
- `dz`   : `g_z − o_z`.
- (tuỳ chọn) `align` : nếu muốn term hướng.

Bản vị-trí dùng được ngay `grasp_point()` và `body_pos()` đang có. Nếu làm thêm term hướng/định-hướng ngàm thì cần thêm một accessor lấy ma trận xoay của site (`data.site_xmat` cho `gripperframe`) — hiện chưa có, phải viết thêm.

## Tham số cần tinh chỉnh & cạm bẫy

- `h_standoff` (độ cao điểm chờ): nên xấp xỉ chiều cao vật + vài cm. Quá cao thì cánh tay bay lố; quá thấp thì phễu không đủ chỗ né.
- `r_funnel` (bán kính miệng phễu): quyết định "canh ngang đến mức nào mới cho hạ". Nên ≥ kích thước vật + sai số.
- `w_xy` vs `w_z`: nếu `w_z` quá lớn so với `w_xy`, agent vẫn ưu tiên hạ sớm → mất tác dụng. Thường để `w_xy` trội hơn trong vùng ngoài phễu.
- **Khớp với `is_grasped`**: điểm thấp nhất của phễu (`dz = h_grasp`) phải nằm trong vùng mà `near_thresh`/`lift_thresh` của `is_grasped` công nhận, nếu không agent tới đáy phễu mà vẫn không được tính `grasped`.
- **Phụ thuộc giả định "kẹp từ trên xuống"**: nếu palette có vật chỉ kẹp được từ ngang (vật cao, mảnh), funnel thẳng đứng sẽ sai. Hiện tại đa số là gắp từ trên nên ổn, nhưng cần xác nhận với `object_palette`.
- Vẫn nên giữ song song khoản **phạt xê dịch vật trước khi kẹp** (đề xuất trước) để bắt mọi cú huých — funnel giảm xác suất va, phạt-xê-dịch bắt phần còn sót.

Nếu bạn đồng ý hướng này, tôi đề xuất làm bản **chỉ-dùng-vị-trí** trước (thêm `d_xy`, `dz` vào `StateMeasure`, phễu hoá nhánh approach trong `potential`, thêm tham số vào `GeoWeights`/`PHI_WEIGHTS`), chạy thử rồi mới cân nhắc thêm term hướng. Trước khi code, tôi muốn xác nhận hai điều: (1) các vật trong `object_palette` đều thiết kế để kẹp từ trên xuống chứ? (2) bạn muốn tôi gộp luôn phạt-xê-dịch-vật trong cùng lần sửa, hay tách riêng?

---
