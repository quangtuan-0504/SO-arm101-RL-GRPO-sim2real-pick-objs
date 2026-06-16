# New Session

Exported: 6/16/2026, 9:01:55 PM

---

**User:**

Tôi muốn dùng VLM để dự đoán các góc quay để điều khiển robot gắp vật thể, hãy giúp tôi thiết kế prompt để cho vào model Qwen3.5 nhằm dự đoán ra các góc quay điều khiển. Trong prompt sẽ gồm mô tả yêu cầu, mục tiêu, mô tả cách điều khiển cánh tay robot, 2 hình ảnh từ 2 camera chứa thông tin hiện trạng của cánh tay robot và các vật thể, mô tả format đầu ra

---

**Assistant:**

Tôi sẽ đọc các file cấu hình và mã nguồn để hiểu rõ cấu trúc robot, các khớp, camera và giới hạn góc quay, từ đó thiết kế prompt chính xác.

Tôi cần xem giới hạn góc của các khớp (control range) trong model và cách simulation đọc/ghi góc.

Tôi đã nắm đủ thông tin về robot SO-101: 6 khớp, giới hạn điều khiển (radian), tư thế home và 2 camera. Dưới đây là prompt được thiết kế cho Qwen-VL, kèm một module Python để tự sinh prompt (đọc giới hạn góc động từ config/model nên luôn khớp với mô phỏng).

Trước tiên, đây là tóm tắt các thông số then chốt mà tôi trích từ mã nguồn để đưa vào prompt:

| Khớp | ctrlrange (rad) | Home (rad) | Ý nghĩa |
|------|-----------------|------------|---------|
| shoulder_pan | -1.920 … 1.920 | 0.0 | xoay thân quanh trục đứng |
| shoulder_lift | -1.745 … 1.745 | -0.8 | nâng/hạ vai |
| elbow_flex | -1.690 … 1.690 | 1.2 | gập khuỷu |
| wrist_flex | -1.658 … 1.658 | 0.9 | gập cổ tay |
| wrist_roll | -2.744 … 2.841 | 1.57 | xoay cổ tay |
| gripper | -0.175 … 1.745 | 1.0 | -0.175≈đóng, 1.745≈mở hết |

Tôi sẽ tạo một module sinh prompt và một file prompt mẫu để bạn dùng ngay.

Giờ tôi tạo file prompt mẫu (đã render sẵn nội dung văn bản) để bạn xem trực tiếp và copy nếu muốn dùng tay.

Tôi kiểm tra nhanh module sinh prompt chạy đúng với mô phỏng thật.

Prompt render đúng nhưng bị thừa thụt lề 8 dấu cách ở mỗi dòng (do `dedent` không cắt được khi bảng chèn vào không có thụt lề). Tôi sửa lại cho sạch.

Giờ bỏ import `dedent` không dùng nữa và chạy lại kiểm tra.

Module chạy đúng, prompt render sạch sẽ. Xong.

Tôi đã thiết kế prompt cho Qwen-VL và đặt vào thư mục mới `vlm/`:

- `vlm/prompt_builder.py` — sinh prompt động (đọc giới hạn góc trực tiếp từ `Simulation` nên luôn khớp model/config). Hàm chính: `build_messages(sim, target_desc, wrist_img_b64, stand_img_b64)` trả về `messages` định dạng OpenAI-compatible mà API Qwen-VL nhận trực tiếp.
- `vlm/PROMPT_TEMPLATE.md` — bản render sẵn để bạn đọc/copy thủ công + ghi chú thiết kế.

Cấu trúc prompt gồm đúng các phần bạn yêu cầu:

1. **System prompt** — vai trò (bộ điều khiển thị giác cho SO-101) + 5 nguyên tắc bất biến (chỉ suy luận từ ảnh, góc phải trong giới hạn, điều khiển vòng kín, chỉ đóng kẹp khi căn đúng, chỉ xuất JSON).
2. **Mục tiêu** — nêu rõ vật cần gắp.
3. **Mô tả 2 camera** — ảnh 1 wrist_cam (cận cảnh để căn kẹp), ảnh 2 stand_cam (toàn cảnh để ước lượng tư thế/độ cao/khoảng cách).
4. **Mô tả cách điều khiển** — bảng 6 khớp với tên, giới hạn radian thực tế, tư thế home và ý nghĩa/chiều dương từng khớp; quy ước kẹp; 4 pha gắp (approach → align → grasp → lift).
5. **2 ảnh** — đính kèm đúng thứ tự sau phần text.
6. **Format đầu ra** — JSON nghiêm ngặt: `reasoning`, `target_object`, `phase`, `gripper_action`, `joint_angles_rad` (đủ 6 khớp), `done`, `confidence`, kèm ví dụ hợp lệ.

Vài lựa chọn thiết kế đáng lưu ý:

- **Dùng radian** thay vì độ vì `Simulation.set_ctrl` và actuator của model đều nhận radian — tránh một bước quy đổi dễ sai. Nếu muốn Qwen suy luận theo độ, chỉ cần đổi prompt rồi `math.radians()` trước khi áp dụng.
- **Điều khiển vòng kín theo pha + cờ `done`**: gắp xong trong một lần dự đoán từ ảnh rất khó, nên thiết kế để gọi lặp (chụp ảnh → hỏi model → set_ctrl → step → lặp). Đầu ra đã có `set_ctrl` tự `clip` vào giới hạn nên model lỡ vượt biên vẫn an toàn.

Bạn có muốn tôi viết luôn vòng lặp điều khiển (chụp ảnh 2 camera → gọi API Qwen → parse JSON → `set_ctrl` → bước mô phỏng) để chạy thử end-to-end không?

---

**User:**

viết luôn vòng lặp điều khiển (chụp ảnh 2 camera → gọi API Qwen → parse JSON → set_ctrl → bước mô phỏng) để chạy thử end-to-end

---

**Assistant:**

Tôi cần xem cách project render ảnh từ camera (offscreen) để chụp đúng wrist_cam và stand_cam, rồi mới viết vòng lặp.

Tôi xem requirements và các package có sẵn trong venv để chọn cách gọi API phù hợp.

Không có PIL/requests/openai. Để giữ "zero extra dependency" như phong cách project (tự encode PNG bằng zlib), tôi sẽ render bằng `mujoco.Renderer`, encode PNG thủ công, và gọi API qua `urllib` của stdlib. Kèm chế độ mock để chạy thử end-to-end không cần API key.

Giờ tạo client gọi Qwen-VL (OpenAI-compatible qua urllib) + parser JSON + mock client.

Giờ viết vòng lặp điều khiển chính, kèm CLI.

Giờ chạy thử end-to-end ở chế độ mock để xác nhận đường ống render → parse → set_ctrl → step hoạt động.

Hoạt động end-to-end. Kiểm tra nhanh ảnh debug có được render và lưu hợp lệ.

Mọi thứ chạy đúng và render ra ảnh PNG hợp lệ. Tôi thêm `vlm_runs/` vào `.gitignore` vì đó là output debug.

Xong. Vòng lặp điều khiển end-to-end đã chạy và verify được (chế độ mock, 4 bước → DONE, render ra 8 ảnh PNG hợp lệ). Toàn bộ chỉ dùng stdlib + mujoco/numpy, không cần cài thêm package.

Các file mới trong `vlm/`:

- `render_utils.py` — `CameraRenderer` render offscreen 2 camera (`wrist_cam`, `stand_cam`) qua `mujoco.Renderer`, và encode RGB → data URL base64 PNG (tự encode bằng zlib, không cần PIL).
- `qwen_client.py` — `QwenVLClient` gọi endpoint OpenAI-compatible qua `urllib`, `parse_control_json` chịu được khi model bọc ` ```json `, và `MockQwenClient` để chạy thử không cần mạng.
- `control_loop.py` — vòng lặp chính + CLI: render → build messages → predict → `set_ctrl` (tự clip giới hạn) → `sim.step` để cánh tay tiến tới góc mục tiêu → lặp đến khi `done` hoặc hết `max_steps`. Lưu ảnh từng bước vào `vlm_runs/<timestamp>/` để debug.

Cách dùng:

```bash
# Chạy thử end-to-end không cần API key
.venv/bin/python -m vlm.control_loop --target cube_red --mock

# Chạy thật với Qwen-VL (DashScope compatible-mode)
export QWEN_API_KEY=sk-xxx
export QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
.venv/bin/python -m vlm.control_loop --target "khoi lap phuong do cube_red" --model qwen-vl-max-latest
```

Vài điểm thiết kế đáng lưu ý:

- **Vòng kín có "settle"**: sau mỗi dự đoán, mô phỏng chạy `settle_seconds` (mặc định 0.6s) để PID đưa khớp tới góc mục tiêu trước khi chụp ảnh mới — phản ánh đúng trạng thái cho lần dự đoán sau.
- **An toàn biên**: `apply_prediction` đưa qua `sim.set_ctrl`, vốn đã `np.clip` vào `ctrlrange`, nên model lỡ xuất góc vượt biên vẫn không gây lỗi.
- **`--mock`** dùng kịch bản đơn giản (approach → grasp đóng kẹp → lift + done) chỉ để xác nhận đường ống; logic gắp thật do Qwen-VL quyết định khi chạy chế độ thật.

Lưu ý về model: `app.py` mặc định đặt cánh tay ở home và mô phỏng dùng position actuator, nên một lần `set_ctrl` rồi step ngắn có thể chưa tới đúng góc nếu bước nhảy lớn — bạn có thể tăng `--settle-seconds` nếu thấy cánh tay chưa kịp ổn định giữa các lần gọi. Bạn có muốn tôi thêm tùy chọn hiển thị trực quan (mở cửa sổ `ThreeViewApp`) chạy song song với vòng lặp VLM để xem cánh tay di chuyển theo thời gian thực không?

---

**User:**

Tôi sử dụng model VLM qwen3.6-flash được cung cấp từ QwenCloud, dưới đây là mã mà họ hướng dẫn gọi tới model
Cách 1 dùng OpenAI Completions API
from openai import OpenAI
import os
client = OpenAI(
# If the environment variable is not set, replace it with your Model Studio API key: api_key="sk-xxx"
api_key=os.getenv("DASHSCOPE_API_KEY"),
base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
)
messages = [{"role": "user", "content": "Who are you"}]
completion = client.chat.completions.create(
model="qwen3.6-flash",  # You can replace this with another deep thinking models
messages=messages,
extra_body={"enable_thinking": True},
stream=True
)
is_answering = False  # Indicates whether the response phase has started
print("\n" + "=" * 20 + "Thinking process" + "=" * 20)
for chunk in completion:
if not chunk.choices:
continue
delta = chunk.choices[0].delta
if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
if not is_answering:
print(delta.reasoning_content, end="", flush=True)
if hasattr(delta, "content") and delta.content:
if not is_answering:
print("\n" + "=" * 20 + "Full response" + "=" * 20)
is_answering = True
print(delta.content, end="", flush=True)
Cách 2 dùng cURL bạn có thể dùng thư viện python để viết mã
curl -X POST https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions \
-H "Authorization: Bearer $DASHSCOPE_API_KEY" \
-H "Content-Type: application/json" \
-d '{
"model": "qwen3.6-flash",
"messages": [
{
"role": "user", 
"content": "Who are you"
}
],
"stream": true,
"enable_thinking": true
}'
Hãy chọn một cách mà bạn thấy tối ưu hơn và sửa mã để có thể sử dụng tương thích với model

---

**Assistant:**

Tôi xem file `.env` để biết tên biến môi trường cần nạp (không in giá trị bí mật ra).

`.env` dùng biến `QWEN_API_KEY`. Một lưu ý bảo mật: `.env` chứa API key thật nhưng chưa nằm trong `.gitignore` — tôi sẽ thêm vào để tránh commit nhầm key.

Về lựa chọn cách gọi: tôi chọn **Cách 1 (OpenAI SDK)** vì model `qwen3.6-flash` là model "deep thinking" cần `enable_thinking` + `stream=True`, và SDK xử lý SSE streaming + tách `reasoning_content`/`content` sạch sẽ, ít lỗi hơn nhiều so với tự parse SSE bằng urllib. Đánh đổi là thêm dependency `openai`, chấp nhận được.

Tôi cập nhật `qwen_client.py` để dùng OpenAI SDK với streaming + thinking, và `control_loop.py` để nạp `.env` + dùng base_url/model mới.

Giờ cập nhật `control_loop.py`: nạp `.env`, đổi base_url/model mặc định, hỗ trợ biến `QWEN_API_KEY`/`DASHSCOPE_API_KEY`, và cờ thinking/verbose.

Giờ thêm `openai` vào requirements, thêm `.env` vào `.gitignore`, và cài `openai` vào venv.

Cài xong openai. Kiểm tra lại mock vẫn chạy và `.env` được nạp đúng (không in giá trị key).

Mock chạy đúng. Giờ kiểm tra kết nối thật tới `qwen3.6-flash` để xác nhận key/endpoint/streaming hoạt động (gọi text ngắn, chưa parse JSON).

Kết nối thật OK. Giờ chạy 1 bước thật với 2 ảnh để xác nhận model nhận ảnh và trả JSON parse được.

Hoạt động end-to-end với `qwen3.6-flash` thật: model nhận cả 2 ảnh, suy luận được vị trí khối đỏ ("nằm lệch về bên phải khung hình"), trả JSON hợp lệ, parse → clip → `set_ctrl` → step thành công.

Tóm tắt thay đổi:

Tôi chọn **Cách 1 (OpenAI SDK)** thay vì tự parse SSE bằng urllib, vì `qwen3.6-flash` là model deep-thinking cần `enable_thinking=True` + `stream=True` và trả `reasoning_content` tách riêng — SDK xử lý streaming/parse gọn và bền hơn.

- `vlm/qwen_client.py`: `QwenVLClient` viết lại dùng `openai.OpenAI`. Mặc định `enable_thinking=True` → gọi streaming, gom phần `content` (bỏ qua `reasoning_content`, có `--verbose` để in suy luận). Khi `--no-thinking` thì gọi không streaming. `parse_control_json` và `MockQwenClient` giữ nguyên.
- `vlm/control_loop.py`: thêm `load_dotenv` (stdlib, tự đọc `.env` dạng `KEY=value`), đổi mặc định `base_url=https://dashscope-intl.aliyuncs.com/compatible-mode/v1`, `model=qwen3.6-flash`, đọc key theo thứ tự `--api-key` → `QWEN_API_KEY` → `DASHSCOPE_API_KEY`. Thêm cờ `--no-thinking`, `--verbose`.
- `requirements.txt`: thêm `openai>=1.30` (đã cài vào `.venv`).
- `.gitignore`: thêm `.env` để không commit nhầm API key.

Cách chạy:

```bash
# Thật (key tự đọc từ .env)
.venv/bin/python -m vlm.control_loop --target "khoi lap phuong do cube_red"

# Xem quá trình suy luận của model
.venv/bin/python -m vlm.control_loop --target cube_red --verbose

# Mock, không cần mạng
.venv/bin/python -m vlm.control_loop --target cube_red --mock
```

Một lưu ý bảo mật: API key trong `.env` đang là key thật và trước đó chưa được gitignore. Tôi đã thêm `.env` vào `.gitignore`, nhưng nếu file này từng được commit lên git thì bạn nên thu hồi (revoke) key đó và tạo key mới, vì nó vẫn còn trong lịch sử git.

---

**User:**

thêm tùy chọn hiển thị trực quan (mở cửa sổ ThreeViewApp) chạy song song với vòng lặp VLM để xem cánh tay di chuyển theo thời gian thực 

---

**Assistant:**

Tôi sẽ thêm chế độ trực quan: cửa sổ `ThreeViewApp` chạy ở main thread (bắt buộc với GLFW trên macOS), còn lệnh gọi Qwen chạy ở thread nền để GUI không bị treo trong lúc chờ mạng. Main thread render ảnh và áp góc; thread nền chỉ làm network I/O.

Trước hết, thêm hook tùy chọn vào vòng lặp GLFW của `ThreeViewApp`.

Giờ tạo worker chạy inference ở thread nền (thread-safe, không đụng GL).

Giờ thêm `VLMDriver` và chế độ `--view` vào `control_loop.py`.

Giờ thêm `VLMDriver` và `run_visual` (đặt trước `build_client`).

I'll insert the new code right before `build_client`.

Giờ thêm cờ `--view` và rẽ nhánh trong `main`.

Tôi vô tình tạo lỗi xuống dòng trong chuỗi help. Sửa lại.

Giờ kiểm tra logic worker + driver headless (dùng mock + ảnh giả) mà không cần mở cửa sổ GLFW.

Worker chạy đúng (approach→grasp→lift→done). Xác nhận CLI `--view` đã có và module import sạch.

Đã xong. Tôi không thể mở cửa sổ GLFW trong môi trường này (không có display) để chạy thử trực quan, nhưng toàn bộ logic ghép nối đã được kiểm tra: hook `on_frame` trong vòng lặp GLFW, worker thread (mock chạy đúng chuỗi approach→grasp→lift→done), driver áp góc + gửi ảnh, và nhánh CLI `--view`.

Tóm tắt thay đổi:

- `sim/viewer3.py` — `ThreeViewApp.run(on_frame=None)`: thêm callback tùy chọn gọi mỗi khung sau khi bước mô phỏng. Không ảnh hưởng cách dùng cũ.
- `vlm/worker.py` (mới) — `VLMWorker`: thread nền nhận cặp ảnh RGB (numpy) → encode base64 → build messages → gọi Qwen → parse JSON, trả kết quả qua hàng đợi. Chỉ làm CPU/mạng, không đụng OpenGL (an toàn trên macOS).
- `vlm/control_loop.py` — thêm `VLMDriver` + `run_visual()` + cờ `--view`. Driver tạo `CameraRenderer` ở main thread, mỗi khung: áp kết quả mới (nếu có) rồi gửi ảnh mới khi worker rảnh và đã qua `settle_seconds`.

Kiến trúc chạy song song:
- Main thread: GLFW render 3 cửa sổ + step mô phỏng liên tục ~60fps → bạn thấy cánh tay di chuyển mượt theo thời gian thực; vẫn điều khiển tay bằng bàn phím được.
- Thread nền: gọi `qwen3.6-flash` (chậm do mạng) mà không làm treo GUI. Mỗi lúc chỉ một yêu cầu inference; áp xong thì chờ arm ổn định rồi mới chụp ảnh gửi tiếp.

Cách chạy:

```bash
# Trực quan + VLM thật
.venv/bin/python -m vlm.control_loop --target "khoi lap phuong do cube_red" --view

# Trực quan + mock (xem cơ chế chạy, không cần mạng)
.venv/bin/python -m vlm.control_loop --target cube_red --view --mock

# Headless như cũ (không cửa sổ)
.venv/bin/python -m vlm.control_loop --target cube_red
```

Lưu ý khi bạn chạy máy local: `VLMDriver` dùng một `mujoco.Renderer` riêng (context GL thứ hai) song song với cửa sổ GLFW. Mỗi `window.render()` đều `make_context_current` lại nên thường không xung đột. Nếu trên máy bạn gặp lỗi context khi bật `--view`, báo tôi — phương án dự phòng là đọc pixel trực tiếp từ cửa sổ Camera 1/Camera 2 (qua `mjr_readPixels`) thay vì tạo renderer thứ hai, để tránh hoàn toàn context phụ.

---

**User:**

Tôi chưa thấy sử dụng args.max_steps khi chạy --view

---

**Assistant:**

Đúng, `run_visual` bỏ qua `max_steps`. Ở chế độ `--view`, vòng lặp chạy đến khi đóng cửa sổ hoặc model báo `done`, nên `max_steps` không có tác dụng. Tôi thêm giới hạn số lần inference cho driver.

Giờ cho `run_visual` nhận và truyền `max_steps`.

Cập nhật `main` truyền `max_steps` và làm rõ ý nghĩa trong help.

Kiểm tra logic giới hạn `max_steps` của driver chạy đúng headless (mock, không cần GUI).

Đã sửa. Giờ `--view` tôn trọng `max_steps`: driver dừng gửi ảnh sau khi đạt số lần inference tối đa, in thông báo một lần và vẫn giữ cửa sổ mở để bạn xem/điều khiển tay đến khi nhấn ESC.

Thay đổi trong `vlm/control_loop.py`:
- `VLMDriver` nhận tham số `max_steps` (0 = không giới hạn). Khi `_step_idx >= max_steps`, ngừng submit, thông báo một lần.
- `run_visual` nhận và truyền `max_steps` xuống driver.
- `main` truyền `args.max_steps` vào `run_visual`.
- Help của `--max-steps` ghi rõ: với `--view`, đặt `0` để chạy không giới hạn (chỉ dừng khi model báo `done` hoặc đóng cửa sổ).

Đã verify headless với mock: driver dừng đúng sau 3 lần khi `--max-steps 3`.

Cách dùng:
```bash
# view, dừng sau 12 lần inference (mặc định)
.venv/bin/python -m vlm.control_loop --target cube_red --view

# view, chạy không giới hạn đến khi done hoặc ESC
.venv/bin/python -m vlm.control_loop --target cube_red --view --max-steps 0
```

---
