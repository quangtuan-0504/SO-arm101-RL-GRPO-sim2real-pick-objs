# Mô phỏng cánh tay robot SO-101 — Gắp vật thể & điều khiển bằng VLM

Ứng dụng mô phỏng cánh tay robot **SO-101** (SO-ARM101) bằng [MuJoCo], với
**mỗi camera hiển thị trong một cửa sổ riêng** và **môi trường trò chơi cấu hình
được** qua một file YAML.

Chạy native trên **Mac Mini (Apple Silicon)**, không cần GPU rời (trừ tính năng
huấn luyện).

## Ba tính năng chính

| # | Tính năng | Điểm vào | Tài liệu |
|---|-----------|----------|----------|
| 1 | **Chơi bằng tay** — người dùng điều khiển cánh tay bằng bàn phím để gắp vật và thả vào giỏ | `app.py` | [Phần 1](#1-chơi-bằng-tay-điều-khiển-cánh-tay-qua-bàn-phím) |
| 2 | **VLM tự điều khiển** — một Vision-Language Model (Qwen-VL) nhìn 2 ảnh camera rồi tự sinh góc khớp để gắp-và-thả | `vlm/control_loop.py` | [Phần 2](#2-vlm-tự-động-điều-khiển-cánh-tay) |
| 3 | **Huấn luyện VLM bằng GRPO** — dạy VLM học gắp-và-thả qua tương tác môi trường | `train/grpo_train.py` | [Phần 3](#3-huấn-luyện-vlm-bằng-grpo) → chi tiết trong [`GRPO_PIPELINE.md`](GRPO_PIPELINE.md) |

Cả ba tính năng dùng chung một lõi mô phỏng (`sim/`) và một file cấu hình môi
trường (`config.yaml`), nên cảnh vật, giới hạn khớp và camera luôn nhất quán
giữa chơi tay, suy luận VLM và huấn luyện.

## Cài đặt

```bash
# Cách 1: dùng script (tự tạo venv + cài thư viện lần đầu, rồi chạy game)
./run.sh

# Cách 2: thủ công
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/python app.py
```

> Yêu cầu: Python 3.10+ (đã thử nghiệm trên Python 3.14 + macOS arm64).
> Chạy app trực tiếp bằng `python app.py` (cửa sổ GLFW chạy trên main thread,
> **không** cần `mjpython`).

`requirements.txt` gồm: `mujoco`, `numpy`, `PyYAML`, `glfw` (mô phỏng + hiển
thị), `openai` (gọi Qwen-VL cho tính năng 2), `fastapi`/`uvicorn`/`requests`
(game service cho tính năng 3). Riêng phần huấn luyện trên GPU dùng
`train/training_requirements.txt`.

---

## 1. Chơi bằng tay: điều khiển cánh tay qua bàn phím

```bash
./.venv/bin/python app.py                       # dùng config.yaml
./.venv/bin/python app.py duong/dan/config.yaml # dùng file cấu hình khác
```

### Đặc điểm

- Dùng đúng model robot SO-101 đã tải về (`models/so101/`), **không thiết kế lại**.
- Điều khiển 6 khớp + kẹp gắp bằng bàn phím (chuyển động liên tục khi giữ phím),
  hoặc **nhập góc bằng số (đơn vị độ)** cho từng khớp rồi áp dụng đồng thời.
- Mỗi khâu của cánh tay được **tô một màu riêng** (`link_colors` trong config)
  để dễ phân biệt khâu khi quan sát bằng mắt hoặc bằng VLM.
- 3 cửa sổ độc lập, mỗi cửa sổ một camera (di chuyển/sắp xếp tự do):
  - **Operator** — góc nhìn người điều khiển, xoay/zoom bằng chuột.
  - **Camera 1 (Wrist)** — gắn cứng trên cổ tay, quan sát kẹp gắp so với vật.
  - **Camera 2 (Stand)** — gắn trên giá đỡ, quan sát toàn bộ cánh tay; **vị trí chỉnh được**.
- Mỗi cửa sổ có **nút "Chup man hinh"** (bấm chuột) và phím tắt `P`; ảnh PNG lưu
  vào thư mục `screenshots/`.
- Nhiệm vụ: gắp vật rồi **thả vào giỏ** (`basket` trong config).

### Điều khiển

| Phím | Tác dụng |
|------|----------|
| `A` / `D` | xoay thân (shoulder_pan) |
| `W` / `S` | vai lên/xuống (shoulder_lift) |
| `Q` / `E` | khuỷu tay (elbow_flex) |
| `R` / `F` | cổ tay gập (wrist_flex) |
| `Z` / `X` | cổ tay xoay (wrist_roll) |
| `O` / `C` | mở / đóng kẹp (gripper) |
| `SPACE` | mở–đóng kẹp nhanh |
| `T` | bật/tắt chế độ **nhập góc bằng số (đơn vị độ)** |
| `TAB` / `SHIFT+TAB` | chọn khớp kế tiếp / trước đó khi nhập số |
| `ENTER` | chốt giá trị khớp hiện tại rồi chuyển sang khớp kế tiếp |
| `G` | áp dụng tất cả góc đã nhập cùng lúc (độ → radian) |
| `H` | về tư thế home |
| `BACKSPACE` | reset toàn bộ (cánh tay + vật thể) |
| `P` | chụp màn hình cửa sổ đang dùng (lưu vào `screenshots/`) |
| `ESC` | thoát |
| Chuột (trong cửa sổ Operator) | kéo = xoay, chuột phải kéo = di chuyển, lăn = zoom |
| Bấm nút "Chup man hinh" (mỗi cửa sổ) | lưu ảnh PNG của cửa sổ đó |

### Kiểm thử nhanh (không cần màn hình)

```bash
./.venv/bin/python smoke_test.py
```

Lệnh này dựng cảnh từ `config.yaml`, chạy mô phỏng, render cả 3 khung nhìn
offscreen và ghép thành `smoke_output.png` để kiểm tra trực quan.

---

## 2. VLM tự động điều khiển cánh tay

Thay vì người bấm phím, một **Vision-Language Model** (mặc định
`qwen3.6-flash` qua DashScope) nhìn ảnh 2 camera và tự quyết định góc 6 khớp cho
bước tiếp theo. Toàn bộ code nằm trong thư mục `vlm/`.

### Vòng lặp điều khiển (closed loop)

Mỗi bước suy luận:

1. Render offscreen 2 camera (`wrist_cam`, `stand_cam`) → encode PNG base64
   thành data URL (`vlm/render_utils.py`).
2. Dựng prompt + 2 ảnh thành `messages` theo chuẩn OpenAI (`vlm/prompt_builder.py`).
3. Gọi model, parse JSON trong phản hồi (`vlm/qwen_client.py`).
4. Áp `joint_angles_rad` vào sim bằng `set_ctrl` — **tự clip trong `ctrlrange`**
   nên model không thể đẩy khớp ra ngoài giới hạn.
5. Bước mô phỏng `settle_seconds` để PID đưa khớp tới góc mục tiêu.
6. Lặp lại đến khi model báo `done: true` hoặc đạt `--max-steps`.

Prompt được **sinh động từ `Simulation`**: bảng giới hạn khớp, tư thế home và
tư thế hiện tại đọc trực tiếp từ mô phỏng, nên prompt không bao giờ lệch khỏi
tham số thật. Prompt cũng mô tả 6 pha nhiệm vụ (`approach` → `align` → `grasp` →
`lift` → `carry` → `release`) và nhắc model rằng góc xuất ra phải **khác** tư thế
hiện tại (tránh việc model copy lại home pose và cánh tay đứng yên).

### JSON model phải trả về

```json
{
  "reasoning": "vị trí vật, giỏ, kẹp gắp trong 2 ảnh và bước kế tiếp",
  "target_object": "cube_red",
  "phase": "approach",
  "gripper_action": "open",
  "joint_angles_rad": {
    "shoulder_pan": 0.0, "shoulder_lift": -1.0, "elbow_flex": 0.9,
    "wrist_flex": 1.3, "wrist_roll": 1.57, "gripper": 1.2
  },
  "done": false,
  "confidence": 0.6
}
```

`done` chỉ được `true` khi vật đã nằm trong giỏ. Parser chịu được trường hợp
model bọc JSON trong khối ```` ```json ````.

### Cách chạy

Đặt API key trong file `.env` ở gốc dự án (hoặc biến môi trường
`QWEN_API_KEY` / `DASHSCOPE_API_KEY`):

```
QWEN_API_KEY=sk-...
# tuỳ chọn: QWEN_BASE_URL=https://dashscope-intl.aliyuncs.com/compatible-mode/v1
```

```bash
# Chạy thử KHÔNG cần API — client giả lập diễn kịch bản gắp-và-thả
./.venv/bin/python -m vlm.control_loop --target cube_red --mock

# Chạy thật, headless (không mở cửa sổ), in reasoning ra terminal
./.venv/bin/python -m vlm.control_loop \
    --target "the red cube (cube_red)" --max-steps 5 --no-thinking --verbose

# Chạy thật + mở 3 cửa sổ camera, VLM chạy song song ở thread nền
./.venv/bin/python -m vlm.control_loop \
    --target "the red cube (cube_red)" --max-steps 5 --no-thinking --verbose --view
```

### Tham số dòng lệnh

| Tham số | Mặc định | Ý nghĩa |
|---------|----------|---------|
| `--config` | `config.yaml` | file cấu hình môi trường |
| `--target` | `"the red cube (cube_red)"` | mô tả vật cần gắp (đưa vào prompt) |
| `--max-steps` | `12` | số lần suy luận tối đa; với `--view` đặt `0` = không giới hạn |
| `--settle-seconds` | `0.6` | thời gian mô phỏng chờ khớp tới góc mục tiêu sau mỗi bước |
| `--model` | `qwen3.6-flash` | tên model |
| `--api-key` / `--base-url` | đọc từ `.env` | ghi đè thông tin endpoint |
| `--no-thinking` | tắt | tắt chế độ deep-thinking (gọi không streaming, nhanh & rẻ hơn) |
| `--verbose` | tắt | in reasoning của model trong lúc stream |
| `--mock` | tắt | dùng `MockQwenClient`, không gọi mạng |
| `--mock-done-after` | `3` | bước thứ N mà mock chuyển từ `approach` sang `grasp` (rồi lift/carry/release) |
| `--view` | tắt | mở `ThreeViewApp` chạy song song với vòng lặp VLM |
| `--no-debug-img` | tắt | không lưu ảnh debug từng bước |

### Chế độ `--view` (hiển thị + VLM song song)

Trên macOS, GLFW và render **phải** ở main thread. Vì vậy:

- Main thread (`VLMDriver.on_frame`, gọi mỗi khung GLFW) render ảnh và áp góc.
- `VLMWorker` (thread nền) chỉ làm việc CPU/mạng: encode base64 → build prompt →
  gọi API → parse JSON.
- Chỉ giữ **một** yêu cầu suy luận tại một thời điểm; sau khi áp kết quả thì chờ
  `settle_seconds` rồi mới gửi cặp ảnh mới.

Nhờ vậy cửa sổ không bị đóng băng khi chờ mạng, và bạn **vẫn điều khiển tay bằng
bàn phím** trong lúc VLM đang chạy.

### Ảnh debug

Mỗi lần chạy tạo một thư mục `vlm_runs/<YYYYmmdd_HHMMSS>/` chứa
`stepNN_wrist.png` và `stepNN_stand.png` — chính xác những gì model đã "thấy" ở
từng bước. Rất hữu ích khi model ra quyết định sai và cần biết vì sao.

### File trong `vlm/`

```
vlm/
├── control_loop.py     # điểm vào: CLI, run_loop (headless), run_visual (--view), VLMDriver
├── prompt_builder.py   # system/user prompt + messages 2 ảnh (đọc giới hạn khớp từ sim)
├── qwen_client.py      # QwenVLClient (OpenAI SDK, streaming thinking) + MockQwenClient
├── worker.py           # VLMWorker: suy luận ở thread nền, giao tiếp qua queue
├── render_utils.py     # CameraRenderer + encode PNG/base64 thuần zlib (không cần PIL)
└── PROMPT_TEMPLATE.md  # bản prompt tham khảo dạng văn bản
```

`vlm/prompt_builder.py` chạy trực tiếp được để xem prompt sinh ra:

```bash
./.venv/bin/python -m vlm.prompt_builder
```

Ngoài ra `notebooks/ollama_qwen_colab.ipynb` là notebook thử self-host Qwen-VL
bằng Ollama (thay cho API trả phí).

---

## 3. Huấn luyện VLM bằng GRPO

Tính năng thứ ba dạy VLM **học** cách điều khiển cánh tay thay vì chỉ suy luận
zero-shot: dùng `trl` GRPOTrainer, mỗi trạng thái sinh G lệnh, branch cả G lệnh
trong mô phỏng, chấm điểm rồi commit lệnh tốt nhất.

Kiến trúc 2 máy: **Mac** chạy game service HTTP (FastAPI, không cần GPU), **server
GPU** chạy training + vLLM, nối với nhau bằng SSH reverse tunnel.

```
Mac (game/service.py)  ◄── HTTP ──►  GPU (train/grpo_train.py)
  /reset /branch /commit              trl GRPOTrainer + vLLM
```

Ba hàm reward: `format` (gate 0/1) → `geometric` (PBRS, tín hiệu chính) →
`judge` (VLM chấm điểm, nâng cấp sau).

Chuỗi lệnh tối giản:

```bash
# 1. Sinh dataset map/level (Mac, chạy 1 lần)
./.venv/bin/python -m dataset.generator --maps 5 --seed 0

# 2. Chạy game service (Mac) — cổng 8200, tránh 8000
caffeinate -dimsu ./.venv/bin/python -m game.service --port 8200

# 3. Huấn luyện (server GPU)
python -m train.grpo_train --model Qwen/Qwen2.5-VL-3B-Instruct \
    --game-url http://127.0.0.1:8200 --group-size 8 --use-vllm
```

> **Chi tiết đầy đủ xem [`GRPO_PIPELINE.md`](GRPO_PIPELINE.md)**: sơ đồ kiến
> trúc, bảng thành phần & trạng thái test, cách dựng SSH reverse tunnel từ Mac
> ra Vast AI, ràng buộc batching của `trl` (rất dễ sai), công thức từng hàm
> reward, và các bước tiếp theo được đề xuất.

Xem trước các map đã sinh:

```bash
./.venv/bin/python -m tools.preview_maps            # render tất cả map
./.venv/bin/python -m tools.preview_maps --map map2 # chỉ 1 map
```

---

## Cấu hình môi trường (`config.yaml`)

Mọi thiết lập "môi trường trò chơi" nằm trong `config.yaml`, không cần sửa code.
File này dùng cho cả tính năng 1 và 2 (tính năng 3 sinh map riêng qua
`dataset/generator.py`).

- `arm.base_pos`, `arm.base_euler_deg` — vị trí & hướng đặt cánh tay.
- `arm.home_pose` — tư thế ban đầu của 6 khớp (radian).
- `link_colors` — màu từng khâu cánh tay (khoá = tên body trong model gốc).
- `objects[]` — danh sách vật cần gắp. Mỗi vật:
  - `shape`: `box` | `cylinder` | `sphere`
  - `size`: box → `[nửa_x, nửa_y, nửa_z]`; cylinder → `[bán_kính, nửa_cao]`; sphere → `[bán_kính]`
  - `pos`, `rgba`, `mass`
- `basket` — giỏ tĩnh để thả vật vào: `enabled`, `pos`, `inner_size`,
  `wall_thickness`, `floor_thickness`, `rgba`.
- `table` — mặt bàn tuỳ chọn (`enabled: true/false`).
- `wrist_camera` — camera 1: khoảng lùi `back`, nâng `up`, góc nhìn `fovy`.
- `stand_camera` — camera 2: `pos` (vị trí giá đỡ, **chỉnh được**), `look_at`, `fovy`.
- `operator_camera` — góc nhìn ban đầu của người điều khiển.
- `window` — kích thước các cửa sổ: `operator_width/height` cho cửa sổ Operator,
  `camera_width/height` cho hai cửa sổ Camera 1 & 2.

Ví dụ thêm một vật hình cầu:

```yaml
objects:
  - name: ball
    shape: sphere
    size: [0.02]
    pos: [0.16, 0.0, 0.02]
    rgba: [0.9, 0.7, 0.1, 1.0]
    mass: 0.02
```

## Cấu trúc dự án

```
robot_arm_sim_v2/
├── app.py                 # [TN1] điểm vào game: nạp config + mở 3 cửa sổ
├── smoke_test.py          # kiểm thử headless, xuất ảnh ghép
├── config.yaml            # cấu hình môi trường (sửa file này)
├── run.sh                 # tạo venv + cài thư viện + chạy app
├── requirements.txt       # thư viện cho Mac (mô phỏng, VLM, game service)
├── models/so101/          # model SO-101 gốc (MJCF + assets STL)
│
├── sim/                   # lõi mô phỏng, dùng chung cho cả 3 tính năng
│   ├── scene_builder.py   #   dựng cảnh động từ config (MjSpec)
│   ├── simulation.py      #   MjModel/MjData, home, reset, get_state/set_state
│   ├── metrics.py         #   đo hình học + potential cho PBRS
│   ├── constants.py       #   tên khớp, hằng số dùng chung
│   ├── controls.py        #   ánh xạ bàn phím
│   └── viewer3.py         #   mỗi camera một cửa sổ GLFW + nút chụp màn hình
│
├── vlm/                   # [TN2] VLM tự điều khiển (xem phần 2)
│
├── game/                  # [TN3] môi trường + HTTP service cho huấn luyện
│   ├── env.py             #   1 màn: reset / branch / commit
│   └── service.py         #   FastAPI service chạy trên Mac
├── dataset/               # [TN3] sinh map & level table
│   ├── generator.py       #   → levels.json, robot_spec.json
│   └── object_palette.py
├── rewards/               # [TN3] format / geometric (PBRS) / judge
├── train/                 # [TN3] GRPO trainer, policy, rollout, logging, plot
├── tools/preview_maps.py  # render ảnh xem trước các map đã sinh
├── tests/                 # kiểm thử chạy được trên Mac (script độc lập)
├── notebooks/             # thử self-host Qwen-VL bằng Ollama
│
├── screenshots/           # ảnh chụp từ game (TN1)
├── vlm_runs/              # ảnh debug từng bước suy luận (TN2)
├── logs/                  # frame + metrics của các phiên huấn luyện (TN3)
├── GRPO_PIPELINE.md       # tài liệu chi tiết pipeline huấn luyện
└── chat_with_kiro/        # nhật ký thiết kế theo từng phiên trao đổi
```

## Kiểm thử

Các test trong `tests/` là **script độc lập** (không dùng pytest), chạy từng file
bằng `-m`:

```bash
./.venv/bin/python -m tests.test_metrics_snapshot   # lưu/khôi phục state + accessor hình học
./.venv/bin/python -m tests.test_env_rewards        # reward của môi trường 1 màn
./.venv/bin/python -m tests.test_game_service       # game service HTTP
./.venv/bin/python -m tests.test_rollout            # rollout 1 màn (MockPolicy)
./.venv/bin/python -m tests.test_episode_manager    # pool màn song song (MockPolicy)
```

Tất cả chạy được hoàn toàn trên Mac (dùng `MockPolicy` cho phần cần VLM). Phần
cần GPU (`train/policy.py`, `train/grpo_train.py`) chưa có test — xem bảng trạng
thái trong [`GRPO_PIPELINE.md`](GRPO_PIPELINE.md).

## Ghi chú kỹ thuật

- Model robot lấy từ `SO-ARM100/Simulation/SO101/` (định dạng MJCF MuJoCo) và được
  giữ nguyên. Camera, vật thể, bàn, giỏ, vị trí đế được **thêm vào lúc chạy** bằng
  API `mujoco.MjSpec` nên file model gốc không bị thay đổi.
- Camera cổ tay được tính quaternion để nhìn vào điểm gắp (site `gripperframe`),
  gắn cứng vào thân `gripper` như camera thật.
- Vật thể dùng ma sát cao và khối lượng nhẹ (~20 g) để cánh tay nhỏ gắp được;
  mô phỏng vật lý ở mức cơ bản đúng yêu cầu.
- 6 khớp là **position motor**: ta đặt góc mục tiêu tuyệt đối, PID nội bộ đưa
  khớp tới đó. `Simulation.set_ctrl` luôn clip vào `ctrlrange`, nên cả bàn phím
  lẫn VLM đều không thể đẩy khớp ra ngoài giới hạn an toàn.
- Quy ước kẹp gắp (`gripper`): giá trị **lớn** = mở rộng, giá trị **nhỏ** (gần
  giới hạn dưới) = kẹp chặt.
- Encode PNG trong `vlm/render_utils.py` viết bằng `zlib` + `struct` thuần, không
  cần Pillow — giảm phụ thuộc khi chạy trên máy khác.

[MuJoCo]: https://mujoco.org
