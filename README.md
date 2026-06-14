# Mô phỏng cánh tay robot SO-101 — Gắp vật thể

Ứng dụng mô phỏng cánh tay robot **SO-101** (SO-ARM101) bằng [MuJoCo],
cho phép điều khiển cánh tay gắp các vật thể, với **mỗi camera hiển thị trong
một cửa sổ riêng** và **môi trường trò chơi cấu hình được** qua một file YAML.

Chạy native trên **Mac Mini (Apple Silicon)**, không cần GPU rời.

## Tính năng

- Dùng đúng model robot SO-101 đã tải về (`models/so101/`), **không thiết kế lại**.
- Điều khiển 6 khớp + kẹp gắp bằng bàn phím (chuyển động liên tục khi giữ phím).
- 3 cửa sổ độc lập, mỗi cửa sổ một camera (di chuyển/sắp xếp tự do):
  - **Operator** — góc nhìn người điều khiển, xoay/zoom bằng chuột.
  - **Camera 1 (Wrist)** — gắn cứng trên cổ tay, quan sát kẹp gắp so với vật.
  - **Camera 2 (Stand)** — gắn trên giá đỡ, quan sát toàn bộ cánh tay; **vị trí chỉnh được**.
- Mỗi cửa sổ có **nút "Chup man hinh"** (bấm chuột) và phím tắt `P`; ảnh PNG lưu
  vào thư mục `screenshots/`.
- Môi trường cấu hình hoàn toàn qua `config.yaml`: vật thể (hình dạng/kích thước/vị trí/
  số lượng/màu/khối lượng), vị trí đặt cánh tay, vị trí 2 camera, góc nhìn người điều khiển.

## Cài đặt & chạy

```bash
# Cách 1: dùng script (tự tạo venv + cài thư viện lần đầu)
./run.sh

# Cách 2: thủ công
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/python app.py
```

> Yêu cầu: Python 3.10+ (đã thử nghiệm trên Python 3.14 + macOS arm64).
> Chạy app trực tiếp bằng `python app.py` (cửa sổ GLFW chạy trên main thread,
> **không** cần `mjpython`).

Dùng file cấu hình khác:

```bash
./.venv/bin/python app.py duong/dan/config_cua_ban.yaml
```

## Điều khiển

| Phím | Tác dụng |
|------|----------|
| `A` / `D` | xoay thân (shoulder_pan) |
| `W` / `S` | vai lên/xuống (shoulder_lift) |
| `Q` / `E` | khuỷu tay (elbow_flex) |
| `R` / `F` | cổ tay gập (wrist_flex) |
| `Z` / `X` | cổ tay xoay (wrist_roll) |
| `O` / `C` | mở / đóng kẹp (gripper) |
| `SPACE` | mở–đóng kẹp nhanh |
| `H` | về tư thế home |
| `BACKSPACE` | reset toàn bộ (cánh tay + vật thể) |
| `P` | chụp màn hình cửa sổ đang dùng (lưu vào `screenshots/`) |
| `ESC` | thoát |
| Chuột (trong cửa sổ Operator) | kéo = xoay, chuột phải kéo = di chuyển, lăn = zoom |
| Bấm nút "Chup man hinh" (mỗi cửa sổ) | lưu ảnh PNG của cửa sổ đó |

## Cấu hình môi trường (`config.yaml`)

Mọi thiết lập "môi trường trò chơi" nằm trong `config.yaml`, không cần sửa code:

- `arm.base_pos`, `arm.base_euler_deg` — vị trí & hướng đặt cánh tay.
- `arm.home_pose` — tư thế ban đầu của 6 khớp.
- `objects[]` — danh sách vật cần gắp. Mỗi vật:
  - `shape`: `box` | `cylinder` | `sphere`
  - `size`: box → `[nửa_x, nửa_y, nửa_z]`; cylinder → `[bán_kính, nửa_cao]`; sphere → `[bán_kính]`
  - `pos`, `rgba`, `mass`
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

## Kiểm thử nhanh (không cần màn hình)

```bash
./.venv/bin/python smoke_test.py
```

Lệnh này dựng cảnh từ `config.yaml`, chạy mô phỏng, render cả 3 khung nhìn
offscreen và ghép thành `smoke_output.png` để kiểm tra trực quan.

## Cấu trúc dự án

```
robot_arm_sim_v2/
├── app.py                 # điểm vào: nạp config + mở cửa sổ 3 khung nhìn
├── smoke_test.py          # kiểm thử headless, xuất ảnh ghép
├── config.yaml            # cấu hình môi trường (sửa file này)
├── models/so101/          # model SO-101 gốc (MJCF + assets STL)
└── sim/
    ├── scene_builder.py   # dựng cảnh động từ config (MjSpec)
    ├── simulation.py      # lõi mô phỏng (MjModel/MjData, home, reset)
    ├── controls.py        # ánh xạ bàn phím
    └── viewer3.py         # mỗi camera một cửa sổ GLFW + nút chụp màn hình
```

## Ghi chú kỹ thuật

- Model robot lấy từ `SO-ARM100/Simulation/SO101/` (định dạng MJCF MuJoCo) và được
  giữ nguyên. Camera, vật thể, bàn, vị trí đế được **thêm vào lúc chạy** bằng API
  `mujoco.MjSpec` nên file model gốc không bị thay đổi.
- Camera cổ tay được tính quaternion để nhìn vào điểm gắp (site `gripperframe`),
  gắn cứng vào thân `gripper` như camera thật.
- Vật thể dùng ma sát cao và khối lượng nhẹ (~20 g) để cánh tay nhỏ gắp được;
  mô phỏng vật lý ở mức cơ bản đúng yêu cầu.

[MuJoCo]: https://mujoco.org
