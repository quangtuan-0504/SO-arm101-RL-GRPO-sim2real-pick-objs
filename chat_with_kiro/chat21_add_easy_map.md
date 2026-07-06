# tôi muốn tạo thêm 5 map, mỗi map chỉ có một vật...

Exported: 7/6/2026, 11:20:15 AM

---

**User:**

tôi muốn tạo thêm 5 map, mỗi map chỉ có một vật thể. Mục đích là để cho mô hình học từ dễ lên khó thay vì học chơi trong những map khó ngay từ đầu

---

**Assistant:**

Để tôi xem cấu trúc map hiện tại trước khi thêm map mới.

Để tôi xem cách training/service đọc levels và thứ tự dùng map.

Tôi đã nắm cấu trúc. Mỗi map hiện có nhiều vật (distractor) và 3 level. Tôi sẽ thêm 5 map mới (map5–map9), mỗi map **chỉ 1 vật** và 1 level tương ứng — đây là các màn dễ nhất để mô hình học trước. Tôi giữ nguyên map0–map4.

Xem phần cuối file để chèn chính xác:

Giờ chèn 5 map mới (mỗi map 1 vật, 1 level). Tôi đặt vật ở các vị trí khác nhau trong vùng với tới, luân phiên 3 loại để mô hình làm quen từng loại trong bối cảnh không có vật gây nhiễu.

Giờ kiểm tra JSON hợp lệ và đếm số level/map:

Xong. JSON hợp lệ. Đã thêm 5 map mới vào `dataset/levels.json`, giữ nguyên map0–map4:

- **map5** – chỉ `cube_red` tại (0.17, 0.0)
- **map6** – chỉ `cyl_green` tại (0.18, 0.03)
- **map7** – chỉ `sphere_blue` tại (0.16, -0.03)
- **map8** – chỉ `cube_red` tại (0.205, 0.06)
- **map9** – chỉ `sphere_blue` tại (0.15, -0.06)

Mỗi map chỉ có 1 vật và 1 level, không có vật gây nhiễu — đây là các màn dễ nhất để mô hình học gắp trước khi chuyển sang map nhiều vật. Tôi luân phiên qua cả 3 loại (đỏ/xanh lá/xanh dương) và đặt ở các vị trí khác nhau trong vùng với tới, tránh vùng giỏ.

Hai lưu ý:

1. `_pick_level()` trong `train/episode_manager.py` đang chọn level **ngẫu nhiên đều** trên toàn bộ list. Việc thêm các map dễ chỉ làm tăng tỉ lệ gặp màn dễ, chứ chưa phải curriculum thật sự "dễ trước, khó sau". Nếu bạn muốn ép thứ tự học (ưu tiên map đơn vật ở đầu rồi giảm dần), tôi có thể chỉnh `_pick_level` thành lấy mẫu có trọng số hoặc theo lịch (schedule) theo bước huấn luyện.

2. Các map này được thêm trực tiếp vào `levels.json`. Nếu chạy lại `python -m dataset.generator` thì file sẽ bị ghi đè và mất 5 map này (đồng thời generator hiện chưa sinh `instruction_en`). Nếu muốn tái tạo được, tôi có thể bổ sung hàm sinh map đơn-vật + `instruction_en` vào `generator.py`.

Bạn muốn tôi làm tiếp ý 1 hoặc ý 2 không?

---

**User:**

làm tiếp ý 2

---

**Assistant:**

Tôi sẽ cập nhật generator để (a) sinh `instruction_en`, và (b) sinh các map đơn-vật làm curriculum dễ. Trước hết thêm `desc_en` cho từng loại vật.

Giờ cập nhật `generator.py`: thêm `instruction_en`, thêm hàm sinh map đơn-vật, và mở rộng `generate()` + `main()`.

Giờ cập nhật `main()` để thêm tham số `--easy-maps` và tóm tắt:

Cập nhật docstring đầu file cho khớp:

Giờ test `generate()` (không gọi `robot_spec` vốn cần MuJoCo) để xác nhận cấu trúc đúng:

Dùng venv của project:

Xác nhận generator giờ sinh đúng cấu trúc + có `instruction_en`. Kiểm tra map0–map4 sinh ra có khớp với `levels.json` hiện tại không (để regenerate không làm đổi các map khó):

map0–map4 tái tạo **giống hệt** (id, objects, instruction_en đều khớp). Giờ tôi ghi lại `levels.json` từ generator để file khớp chính xác với output của generator (map dễ có vị trí/loại chuẩn hoá, nhất quán khi regenerate). `robot_spec.json` không cần đụng vì giới hạn khớp không đổi.

Xong ý 2. Generator giờ tự sinh được mọi thứ và `levels.json` đã khớp chính xác với output của generator nên có thể tái tạo bất cứ lúc nào.

Thay đổi đã làm:

**`dataset/object_palette.py`**
- Thêm trường `desc_en` cho `ObjectType` (red cube / green cylinder / blue sphere).

**`dataset/generator.py`**
- `generate()` giờ sinh thêm `instruction_en` cho mọi level (trước đây chỉ có trong file, không sinh được — regenerate sẽ mất).
- Thêm tham số `n_easy_maps` (mặc định 5): sinh các map đơn-vật (1 vật, 1 màn, không distractor), đánh số nối tiếp sau các map đầy đủ, loại vật luân phiên qua 3 loại cơ bản.
- Tách helper `_instruction_vi`, `_instruction_en`, `_clean_objects`, `_build_single_object`.
- `main()` thêm cờ `--easy-maps` và cập nhật log tóm tắt.

**`dataset/levels.json`** — ghi lại từ generator: 20 level = 5 map đầy đủ (×3 màn) + 5 map dễ (×1 màn).

Kiểm chứng:
- map0–map4 tái tạo **giống hệt** bản cũ (id, objects, instruction_en khớp 0 sai lệch) → các map khó không bị ảnh hưởng.
- map5–map9 mỗi map đúng 1 vật, 1 level, có cả `instruction_vi` và `instruction_en`.

Lệnh tái tạo (chạy trên Mac vì `robot_spec` cần MuJoCo):

```bash
./.venv/bin/python -m dataset.generator --maps 5 --easy-maps 5 --seed 0
```

Lưu ý nhỏ: map dễ giờ luân phiên loại theo `i % 3` (map5=cube_red, map6=cyl_green, map7=sphere_blue, map8=cube_red, map9=cyl_green) — hơi khác bộ tôi thêm tay ở ý 1, nhưng đây là phiên bản chuẩn hoá, reproducible. Nếu muốn mỗi loại xuất hiện đều hơn hoặc nhiều map dễ hơn, chỉ cần đổi `--easy-maps`.

---

**User:**

trong quá trình huấn luyện tôi muốn huấn luyện model trên các màn chơi của các map có 1 object đơn giản trước

---

**Assistant:**

Đây chính là ý 1 — cần cho training ưu tiên các map 1 vật trước. Tôi xem kỹ `episode_manager` và `grpo_train` để biết chỗ chèn logic curriculum.

Tôi hiểu rồi. `_pick_level()` đang chọn ngẫu nhiên đều. Tôi sẽ thêm curriculum vào `EpisodeManager`: xếp level theo độ khó (số vật thể), giai đoạn đầu chỉ lấy map 1 vật, rồi mở dần các tier khó hơn theo tiến trình huấn luyện. Mặc định bật trong `grpo_train`, có cờ tắt.

Trước tiên xem phần cuối `episode_manager.py` để biết `close()` và cấu trúc còn lại:

Giờ triển khai curriculum. Sửa `__init__`:

Now I have the exact text. Let me make the edits:

Giờ cập nhật `_pick_level` và tăng bộ đếm episode ở `_spawn`/`_reset_episode`:

Giờ nối dây vào `grpo_train.py`: thêm cờ CLI và truyền vào `EpisodeManager`.

Giờ kiểm thử logic curriculum mà không cần game service (bỏ qua `_preflight`) bằng cách dựng object thủ công:

Curriculum hoạt động đúng: episode 0–50 chỉ map 1 vật, rồi mở dần map 3, 4, 5 vật, mở hết sau ~200 episode. Kiểm tra lỗi cú pháp/type trên các file đã sửa:

Không có lỗi. Xong.

Đã thêm curriculum "dễ trước, khó sau" cho huấn luyện.

**`train/episode_manager.py`**
- `EpisodeManager` nhận thêm 2 tham số: `curriculum` (bật/tắt) và `curriculum_episodes` (nhịp mở khoá).
- Khi khởi tạo, level được xếp thành các **tier theo độ khó = số vật thể trong map**. Với `levels.json` hiện tại là 4 tier: 1 vật (5 level), 3 vật (6), 4 vật (3), 5 vật (6).
- `_pick_level()` giờ chỉ lấy mẫu trong các tier đã mở khoá. Số tier mở tăng tuyến tính theo số episode đã bắt đầu: khởi đầu chỉ map 1 vật, mở hết sau `curriculum_episodes`. Có log `[curriculum] ...` mỗi khi mở thêm tier.
- Mặc định `curriculum=False` để không phá test/hành vi cũ.

**`train/grpo_train.py`**
- Thêm cờ `--no-curriculum` (mặc định curriculum **bật**) và `--curriculum-episodes` (mặc định 200), truyền vào `EpisodeManager`.

Kiểm chứng bằng mô phỏng: episode 0–50 chỉ ra map 1 vật, episode 100 thêm map 3 vật, 150 thêm 4 vật, ≥200 mở toàn bộ — đúng ý "học map 1 vật trước".

Cách dùng:
```bash
# mặc định đã bật curriculum
python -m train.grpo_train --model Qwen/Qwen2.5-VL-3B-Instruct --game-url http://127.0.0.1:8000

# điều chỉnh nhịp mở khoá (chậm hơn = ở map dễ lâu hơn)
python -m train.grpo_train ... --curriculum-episodes 400

# tắt hẳn (ngẫu nhiên đều như trước)
python -m train.grpo_train ... --no-curriculum
```

Vài điểm cần lưu ý:
- Tiến trình đo bằng **số episode đã bắt đầu**, không phải số training step. Một episode kéo dài tối đa `max_steps` (20) rồi reset. Nếu muốn bám sát số optimizer-step hơn, có thể đổi bộ đếm sang đếm lần `score_and_advance`.
- Khi mở tier khó, các tier dễ **vẫn nằm trong pool** (lấy mẫu đều trên các tier đã mở), nên mô hình không quên hẳn map dễ. Nếu muốn giảm dần tỉ trọng map dễ khi đã mở map khó, mình có thể chuyển sang lấy mẫu có trọng số theo tier — báo mình nếu bạn muốn kiểu đó.

---
