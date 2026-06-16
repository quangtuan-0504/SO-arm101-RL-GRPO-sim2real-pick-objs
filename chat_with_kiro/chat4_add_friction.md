# Khi tôi gắp trúng vật thể thì kẹp gắp không thể...

Exported: 6/16/2026, 1:04:11 PM

---

**User:**

Khi tôi gắp trúng vật thể thì kẹp gắp không thể giữ chặt vật thể 

---

**Assistant:**

Tôi sẽ xem qua code để hiểu cách kẹp gắp và mô phỏng vật lý đang hoạt động.

Tôi đã hiểu rõ vấn đề. Đây là vấn đề kinh điển khi mô phỏng kẹp gắp trong MuJoCo. Để tôi giải thích nguyên nhân rồi sửa.

## Nguyên nhân

Vật thể trượt khỏi kẹp không phải do lực kẹp yếu (actuator gripper có `forcerange` ±3.35 N·m với `kp=998` — thừa sức kẹp vật 0.02 kg). Vấn đề nằm ở **mô hình tiếp xúc/ma sát mặc định của MuJoCo**:

1. **`impratio=1` (mặc định)** — đây là thủ phạm chính. Tỉ lệ trở kháng giữa phương pháp tuyến và phương tiếp tuyến quá thấp khiến lực ma sát "mềm", vật trượt ngay cả khi đang bị ép chặt.
2. **`condim=3` (mặc định)** — tiếp xúc chỉ có ma sát trượt, không có ma sát xoắn/lăn, nên vật **xoay/lăn** tuột ra khỏi kẹp (đặc biệt là hình trụ).
3. **Friction cone hình chóp (pyramidal)** — kém chính xác cho việc kẹp so với cone elliptic.
4. **Integrator Euler** — kém ổn định khi có tiếp xúc kẹp giữ.

Các tham số `friction[1]`, `friction[2]` (xoắn/lăn) bạn đặt trong code hiện tại đang **bị bỏ qua** vì `condim=3`.

Tôi sẽ sửa trong `scene_builder.py` (giữ nguyên model gốc như thiết kế).

Giờ cập nhật geom của vật thể để bật ma sát xoắn/lăn (`condim`) và làm tiếp xúc cứng hơn:

Giờ kiểm tra xem model có compile được với các thay đổi không:

Chạy thử vài trăm bước mô phỏng để chắc chắn solver mới ổn định, không bị "nổ":

Mô phỏng ổn định, không có giá trị vô cực, vận tốc trong ngưỡng hợp lý.

## Tóm tắt

Đã sửa trong `sim/scene_builder.py` (model gốc giữ nguyên):

**Thêm `_configure_physics()` — chỉnh solver tiếp xúc:**
- `impratio = 50` — nguyên nhân chính. Làm ma sát "cứng" hơn rất nhiều để vật không trượt khi bị ép.
- `cone = elliptic` — nón ma sát chính xác hơn cho việc kẹp.
- `integrator = implicitfast` — ổn định hơn Euler khi kẹp giữ vật lâu.
- Tăng `iterations`/`ls_iterations` để tiếp xúc hội tụ chắc chắn.

**Cập nhật geom vật thể:**
- `condim = 6` — bật ma sát **xoắn + lăn**, ngăn vật xoay/lăn tuột khỏi kẹp. Trước đó `condim=3` mặc định nên hai thành phần `friction[1]`, `friction[2]` bạn đặt bị bỏ qua hoàn toàn.
- Tăng `friction` lên `[2.0, 0.1, 0.005]`.
- Thêm `solref`/`solimp` cho tiếp xúc cứng hơn để vật không bị ép lún rồi bật ra.

Bạn chạy lại `run.sh` và thử kẹp lại. Nếu vẫn còn hơi trượt với hình trụ, tăng `impratio` lên 100 trong `_configure_physics`. Còn nếu mô phỏng giật/rung khi kẹp, giảm `impratio` xuống ~20 và giảm `solimp[0]` xuống 0.9.

Lưu ý: lực kẹp vốn đã dư (actuator ±3.35 N·m, đủ giữ vật ~0.02 kg), nên vấn đề thực sự là mô hình ma sát chứ không phải lực — vì vậy tôi không đụng tới actuator hay khối lượng.

---
