# Tôi muốn dùng VLM để dự đoán các góc quay để đi...

Exported: 6/16/2026, 1:05:48 PM

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
