Tôi muốn thêm text box để nhập góc quay trong bảng điều khiển, hiển thị thêm khoảng giá trị

--------------------------------------------------

Now I understand the architecture. This is a MuJoCo + GLFW app where the "control panel" is the overlay text on the operator window, and there's no widget toolkit. I'll add a keyboard-driven angle text box and show each joint's value range in degrees.

Let me add helper methods to the simulation first.

Now update the help text in controls.py:

Now the main changes in viewer3.py. First add the `math` import and input-box sizing constants:

Register a char callback so typing flows into the angle input:

Now update `_on_key` to handle the input box keys (and add `_on_char`):

Add the input state to `ThreeViewApp.__init__`:

Now add the input handling methods and update `render_help`. Let me replace the `render_help` method:

Finally, gate the held-key movement so typing an angle doesn't also move the arm:

Let me verify the modules compile cleanly:

Xong rồi. Mình đã thêm hộp nhập góc quay vào bảng điều khiển (cửa sổ Operator) và hiển thị khoảng giá trị cho từng khớp.

Cách dùng:
- Nhấn **T** để bật/tắt chế độ nhập góc.
- Nhấn **TAB** (Shift+TAB để lùi) để chọn khớp muốn đặt.
- Gõ số (có thể kèm dấu `-` và dấu `.`), nhấn **ENTER** để áp dụng, **ESC** để hủy.
- Góc nhập theo **độ**, được kẹp tự động trong giới hạn của khớp.

Phần hiển thị:
- Hộp "NHAP GOC QUAY" ở phía trên cửa sổ Operator: khớp đang chọn, khoảng cho phép, giá trị hiện tại và chuỗi đang gõ. Hộp đổi màu xanh khi đang nhập.
- Bảng "MUC TIEU KHOP / KHOANG (do)" ở góc dưới-phải giờ hiện cả giá trị hiện tại lẫn khoảng `[min .. max]` theo độ, với dấu `>` đánh dấu khớp đang chọn.

Chi tiết thay đổi:
- `sim/simulation.py`: thêm `get_ctrl_range()` và `set_ctrl()` (đặt giá trị tuyệt đối, có kẹp giới hạn).
- `sim/controls.py`: bổ sung hướng dẫn phím T và TAB.
- `sim/viewer3.py`: thêm char callback, xử lý phím cho chế độ nhập, các hàm `input_*`, hộp nhập `_render_input_box`, hiển thị khoảng giá trị, và tạm dừng điều khiển giữ phím khi đang gõ số để cánh tay không bị di chuyển ngoài ý muốn.

Vì đây là giao diện overlay của MuJoCo/GLFW (không có widget thật), "text box" được làm bằng cách bắt phím và vẽ overlay. Bạn chạy thử bằng `python app.py` để kiểm tra trực quan nhé.