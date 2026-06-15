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
----------------------------------------------
Tôi muốn nhập tất cả các giá trị góc một lần sau đó bấm hoạt động thì cánh tay mới quay theo chỉ định
-----------------------------
The user wants to type targets for all joints first, then press an "activate" button/key so the arm moves to all specified angles at once. Let me redesign the input box to hold a pending target per joint plus an "Hoạt động" button.

Let me view the current state of the relevant sections:

Now let me refactor. First, generalize the cursor hit-test and add an activate button rect to `CameraWindow`:

Now handle a click on the activate button in `_on_mouse_button`:

Now update the key handling. ENTER should confirm the current field and move on, and a new key (G) activates all:

Now update the app state. Replace the simple buffer state with a per-joint pending list:

Now rewrite the input methods and the rendering to support the pending-all-then-activate workflow:

Update the input box dimensions and wire the activate button into the operator render:

Now update the help text in controls.py to reflect the new workflow:

Đã đổi luồng hoạt động đúng như bạn muốn: nhập sẵn góc cho tất cả khớp trước, rồi bấm **Hoạt động** thì cánh tay mới quay.

Cách dùng mới:
- Nhấn **T** để mở bảng nhập. Bảng sẽ hiện sẵn 6 khớp với góc hiện tại (theo độ) và khoảng cho phép `[min..max]`.
- **TAB** (hoặc **ENTER**) chuyển sang khớp kế tiếp; Shift+TAB lùi lại. Khớp đang chọn có dấu `>` và con trỏ `_`.
- Gõ số cho từng khớp (hỗ trợ `-` và `.`), **Backspace** để xóa. Giá trị được giữ lại cho mỗi khớp, không áp dụng ngay.
- Khi đã nhập xong tất cả, bấm nút **>> HOAT DONG <<** (màu xanh, ngay dưới bảng nhập) hoặc nhấn **G**. Lúc này cánh tay mới quay tới toàn bộ góc đã chỉ định cùng lúc.
- **ESC** để hủy/đóng bảng nhập.

Các giá trị được kẹp tự động trong giới hạn của từng khớp, và nhập theo độ (chuyển sang radian khi áp dụng).

Thay đổi chính:
- `viewer3.py`: thay buffer đơn lẻ bằng `pending_deg` (mảng góc mục tiêu cho từng khớp); thêm `input_confirm_field`, `input_activate`, nút "Hoạt động" có thể bấm chuột (`render_activate_button`) và phím tắt G; bảng nhập giờ liệt kê cả 6 khớp cùng khoảng giá trị.
- `controls.py`: cập nhật phần trợ giúp thêm phím G.

Đây là overlay MuJoCo/GLFW nên mình chưa chạy thử trực quan được; bạn chạy `python app.py` để kiểm tra giao diện và thao tác nhé.