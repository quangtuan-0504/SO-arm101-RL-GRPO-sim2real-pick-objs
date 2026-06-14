Tôi muốn tách góc nhìn của mỗi camera ra thành một cửa sổ hiển thị riêng và trang bị thêm nút chụp màn hình cho cửa số đó
-------------------------------------------------------
Để tôi xem qua codebase trước khi thực hiện thay đổi.

Tôi đã hiểu kiến trúc. Hiện tại `ThreeViewApp` dùng một cửa sổ GLFW duy nhất chia 3 viewport. Tôi sẽ tách thành các cửa sổ riêng (chia sẻ context OpenGL), mỗi cửa sổ render một camera và có nút "Chụp màn hình" có thể bấm bằng chuột (kèm phím tắt P).

Bắt đầu với viewer mới:

Tôi cần dọn lại `_init_render` vì có đoạn tham chiếu `operator_win` trước khi nó tồn tại:

Giờ thêm hàm `apply_held_keys_any` vào controls.py và cập nhật phần trợ giúp:

Cập nhật config.yaml để cho phép tùy chỉnh kích thước từng loại cửa sổ:

Kiểm tra cú pháp và import:

Cập nhật README để phản ánh thay đổi:

Chạy smoke test để chắc chắn thay đổi config không làm hỏng gì (nó dựng cảnh từ config.yaml):

Kiểm tra nhanh tiện ích lưu PNG và slugify hoạt động:

Xong. Tôi đã tách mỗi camera ra cửa sổ riêng và thêm nút chụp màn hình cho từng cửa sổ.

Thay đổi chính:

`sim/viewer3.py` — viết lại:
- Thêm lớp `CameraWindow`, mỗi instance là một cửa sổ GLFW gắn với một camera. Tạo 3 cửa sổ: Operator (camera tự do), Camera 1 (cổ tay), Camera 2 (giá đỡ). Các cửa sổ chia sẻ chung OpenGL context nên dùng chung một `MjrContext` (cửa sổ Operator là context chủ).
- Mỗi cửa sổ vẽ một nút "Chup man hinh" ở góc trên-phải (dùng `mjr_rectangle` + overlay text). Bấm chuột trái vào nút hoặc nhấn `P` sẽ chụp nội dung khung hiện tại (đọc bằng `mjr_readPixels`, lật dọc) và lưu PNG vào `screenshots/` với tên `<ten_cua_so>_<timestamp>.png`.
- Ảnh được chụp trước khi vẽ nút nên không dính nút vào ảnh.
- Chỉ cửa sổ Operator nhận thao tác chuột xoay/zoom; hai cửa sổ camera cố định.
- Vòng lặp chạy đến khi đóng cửa sổ Operator; đóng một cửa sổ camera chỉ ngừng vẽ cửa sổ đó.

`sim/controls.py` — thêm `apply_held_keys_any(windows, ...)` để phím điều khiển cánh tay hoạt động ở bất kỳ cửa sổ nào đang focus (mỗi khớp chỉ cập nhật một lần), thêm phím `P` vào phần trợ giúp.

`config.yaml` — `window` giờ có `operator_width/height` và `camera_width/height` riêng.

`README.md`, `app.py`, `.gitignore` — cập nhật mô tả và bỏ qua thư mục `screenshots/`.

Đã xác minh: cú pháp, import (glfw/mujoco), nạp config qua smoke_test, và logic lưu PNG/slugify đều chạy tốt. Phần GUI nhiều cửa sổ cần màn hình thật để chạy nên không test headless được ở đây, nhưng kiến trúc theo đúng mẫu shared-context chuẩn của MuJoCo+GLFW.

Chạy thử: `./.venv/bin/python app.py` — bạn sẽ thấy 3 cửa sổ tách rời, mỗi cửa sổ có nút chụp ở góc phải trên.