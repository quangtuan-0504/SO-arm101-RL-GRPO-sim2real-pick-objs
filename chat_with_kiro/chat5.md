 tôi muốn chỉnh góc nhìn của stand_camera thành nhìn theo phương ngang, hướng từ trái qua phải, trực diện vào cánh tay

 ------------------------------------------

 Tôi đã xem cách `stand_camera` hoạt động. Hệ toạ độ: đế cánh tay ở gốc `[0,0,0]`, cánh tay vươn về phía `+x`, trục `y` là phương ngang trái–phải, `z` là chiều cao.

Để camera nhìn **theo phương ngang** (cùng độ cao với điểm nhìn), **từ trái qua phải** (đặt ở phía `+y` nhìn về `-y`), và **trực diện vào cánh tay**, tôi chỉnh `pos` và `look_at` cùng độ cao `z`:

Đã chỉnh xong. Thay đổi chính:

- `pos: [0.10, 0.55, 0.12]` — camera đặt ở phía bên trái cánh tay (trục `+y`), độ cao `z = 0.12 m`.
- `look_at: [0.10, 0.0, 0.12]` — nhìn về tâm cánh tay, **cùng độ cao `z`** nên hướng nhìn song song mặt sàn (phương ngang), đi từ trái (`+y`) qua phải (`-y`) và trực diện vào cánh tay.

Vài tham số dễ tinh chỉnh nếu cần:
- Muốn camera xa/gần hơn: tăng/giảm giá trị `y` trong `pos` (ví dụ `0.7` để lùi ra xa).
- Muốn ngắm cao/thấp hơn nhưng vẫn ngang: đổi `z` ở **cả hai** dòng `pos` và `look_at` cho bằng nhau.
- Nếu muốn nhìn từ phải qua trái thay vì trái qua phải, đặt `y` của `pos` thành số âm (ví dụ `-0.55`).