"""Các hàm chấm điểm cho GRPO.

Thứ tự ưu tiên (theo thiết kế):
  1. format_reward  — completion sai format KHÔNG gửi game; reward của nó được đặt = (nước hợp lệ tệ nhất − margin)
  2. geometric_reward — tín hiệu CHÍNH (PBRS từ toạ độ thật MuJoCo).
  3. judge_reward   — bổ sung bằng VLM self-host (nâng cấp sau khi pipeline ổn).
"""
