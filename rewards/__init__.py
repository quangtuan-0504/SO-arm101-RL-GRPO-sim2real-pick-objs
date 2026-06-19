"""Các hàm chấm điểm cho GRPO.

Thứ tự ưu tiên (theo thiết kế):
  1. format_reward  — gate 0/1. =0 thì bỏ qua, KHÔNG gửi game, KHÔNG chấm tiếp.
  2. geometric_reward — tín hiệu CHÍNH (PBRS từ toạ độ thật MuJoCo).
  3. judge_reward   — bổ sung bằng VLM self-host (nâng cấp sau khi pipeline ổn).
"""
