"""Service huấn luyện (chạy trên server GPU Vast AI).

CHÚ Ý: các module gọi model/trl/vLLM cần GPU và CHƯA chạy được trên Mac.
Riêng ``env_client`` chỉ là HTTP client tới game service nên test được ngay.
"""
