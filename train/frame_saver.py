"""Lưu ảnh winner ra đĩa theo lấy mẫu để theo dõi hành vi cánh tay khi train.

Triết lý (xem trao đổi trong chat): KHÔNG lưu mọi ảnh của mọi nhánh — chỉ lưu
ảnh của WINNER đã commit (trạng thái thực sự đi tiếp), lấy mẫu thưa:
  - cứ mỗi ``every`` step, VÀ
  - luôn lưu ở step KẾT THÚC episode (success hoặc fail) — khung đáng xem nhất.

Ảnh đã có sẵn dạng data URL base64 (PNG) trong observation nên việc lưu chỉ là
decode + ghi đĩa, KHÔNG tốn thêm băng thông tunnel. Ảnh được nhóm theo
``level/episode`` để xem thành chuỗi; tên file nhúng step + reward + cờ
grasped/in_basket để khớp với metric JSONL.

Nếu có Pillow: ghép wrist|stand thành 1 ảnh cho gọn. Nếu không: ghi thẳng bytes
PNG đã giải mã thành 2 file (không cần thư viện nào).
"""
from __future__ import annotations

import base64
from pathlib import Path


def _decode_data_url(s: str) -> bytes:
    """Giải mã data URL ('data:image/png;base64,...') hoặc base64 thuần -> bytes."""
    if not s:
        return b""
    if s.strip().lower().startswith("data:") and "," in s:
        s = s.split(",", 1)[1]
    try:
        return base64.b64decode(s)
    except Exception:
        return b""


class FrameSaver:
    def __init__(self, out_dir: str, every: int = 25, combine: bool = True):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.every = max(1, every)
        self.n_saved = 0
        # Thử Pillow một lần để ghép ảnh; thiếu thì fallback ghi 2 file.
        self._Image = None
        if combine:
            try:
                from PIL import Image
                self._Image = Image
            except Exception:
                self._Image = None

    def should_save(self, step_idx: int, finished: bool) -> bool:
        return bool(finished) or (step_idx % self.every == 0)

    def save(self, *, episode_id: str, level_id: str, step_idx: int,
             wrist_b64: str, stand_b64: str, finished: bool = False,
             success: bool = False, meta: dict | None = None) -> str | None:
        """Ghi ảnh winner. Trả về đường dẫn file đã ghi (hoặc None nếu rỗng)."""
        meta = meta or {}
        tag = "succ" if success else ("end" if finished else "ok")
        parts = [f"step{step_idx:03d}", tag]
        wr = meta.get("winner_reward")
        if isinstance(wr, (int, float)):
            parts.append(f"r{wr:.2f}")
        if meta.get("grasped"):
            parts.append("grasp")
        if meta.get("in_basket"):
            parts.append("basket")
        stem = "_".join(parts)

        d = self.out_dir / str(level_id) / str(episode_id)
        d.mkdir(parents=True, exist_ok=True)

        wb = _decode_data_url(wrist_b64)
        sb = _decode_data_url(stand_b64)
        if not wb and not sb:
            return None

        # Đường ghép wrist|stand bằng Pillow (đẹp, 1 file).
        if self._Image is not None and wb and sb:
            try:
                import io
                Image = self._Image
                iw = Image.open(io.BytesIO(wb)).convert("RGB")
                istd = Image.open(io.BytesIO(sb)).convert("RGB")
                h = max(iw.height, istd.height)
                canvas = Image.new("RGB", (iw.width + istd.width, h), (20, 20, 20))
                canvas.paste(iw, (0, 0))
                canvas.paste(istd, (iw.width, 0))
                path = d / f"{stem}.png"
                canvas.save(path)
                self.n_saved += 1
                return str(path)
            except Exception:
                pass  # rơi xuống fallback

        # Fallback không phụ thuộc thư viện: bytes đã là PNG, ghi thẳng.
        path = None
        if wb:
            path = d / f"{stem}_wrist.png"
            path.write_bytes(wb)
        if sb:
            ps = d / f"{stem}_stand.png"
            ps.write_bytes(sb)
            path = path or ps
        self.n_saved += 1
        return str(path) if path else None
