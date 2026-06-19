"""Tiện ích ảnh dùng phía huấn luyện (GPU). Cần Pillow."""
from __future__ import annotations

import base64
import io


def data_url_to_pil(data_url: str):
    """Giải mã data URL base64 PNG -> PIL.Image (RGB)."""
    from PIL import Image

    if "," in data_url:
        _, b64 = data_url.split(",", 1)
    else:
        b64 = data_url
    raw = base64.b64decode(b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")
