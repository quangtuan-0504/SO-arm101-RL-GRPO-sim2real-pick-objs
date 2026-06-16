"""Render ảnh camera offscreen và encode thành data URL base64 (PNG).

Không phụ thuộc thư viện ngoài: dùng mujoco.Renderer để render và tự encode
PNG bằng zlib/struct (giống cách smoke_test.py / viewer3.py đang làm).
"""
from __future__ import annotations

import base64
import struct
import zlib
from pathlib import Path

import numpy as np
import mujoco


class CameraRenderer:
    """Bọc mujoco.Renderer để render nhiều camera theo tên.

    Tạo một lần rồi tái sử dụng trong vòng lặp điều khiển (rẻ hơn tạo mới
    mỗi bước).
    """

    def __init__(self, model, width: int = 480, height: int = 360):
        self.model = model
        self.width = width
        self.height = height
        self._renderer = mujoco.Renderer(model, height, width)

    def render(self, data, cam_name: str) -> np.ndarray:
        """Trả về ảnh RGB (H, W, 3) uint8 từ camera tên ``cam_name``."""
        self._renderer.update_scene(data, camera=cam_name)
        return self._renderer.render()

    def close(self):
        # mujoco.Renderer giải phóng context khi bị thu hồi; gọi tường minh.
        try:
            self._renderer.close()
        except Exception:
            pass


def rgb_to_png_bytes(rgb: np.ndarray) -> bytes:
    """Encode mảng RGB (H, W, 3) uint8 -> bytes PNG, không cần PIL."""
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w, _ = rgb.shape
    raw = bytearray()
    for y in range(h):
        raw.append(0)  # filter type 0 cho mỗi hàng
        raw.extend(rgb[y].tobytes())

    def chunk(typ: bytes, body: bytes) -> bytes:
        c = typ + body
        return (struct.pack(">I", len(body)) + c
                + struct.pack(">I", zlib.crc32(c) & 0xffffffff))

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(bytes(raw), 6))
    png += chunk(b"IEND", b"")
    return bytes(png)


def rgb_to_data_url(rgb: np.ndarray) -> str:
    """Encode RGB -> data URL base64 dùng trực tiếp trong message image_url."""
    b64 = base64.b64encode(rgb_to_png_bytes(rgb)).decode("ascii")
    return f"data:image/png;base64,{b64}"


def save_png(path, rgb: np.ndarray):
    """Lưu ảnh RGB ra file PNG (tiện debug từng bước)."""
    Path(path).write_bytes(rgb_to_png_bytes(rgb))
