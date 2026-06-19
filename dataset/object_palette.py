"""Bảng các LOẠI vật thể dùng để sinh map.

Mỗi loại có shape/size/màu cố định để VLM phân biệt được bằng thị giác. Vị trí
(``pos``) KHÔNG nằm ở đây — generator sẽ rải vị trí ngẫu nhiên cho từng map.

Giai đoạn đầu (đơn giản): 3 loại, mỗi map có ÍT NHẤT 1 mỗi loại, tối đa 5 vật.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ObjectType:
    type_id: str        # định danh loại (dùng đặt tên & mô tả mục tiêu)
    shape: str          # box | cylinder | sphere
    size: list          # theo quy ước scene_builder
    rgba: list
    mass: float
    desc_vi: str        # mô tả tiếng Việt cho instruction/prompt
    rest_z: float       # z lúc đặt trên sàn (nửa chiều cao vật)


# 3 loại cơ bản, màu tương phản rõ với sàn và với nhau.
OBJECT_TYPES: list[ObjectType] = [
    ObjectType(
        type_id="cube_red", shape="box",
        size=[0.018, 0.018, 0.018], rgba=[0.9, 0.2, 0.2, 1.0], mass=0.02,
        desc_vi="khối lập phương màu đỏ", rest_z=0.018,
    ),
    ObjectType(
        type_id="cyl_green", shape="cylinder",
        size=[0.016, 0.022], rgba=[0.2, 0.8, 0.35, 1.0], mass=0.02,
        desc_vi="khối trụ màu xanh lá", rest_z=0.022,
    ),
    ObjectType(
        type_id="sphere_blue", shape="sphere",
        size=[0.018], rgba=[0.2, 0.45, 0.95, 1.0], mass=0.02,
        desc_vi="quả cầu màu xanh dương", rest_z=0.018,
    ),
]

TYPES_BY_ID = {t.type_id: t for t in OBJECT_TYPES}
