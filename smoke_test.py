"""Kiểm thử headless (không cần cửa sổ): dựng cảnh từ config, chạy mô phỏng
một lúc, render cả 3 khung nhìn offscreen và ghép thành 1 ảnh PNG để kiểm tra.

Chạy: python smoke_test.py
Kết quả: smoke_output.png
"""
from pathlib import Path
import numpy as np
import mujoco

from app import load_config
from sim.simulation import Simulation

ROOT = Path(__file__).resolve().parent


def label_ok(name, cond):
    print(f"  [{'OK' if cond else 'FAIL'}] {name}")
    return cond


def main():
    config = load_config(str(ROOT / "config.yaml"))
    sim = Simulation(config)
    m, d = sim.model, sim.data

    print("Kiem tra model:")
    cams = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, i)
            for i in range(m.ncam)]
    ok = True
    ok &= label_ok("co camera wrist_cam", "wrist_cam" in cams)
    ok &= label_ok("co camera stand_cam", "stand_cam" in cams)
    ok &= label_ok("du 6 actuator", m.nu == 6)
    n_obj = len(config.get("objects", []))
    ok &= label_ok(f"co {n_obj} vat the (freejoint)", m.njnt >= 6 + n_obj)

    # Chạy mô phỏng ~2 giây cho ổn định
    steps = int(2.0 / m.opt.timestep)
    sim.step(steps)
    ok &= label_ok("mo phong on dinh (khong NaN)",
                   bool(np.all(np.isfinite(d.qpos))))

    # Render 3 khung nhìn offscreen
    h, w = 360, 480
    renderer = mujoco.Renderer(m, h, w)

    # Operator: camera tự do
    op_cam = mujoco.MjvCamera()
    op_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    oc = config.get("operator_camera", {})
    op_cam.azimuth = oc.get("azimuth", 135)
    op_cam.elevation = oc.get("elevation", -25)
    op_cam.distance = oc.get("distance", 0.9)
    op_cam.lookat = np.array(oc.get("look_at", [0.12, 0, 0.12]))
    renderer.update_scene(d, camera=op_cam)
    img_op = renderer.render()

    renderer.update_scene(d, camera="wrist_cam")
    img_w = renderer.render()
    renderer.update_scene(d, camera="stand_cam")
    img_s = renderer.render()

    for nm, im in [("operator", img_op), ("wrist", img_w), ("stand", img_s)]:
        ok &= label_ok(f"khung {nm} co noi dung",
                       float(im.std()) > 5.0)

    # Ghép ngang 3 ảnh và lưu
    composite = np.concatenate([img_op, img_w, img_s], axis=1)
    out = ROOT / "smoke_output.png"
    _save_png(out, composite)
    print(f"\nDa luu anh ghep 3 khung nhin: {out}")
    print("KET QUA:", "TAT CA OK" if ok else "CO LOI - xem o tren")
    return 0 if ok else 1


def _save_png(path, rgb):
    try:
        from PIL import Image
        Image.fromarray(rgb).save(path)
    except ImportError:
        # Ghi PNG thủ công bằng zlib + struct (không cần thư viện ngoài)
        import struct, zlib
        h, w, _ = rgb.shape
        raw = bytearray()
        for y in range(h):
            raw.append(0)
            raw.extend(rgb[y].tobytes())
        def chunk(typ, data):
            c = typ + data
            return struct.pack(">I", len(data)) + c + struct.pack(
                ">I", zlib.crc32(c) & 0xffffffff)
        png = b"\x89PNG\r\n\x1a\n"
        png += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        png += chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        png += chunk(b"IEND", b"")
        path.write_bytes(png)


if __name__ == "__main__":
    raise SystemExit(main())
