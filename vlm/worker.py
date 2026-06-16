"""Worker chạy suy luận VLM ở thread nền, tách khỏi vòng lặp GLFW.

Vì trên macOS GLFW + render phải ở MAIN thread, ta KHÔNG render trong thread
nền. Main thread render 2 ảnh (numpy RGB) rồi đẩy sang worker; worker chỉ làm
việc thuần CPU/mạng: encode base64 -> build messages -> gọi Qwen -> parse JSON.
Kết quả trả về qua hàng đợi để main thread áp dụng (set_ctrl).
"""
from __future__ import annotations

import queue
import threading

from vlm.prompt_builder import build_messages
from vlm.render_utils import rgb_to_data_url


class VLMWorker:
    """Một luồng nền nhận cặp ảnh -> trả dict dự đoán."""

    def __init__(self, sim, target_desc: str, client):
        self.sim = sim
        self.target_desc = target_desc
        self.client = client

        self._in: queue.Queue = queue.Queue(maxsize=1)
        self._out: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

        self.busy = False     # có một yêu cầu đang xử lý hay không
        self.done = False     # model đã báo hoàn thành
        self.last_pred = None

    # -- vòng đời -------------------------------------------------------
    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()

    # -- giao tiếp với main thread -------------------------------------
    def submit(self, wrist_rgb, stand_rgb) -> bool:
        """Gửi cặp ảnh mới để suy luận. Trả False nếu worker đang bận."""
        if self.busy:
            return False
        try:
            self._in.put_nowait((wrist_rgb, stand_rgb))
            self.busy = True
            return True
        except queue.Full:
            return False

    def poll(self):
        """Lấy kết quả mới nếu có (không chặn). Trả dict hoặc None."""
        try:
            pred = self._out.get_nowait()
        except queue.Empty:
            return None
        self.busy = False
        self.last_pred = pred
        if isinstance(pred, dict) and pred.get("done"):
            self.done = True
        return pred

    # -- thân luồng nền -------------------------------------------------
    def _run(self):
        while not self._stop.is_set():
            try:
                wrist_rgb, stand_rgb = self._in.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                messages = build_messages(
                    self.sim, self.target_desc,
                    rgb_to_data_url(wrist_rgb), rgb_to_data_url(stand_rgb))
                pred = self.client.predict(messages)
            except Exception as e:  # không để worker chết
                pred = {"error": str(e)}
            self._out.put(pred)
