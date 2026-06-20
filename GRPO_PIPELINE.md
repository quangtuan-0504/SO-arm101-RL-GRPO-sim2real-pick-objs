# Pipeline huấn luyện VLM điều khiển SO-101 bằng GRPO

Tài liệu này mô tả pipeline đã dựng theo thiết kế trong `chat_with_kiro/chat12`.
Mục tiêu: VLM học gắp-và-thả qua tương tác môi trường, dùng `trl` chạy GRPO.

## Kiến trúc 2 máy

```
┌─ Mac mini (KHÔNG GPU) ───────────┐        ┌─ Vast AI (GPU) ─────────────────────┐
│ GAME SERVICE  (game/service.py)  │        │ TRAINING  (train/grpo_train.py)     │
│  FastAPI: /reset /branch /commit │◄─HTTP──┤  trl GRPOTrainer + vLLM             │
│  - dựng map từ level table       │        │  - EpisodeManager (pool màn song song)│
│  - snapshot/restore -> branch G  │        │  - dataset động: prompt = step hiện tại│
│  - render 2 cam, trả ảnh + số đo │        │  - reward = format gate + geometric  │
│  - KHÔNG build prompt, KHÔNG chấm│        │  - (tuỳ chọn) judge VLM self-host    │
└──────────────────────────────────┘        └──────────────────────────────────────┘
```

Game "dumb": nhận góc → step → chụp ảnh → trả ảnh + số đo. Mọi quyết
định (prompt, reward, chọn winner) nằm ở server huấn luyện.

## Đã chốt (theo trao đổi)
- **Camera CỐ ĐỊNH** mọi map (giai đoạn đầu, reward ổn định).
- **Mỗi màn 1 loại vật** (đơn giản trước).
- Build prompt **trên server huấn luyện**, game chỉ gửi ảnh + metadata.
- 3 hàm reward: `format` (gate 0/1) → `geometric` (PBRS, **chính**) → `judge` VLM (**nâng cấp sau**).

## Thành phần & trạng thái test

| Phần | File | Chạy trên | Test |
|------|------|-----------|------|
| Snapshot/restore + accessor hình học | `sim/simulation.py` | Mac | `tests/test_metrics_snapshot.py` ✅ |
| Đo hình học + PBRS potential | `sim/metrics.py` | Mac | ✅ |
| Sinh map + level table (15 level) | `dataset/generator.py` | Mac | sinh `dataset/levels.json`, `robot_spec.json` ✅ |
| Môi trường 1 màn (reset/branch/commit) | `game/env.py` | Mac | `tests/test_env_rewards.py` ✅ |
| Game service HTTP | `game/service.py` | Mac | `tests/test_game_service.py` ✅ |
| Format reward (gate) | `rewards/format_reward.py` | Mac | ✅ |
| Geometric reward (PBRS) | `rewards/geometric_reward.py` | Mac | ✅ |
| HTTP client tới game | `train/env_client.py` | cả hai | ✅ |
| Rollout 1 màn | `train/rollout.py` | Mac (MockPolicy) | `tests/test_rollout.py` ✅ |
| Pool màn song song | `train/episode_manager.py` | Mac (MockPolicy) | `tests/test_episode_manager.py` ✅ |
| Policy VLM thật | `train/policy.py` | **GPU** | chưa |
| trl GRPOTrainer | `train/grpo_train.py` | **GPU** | chưa |
| Judge VLM (reward #3) | `rewards/judge_reward.py` | **GPU** (prompt/parser test Mac ✅) | một phần |

## Cách chạy

### 1. Sinh dataset (Mac, 1 lần)
```bash
./.venv/bin/python -m dataset.generator --maps 5 --seed 0
```
Tạo `dataset/levels.json` (15 level) + `dataset/robot_spec.json` (giới hạn khớp).

### 2. Chạy game service (Mac)
```bash
# Cổng mặc định 8200 (TRÁNH 8000 vì VS Code/Kiro hay chiếm 127.0.0.1:8000).
# caffeinate giữ Mac không ngủ -> tiến trình không bị đóng băng khi train lâu.
caffeinate -dimsu ./.venv/bin/python -m game.service --port 8200
```
Cho server GPU truy cập qua SSH reverse tunnel. CHẠY LỆNH TRÊN MAC, SSH RA
server GPU (KHÔNG ssh vào Mac). Dùng host+port mà Vast AI cấp ở nút "Connect":
```bash
# trên MAC: đẩy ngược cổng 8200 của Mac sang 127.0.0.1:8200 của server GPU
ssh -N -R 8200:localhost:8200 -p <VAST_PORT> root@<VAST_HOST>
# ví dụ (Direct): ssh -N -R 8200:localhost:8200 -p 41875 root@79.116.54.219
```
Kiểm tra trên server GPU: `curl http://127.0.0.1:8200/health`
→ `{"status":"ok","n_levels":15,...}`.
(Không cần bật Remote Login/sshd trên Mac — Mac là phía đi RA, không nhận vào.)

> Vì sao KHÔNG dùng 8000: nếu cả VS Code/Kiro (gắn `127.0.0.1:8000`) và game
> service (gắn `*:8000`) cùng nghe 8000, request tới `127.0.0.1:8000` lọt vào
> VS Code và treo tới timeout. Dùng cổng riêng (8200) để tránh hẳn.

### 3. Huấn luyện GRPO (server GPU) — CHƯA chạy được trên Mac
```bash
pip install "trl>=0.27" "transformers>=4.56" accelerate "vllm>=0.11" pillow
python -m train.grpo_train \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --game-url http://127.0.0.1:8200 \
    --n-parallel 2 --group-size 4 --use-vllm
```

## Ánh xạ vào GRPO (single-step, "Cách A")
- 1 **prompt** = 2 ảnh trạng thái i + instruction + bảng khớp.
- `num_generations = G` = số lệnh VLM sinh cho cùng trạng thái i.
- `reward_func` branch G lệnh trong game → chấm `format + geometric` → commit
  winner (advance môi trường sang i+1).
- "Chơi 2 màn song song" = batch gồm `n_parallel` prompt phân biệt.

### Ràng buộc thứ tự (quan trọng)
Môi trường advance DỰA TRÊN completion, nên:
- 1 batch = đúng `n_parallel` prompt phân biệt, mỗi prompt G completion
  ⇒ `per_device_train_batch_size * world_size * grad_accum == n_parallel`.
- `steps_per_generation=1`, `shuffle_dataset=False`, `dataloader_num_workers=0`
  để dataset chỉ lấy prompt step kế **sau khi** reward đã commit winner.

Các ràng buộc này đã set sẵn trong `train/grpo_train.py`. Đây là điểm dễ sai và
**cần kiểm chứng thực tế trên GPU** vì phụ thuộc hành vi batching của phiên bản `trl`.

## Reward (chi tiết)
1. **format** (`rewards/format_reward.py`): 1.0 nếu JSON đủ khoá/đúng enum/đủ 6
   khớp số; 0.0 nếu sai. Completion 0 điểm KHÔNG gửi game, KHÔNG chấm tiếp.
2. **geometric** (`rewards/geometric_reward.py`): PBRS `Φ(s')−Φ(s)` + mốc rời rạc
   (`grasped`, `in_basket`) + phạt đập vật. `Φ` chuyển approach→carry theo
   `grasped` (đo khách quan từ MuJoCo), KHÔNG theo nhãn pha model tự khai → chống
   gian lận. Đây là tín hiệu CHÍNH.
3. **judge** (`rewards/judge_reward.py`, tuỳ chọn): VLM self-host chấm 0–100 dựa
   trên ảnh i và các ảnh i+1, yêu cầu chấm phân biệt rõ (tránh reward-collapse).

## Bước tiếp theo gợi ý
1. Trên GPU: chạy `train/rollout.py` với `VLLMPolicy` thật (thay MockPolicy) để
   sanity-check sinh completion + branch + reward TRƯỚC khi bật GRPOTrainer.
2. Bật GRPO với `format + geometric` cho tới khi hội tụ (rẻ, ổn định, dễ debug).
3. Khi ổn, layer thêm judge VLM (reward #3) qua một reward_func thứ hai.
4. Giai đoạn 2: domain randomization camera, nhiệm vụ đa vật.
```
