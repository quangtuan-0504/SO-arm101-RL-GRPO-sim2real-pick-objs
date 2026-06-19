"""Huấn luyện GRPO bằng trl — CHẠY TRÊN SERVER GPU (Vast AI). CHƯA test trên Mac.

Ánh xạ pipeline vào trl (single-step GRPO, theo "Cách A"):
    prompt        = 2 ảnh trạng thái i + instruction + bảng khớp (1 episode/step)
    num_generations = G (số lệnh VLM sinh cho cùng 1 trạng thái)
    reward        = env_reward_func: format gate (0/1) + geometric (PBRS)
                    -> branch G lệnh trong game -> chấm -> commit winner (advance)

Kiến trúc chạy:
    [GPU] vLLM sinh G completions  ──►  env_reward_func
                                          │  (HTTP)
                                          ▼
    [Mac] game service  ──  branch G nhánh, trả ảnh+số đo, commit winner

────────────────────────────────────────────────────────────────────────────
RÀNG BUỘC THỨ TỰ (quan trọng, vì môi trường advance DỰA TRÊN completion):
  - Mỗi batch chứa ĐÚNG ``n_parallel`` prompt PHÂN BIỆT (mỗi episode 1 prompt),
    mỗi prompt sinh G completions. => per_device_train_batch_size * world_size
    * grad_accum  ==  n_parallel.
  - ``steps_per_generation`` = 1 và KHÔNG prefetch/shuffle để dataset chỉ lấy
    prompt step kế TIẾP SAU KHI reward đã commit winner của step hiện tại.
  - dataloader_num_workers = 0.
Nếu phá các ràng buộc này, dataset có thể lấy trạng thái cũ -> reward lệch.
────────────────────────────────────────────────────────────────────────────

Cài trên GPU server:
    pip install "trl>=0.27" "transformers>=4.56" accelerate "vllm>=0.11" pillow
Sinh dataset (chạy 1 lần, có thể chạy trên Mac rồi copy sang):
    python -m dataset.generator --maps 5 --seed 0
Chạy game service trên Mac và mở cổng cho server GPU truy cập (ví dụ qua SSH
reverse tunnel): GAME_URL=http://<mac-host>:8000

    python -m train.grpo_train --model Qwen/Qwen2.5-VL-3B-Instruct \
        --game-url http://127.0.0.1:8000 --n-parallel 2 --group-size 4
"""
from __future__ import annotations

import argparse
import json
import os
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _extract_text(completion) -> str:
    """Lấy text từ một completion (trl có thể trả str hoặc list message dict)."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):  # dạng conversational [{role, content}]
        last = completion[-1]
        c = last.get("content", "")
        if isinstance(c, list):  # content nhiều phần
            return " ".join(p.get("text", "") for p in c if isinstance(p, dict))
        return c
    return str(completion)


def build_reward_func(manager):
    """Tạo reward function cho trl, đóng quanh EpisodeManager.

    trl gọi: reward_func(prompts, completions, **kwargs). kwargs chứa các cột
    dataset (ở đây có ``episode_id``), lặp lại theo từng completion. Ta gom theo
    episode_id (mỗi episode = 1 nhóm G completion liền nhau trong batch), chấm
    bằng manager.score_and_advance (đã commit winner để advance môi trường), rồi
    trả reward phẳng đúng thứ tự đầu vào.
    """
    def env_reward_func(prompts=None, completions=None, episode_id=None, **kwargs):
        n = len(completions)
        texts = [_extract_text(c) for c in completions]

        # Gom các vị trí theo episode_id (giữ nguyên thứ tự xuất hiện).
        groups: "OrderedDict[str, list[int]]" = OrderedDict()
        for i, eid in enumerate(episode_id):
            groups.setdefault(eid, []).append(i)

        rewards = [0.0] * n
        for eid, idxs in groups.items():
            comp = [texts[i] for i in idxs]
            res = manager.score_and_advance(eid, comp)
            for local, gi in enumerate(idxs):
                rewards[gi] = res.rewards[local]
            # Log gọn (xuất hiện trong stdout; có thể đẩy sang trackio/wandb).
            print(f"[reward] {eid} lvl={res.info['level_id']} "
                  f"step={res.info['step_idx']} n_valid={res.info['n_valid']} "
                  f"r={[round(x, 3) for x in res.rewards]} "
                  f"finished={res.finished}")
        return rewards

    env_reward_func.__name__ = "env_reward"
    return env_reward_func


def make_dataset(manager, total_samples: int):
    """IterableDataset động: round-robin các episode, lấy prompt step HIỆN TẠI.

    Vì __iter__ đọc ``manager.current_prompt`` tại thời điểm lấy mẫu, prompt luôn
    phản ánh trạng thái mới nhất (sau khi reward func đã advance ở batch trước).
    """
    from datasets import IterableDataset

    eids = manager.episode_ids

    def gen():
        produced = 0
        while produced < total_samples:
            for eid in eids:
                yield {"prompt": manager.current_prompt(eid), "episode_id": eid}
                produced += 1
                if produced >= total_samples:
                    break

    return IterableDataset.from_generator(gen)


def main():
    p = argparse.ArgumentParser(description="GRPO train SO-101 VLM (GPU)")
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--game-url", default=os.environ.get("GAME_URL",
                                                        "http://127.0.0.1:8000"))
    p.add_argument("--n-parallel", type=int, default=2)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--total-samples", type=int, default=2000)
    p.add_argument("--output-dir", default=str(ROOT / "checkpoints" / "grpo"))
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--use-vllm", action="store_true")
    args = p.parse_args()

    # Import nặng đặt trong main để Mac không cần cài torch/trl.
    from trl import GRPOConfig, GRPOTrainer
    from train.episode_manager import EpisodeManager
    from dataset.generator import load_levels

    levels = load_levels(ROOT / "dataset" / "levels.json")
    joints = json.loads(
        (ROOT / "dataset" / "robot_spec.json").read_text())["joints"]

    manager = EpisodeManager(
        args.game_url, levels, joints,
        n_parallel=args.n_parallel, group_size=args.group_size)

    reward_func = build_reward_func(manager)
    dataset = make_dataset(manager, args.total_samples)

    # Ràng buộc: 1 batch = n_parallel prompt phân biệt, mỗi prompt G completion.
    config = GRPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.n_parallel,
        gradient_accumulation_steps=1,
        num_generations=args.group_size,
        steps_per_generation=1,
        learning_rate=args.lr,
        max_prompt_length=None,      # ảnh + prompt dài; để None tránh cắt
        max_completion_length=512,
        temperature=1.0,
        beta=0.0,                    # khởi đầu tắt KL cho đơn giản; bật sau nếu lệch
        use_vllm=args.use_vllm,
        vllm_mode="colocate" if args.use_vllm else None,
        log_completions=True,
        logging_steps=1,
        save_steps=200,
        dataloader_num_workers=0,    # giữ thứ tự lấy mẫu xác định
        shuffle_dataset=False,
    )

    trainer = GRPOTrainer(
        model=args.model,
        args=config,
        train_dataset=dataset,
        reward_funcs=[reward_func],
    )
    try:
        trainer.train()
    finally:
        manager.close()


if __name__ == "__main__":
    main()
