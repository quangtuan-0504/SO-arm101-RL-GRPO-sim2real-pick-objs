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
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class _Tee:
    """Ghi đồng thời ra terminal và file log (để theo dõi + lưu lại).

    Bọc quanh sys.stdout/sys.stderr: mọi thứ in ra màn hình vẫn hiện như cũ,
    đồng thời được chép vào file log để xem lại sau khi chạy xong.
    """

    def __init__(self, stream, log_fp):
        self._stream = stream
        self._log = log_fp

    def write(self, data):
        self._stream.write(data)
        self._log.write(data)
        self._log.flush()

    def flush(self):
        self._stream.flush()
        self._log.flush()

    def __getattr__(self, name):  # ủy quyền isatty()/encoding... cho stream gốc
        return getattr(self._stream, name)


def _setup_logfile(log_file: str | None) -> str:
    """Mở file log và chuyển hướng stdout/stderr qua _Tee. Trả về đường dẫn."""
    if log_file is None:
        log_dir = ROOT / "logs"
        log_dir.mkdir(exist_ok=True)
        log_file = str(log_dir / f"grpo_{datetime.now():%Y%m%d_%H%M%S}.log")
    else:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    fp = open(log_file, "a", encoding="utf-8", buffering=1)
    fp.write(f"\n===== GRPO run @ {datetime.now():%Y-%m-%d %H:%M:%S} =====\n")
    sys.stdout = _Tee(sys.stdout, fp)
    sys.stderr = _Tee(sys.stderr, fp)
    return log_file


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


def build_reward_func(manager, metrics=None, frame_saver=None):
    """Tạo reward function cho trl, đóng quanh EpisodeManager.

    trl gọi: reward_func(prompts, completions, **kwargs). kwargs chứa các cột
    dataset (ở đây có ``episode_id``), lặp lại theo từng completion. Ta gom theo
    episode_id (mỗi episode = 1 nhóm G completion liền nhau trong batch), chấm
    bằng manager.score_and_advance (đã commit winner để advance môi trường), rồi
    trả reward phẳng đúng thứ tự đầu vào.

    Nếu truyền ``metrics`` (MetricsLogger), mỗi step được ghi 1 dòng JSONL để
    trực quan hoá bằng train/plot_metrics.py.
    Nếu truyền ``frame_saver`` (FrameSaver), ảnh winner được lưu đĩa theo lấy mẫu
    (mỗi N step + cuối episode) để theo dõi hành vi cánh tay.
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
            info = res.info
            # Ghi metric (JSONL) để vẽ biểu đồ tiến trình.
            if metrics is not None:
                metrics.log_step(
                    episode_id=eid, level_id=info["level_id"],
                    step_idx=info["step_idx"], group_size=len(idxs),
                    n_valid=info["n_valid"], rewards=res.rewards,
                    geometric_rewards=res.geometric_rewards,
                    format_rewards=res.format_rewards,
                    winner_idx=res.winner_idx, finished=res.finished,
                    success=info.get("success", False),
                    winner_measure=info.get("winner_measure"))
            # Lưu ảnh winner theo lấy mẫu (mỗi N step + cuối episode).
            imgs = info.get("images")
            if (frame_saver is not None and imgs is not None
                    and frame_saver.should_save(info["step_idx"], res.finished)):
                wm = info.get("winner_measure") or {}
                wr = (res.rewards[res.winner_idx]
                      if res.winner_idx is not None else None)
                frame_saver.save(
                    episode_id=eid, level_id=info["level_id"],
                    step_idx=info["step_idx"], wrist_b64=imgs["wrist"],
                    stand_b64=imgs["stand"], finished=res.finished,
                    success=info.get("success", False),
                    meta={"winner_reward": wr, "grasped": wm.get("grasped"),
                          "in_basket": wm.get("in_basket")})
            # Log gọn (xuất hiện trong stdout; có thể đẩy sang trackio/wandb).
            print(f"[reward] {eid} lvl={info['level_id']} "
                  f"step={info['step_idx']} n_valid={info['n_valid']} "
                  f"r={[round(x, 3) for x in res.rewards]} "
                  f"finished={res.finished} success={info.get('success', False)}")
        return rewards

    env_reward_func.__name__ = "env_reward"
    return env_reward_func


def make_dataset(manager, total_samples: int):
    """Dataset map-style + lazy transform: prompt build tại thời điểm TRUY CẬP.

    trl GRPOTrainer không nhận IterableDataset, nên ta dùng datasets.Dataset
    (Arrow) với ``with_transform``. Cột ``prompt`` chỉ là placeholder trong
    Arrow; mỗi lần trl đọc một dòng, transform build prompt LIVE từ trạng thái
    env hiện tại của episode (đã advance ở batch trước). Cột ``episode_id`` được
    trl truyền vào reward function.

    LƯU Ý: cách này dựa vào việc trl đọc dataset LƯỜI (lazy) lúc generate, KHÔNG
    materialize/.map toàn bộ trước. Nếu phiên bản trl materialize dataset, prompt
    sẽ bị "đóng băng" ở trạng thái ban đầu -> phải chuyển sang vòng lặp tuỳ biến
    (xem train/custom_grpo.py).
    """
    from datasets import Dataset

    eids = manager.episode_ids
    n = len(eids)
    episode_col = [eids[i % n] for i in range(total_samples)]

    ds = Dataset.from_dict({
        "prompt": [""] * total_samples,      # placeholder; transform sẽ ghi đè
        "episode_id": episode_col,
    })

    def _live_transform(batch):
        ids = batch["episode_id"]
        return {
            "prompt": [manager.current_prompt(eid) for eid in ids],
            "episode_id": ids,
        }

    ds.set_transform(_live_transform)
    return ds


def _build_config(GRPOConfig, want: dict):
    """Dựng GRPOConfig chỉ với các tham số phiên bản trl hiện tại HỖ TRỢ.

    GRPOConfig là dataclass; các tham số đổi/biến mất giữa các phiên bản trl.
    Lọc theo field thực có để tránh TypeError, in ra cái bị bỏ để bạn biết.
    """
    import dataclasses

    valid = {f.name for f in dataclasses.fields(GRPOConfig)}
    used = {k: v for k, v in want.items() if k in valid}
    dropped = sorted(k for k in want if k not in valid)
    if dropped:
        print(f"[grpo_train] trl phiên bản này KHÔNG có các tham số sau, đã bỏ "
              f"qua (dùng mặc định): {dropped}")
    return GRPOConfig(**used)


def main():
    p = argparse.ArgumentParser(description="GRPO train SO-101 VLM (GPU)")
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--game-url", default=os.environ.get("GAME_URL",
                                                        "http://127.0.0.1:8200"))
    p.add_argument("--n-parallel", type=int, default=2)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--total-samples", type=int, default=2000)
    p.add_argument("--output-dir", default=str(ROOT / "checkpoints" / "grpo"))
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--use-vllm", action="store_true")
    p.add_argument("--game-timeout", type=float, default=180.0,
                   help="Read timeout (giây) cho mỗi request tới game service")
    # --- Curriculum: học từ dễ (map ít vật) lên khó ---
    p.add_argument("--no-curriculum", action="store_true",
                   help="Tắt curriculum (chọn level ngẫu nhiên đều trên mọi map). "
                        "Mặc định BẬT: ưu tiên map 1 vật trước, mở dần map khó.")
    p.add_argument("--curriculum-episodes", type=int, default=200,
                   help="Số episode để curriculum mở dần từ tier dễ nhất tới hết "
                        "tier khó nhất.")
    # --- Tiết kiệm bộ nhớ GPU ---
    p.add_argument("--no-lora", action="store_true",
                   help="Tắt LoRA (full fine-tune, ngốn VRAM hơn nhiều)")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--vllm-gpu-util", type=float, default=0.40,
                   help="Tỉ lệ VRAM cho vLLM trong chế độ colocate (0..1)")
    p.add_argument("--max-model-len", type=int, default=4096,
                   help="Giới hạn ngữ cảnh vLLM (prompt+ảnh+completion) -> bớt KV cache")
    p.add_argument("--no-grad-checkpoint", action="store_true",
                   help="Tắt gradient checkpointing (nhanh hơn nhưng tốn VRAM)")
    p.add_argument("--img-w", type=int, default=384,
                   help="Bề rộng ảnh game render (nhỏ hơn -> ít token thị giác/VRAM)")
    p.add_argument("--img-h", type=int, default=288)
    # --- Log ---
    p.add_argument("--log-file", default=None,
                   help="File ghi log (mặc định logs/grpo_<timestamp>.log). "
                        "Toàn bộ output terminal được chép vào đây để xem lại.")
    p.add_argument("--log-completions", action="store_true",
                   help="In BẢNG prompt+completion đầy đủ mỗi step (RẤT DÀI vì "
                        "prompt to). Mặc định TẮT để log gọn.")
    p.add_argument("--metrics-file", default=None,
                   help="File JSONL ghi metric mỗi step để vẽ biểu đồ "
                        "(mặc định logs/metrics_<timestamp>.jsonl).")
    p.add_argument("--report-to", default="none",
                   help="Backend cho metric trl tích hợp (loss/reward/KL): "
                        "'none' | 'tensorboard' | 'wandb' | 'trackio'.")
    p.add_argument("--save-images", action="store_true",
                   help="Lưu ảnh winner ra đĩa theo lấy mẫu để theo dõi hành vi "
                        "cánh tay. Mặc định TẮT (tiết kiệm đĩa).")
    p.add_argument("--save-img-every", type=int, default=25,
                   help="Lưu ảnh winner mỗi N step (luôn lưu thêm ở cuối episode).")
    p.add_argument("--frames-dir", default=None,
                   help="Thư mục lưu ảnh (mặc định logs/frames_<timestamp>/).")
    args = p.parse_args()

    # Chuyển hướng stdout/stderr ra file log (vẫn hiện trên terminal).
    log_path = _setup_logfile(args.log_file)
    print(f"[grpo_train] ghi log vào: {log_path}")

    # File metric JSONL (mặc định cạnh log, cùng timestamp).
    from train.metrics_logger import MetricsLogger
    metrics_path = args.metrics_file or str(
        ROOT / "logs" / f"metrics_{datetime.now():%Y%m%d_%H%M%S}.jsonl")
    metrics = MetricsLogger(metrics_path)
    print(f"[grpo_train] ghi metric vào: {metrics_path} "
          f"(vẽ biểu đồ: python -m train.plot_metrics {metrics_path})")

    # Bộ lưu ảnh winner (tuỳ chọn).
    frame_saver = None
    if args.save_images:
        from train.frame_saver import FrameSaver
        frames_dir = args.frames_dir or str(
            ROOT / "logs" / f"frames_{datetime.now():%Y%m%d_%H%M%S}")
        frame_saver = FrameSaver(frames_dir, every=args.save_img_every)
        print(f"[grpo_train] lưu ảnh winner vào: {frames_dir} "
              f"(mỗi {args.save_img_every} step + cuối episode)")

    # Import nặng đặt trong main để Mac không cần cài torch/trl.
    from trl import GRPOConfig, GRPOTrainer
    from train.episode_manager import EpisodeManager
    from dataset.generator import load_levels

    levels = load_levels(ROOT / "dataset" / "levels.json")
    joints = json.loads(
        (ROOT / "dataset" / "robot_spec.json").read_text())["joints"]

    manager = EpisodeManager(
        args.game_url, levels, joints,
        n_parallel=args.n_parallel, group_size=args.group_size,
        img_w=args.img_w, img_h=args.img_h,
        timeout=args.game_timeout,
        attach_images=args.save_images,
        curriculum=not args.no_curriculum,
        curriculum_episodes=args.curriculum_episodes)

    reward_func = build_reward_func(manager, metrics=metrics,
                                    frame_saver=frame_saver)
    dataset = make_dataset(manager, args.total_samples)

    # Ràng buộc trl GRPO: per_device_train_batch_size đếm theo COMPLETION.
    #   generation_batch_size = per_device_train_batch_size * grad_accum * world
    #   phải chia hết cho num_generations (G).
    #   số prompt phân biệt mỗi vòng = generation_batch_size / G.
    # Muốn n_parallel prompt phân biệt, mỗi prompt G completion (giả định 1 GPU,
    # grad_accum=1) => per_device_train_batch_size = n_parallel * group_size.
    per_device_bs = args.n_parallel * args.group_size
    n_prompts = per_device_bs // args.group_size
    print(f"[grpo_train] per_device_train_batch_size={per_device_bs} "
          f"(=n_parallel*group_size), num_generations={args.group_size} "
          f"=> {n_prompts} prompt phân biệt/vòng. (Giả định 1 GPU, grad_accum=1; "
          f"nếu nhiều GPU/grad_accum, chỉnh lại để tích chia hết cho G.)")
    want = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=1,
        num_generations=args.group_size,
        steps_per_generation=1,
        learning_rate=args.lr,
        max_prompt_length=None,      # ảnh + prompt dài; để None tránh cắt
        max_completion_length=512,
        temperature=1,             # cao hơn 1.0 -> G completion đa dạng góc,
                                     # GRPO mới có phương sai để học (tránh collapse)
        top_p=0.95,
        beta=0,                    # khởi đầu tắt KL cho đơn giản; bật sau nếu lệch
        use_vllm=args.use_vllm,
        vllm_mode="colocate" if args.use_vllm else None,
        vllm_gpu_memory_utilization=args.vllm_gpu_util,
        vllm_max_model_length=args.max_model_len,
        bf16=True,                   # nửa độ chính xác -> giảm VRAM
        gradient_checkpointing=not args.no_grad_checkpoint,
        log_completions=args.log_completions,  # mặc định TẮT -> log gọn
        num_completions_to_print=2,            # nếu bật, chỉ in 2 mẫu
        logging_steps=1,
        save_steps=200,
        remove_unused_columns=False, # GIỮ cột episode_id cho reward func
        dataloader_num_workers=0,    # giữ thứ tự lấy mẫu xác định (không prefetch)
        shuffle_dataset=False,
        report_to=([] if args.report_to == "none" else [args.report_to]),
    )
    config = _build_config(GRPOConfig, want)

    # LoRA: chỉ train vài % tham số -> optimizer states gần như biến mất,
    # giải phóng VRAM cho bản vLLM + KV cache. Đây là cách giảm bộ nhớ hiệu quả
    # nhất cho GPU 40GB với VLM 3B.
    peft_config = None
    if not args.no_lora:
        from peft import LoraConfig
        peft_config = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
            bias="none", task_type="CAUSAL_LM",
            # Nhắm các lớp tuyến tính của khối ngôn ngữ Qwen (không đụng vision tower).
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        )
        print(f"[grpo_train] LoRA bật: r={args.lora_r} alpha={args.lora_alpha}")
    else:
        print("[grpo_train] LoRA TẮT (full fine-tune).")

    trainer = GRPOTrainer(
        model=args.model,
        args=config,
        train_dataset=dataset,
        reward_funcs=[reward_func],
        peft_config=peft_config,
    )
    try:
        trainer.train()
    finally:
        manager.close()
        metrics.close()


if __name__ == "__main__":
    main()
