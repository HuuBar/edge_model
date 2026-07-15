"""Launch a verl MultiTurn SFT run for the prepared stage data.

这个脚本是 SFT 的“Python 启动器”：
- 它不构造训练样本，样本已经由 ``scripts/build_sft.py`` / ``train.sft_builder`` 生成。
- 它不手写训练循环，训练循环交给 ``verl.trainer.sft_trainer``。
- 它负责把本项目的数据列名、模型路径、batch/length/logger/checkpoint 参数翻译成
  verl/Hydra override，然后用 subprocess 真正启动 verl。
"""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
# 允许直接 `python scripts/train_sft.py`，不用调用方手动设置 PYTHONPATH。
sys.path.insert(0, str(ROOT))

from agent.providers.factory import load_dotenv  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    # 以下参数分两类：
    # 1) 本项目侧路径：model、train/val parquet、save path。
    # 2) verl 训练 override：max_length、batch、lr、logger、step 等。
    parser.add_argument("--model", default="models/original_model/Qwen3-8B")
    parser.add_argument("--train-file", default="data/sft/stage5/train.parquet")
    parser.add_argument("--val-file", default="data/sft/stage5/val.parquet")
    parser.add_argument("--save-path", default="checkpoints/sft/stage5_qwen3_8b")
    parser.add_argument("--experiment", default="sft_stage5")
    parser.add_argument("--project", default=None)
    parser.add_argument("--train-max-samples", type=int, default=1)
    parser.add_argument("--val-max-samples", type=int, default=1)
    parser.add_argument("--total-training-steps", type=int, default=1)
    parser.add_argument("--test-freq", type=int, default=1)
    parser.add_argument("--save-freq", default="-1")
    parser.add_argument("--max-length", type=int, default=12288)
    parser.add_argument("--truncation", choices=["error", "left", "right"], default="left")
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--lr", default="1e-5")
    parser.add_argument("--logger", default="console,wandb")
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--nproc-per-node", type=int, default=64)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", default="29500")
    parser.add_argument("--skip-loss-mask-check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    # 读取 .env 中的 WANDB_API_KEY / WANDB_PROJECT / judge key 等环境变量。
    # SFT 本身不会调用 judge，但 W&B logger 需要这些环境变量。
    load_dotenv(ROOT / ".env")
    project = args.project or os.environ.get("WANDB_PROJECT", "industrial_posttrain_training")
    dist_args = [
        f"--nnodes={args.nnodes}",
        f"--nproc_per_node={args.nproc_per_node}",
    ]
    if args.nnodes == 1:
        dist_args.insert(0, "--standalone")
    else:
        dist_args.extend(
            [
                f"--node_rank={args.node_rank}",
                f"--master_addr={args.master_addr}",
                f"--master_port={args.master_port}",
            ]
        )
    # 关键接线点：这里开始把“本项目参数”翻译成 verl.sft_trainer 能理解的 Hydra overrides。
    # 真正的训练入口是下面的 `python -m torch.distributed.run --module verl.trainer.sft_trainer`。
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        *dist_args,
        "--module",
        "verl.trainer.sft_trainer",
        # 数据列名接线：train.sft_builder 生成的 parquet 里正好有 messages/tools/enable_thinking。
        # MultiTurnSFTDataset 会读取这些列，用 tokenizer.apply_chat_template 构造成监督样本。
        f"data.train_files={ROOT / args.train_file}",
        f"data.val_files={ROOT / args.val_file}",
        "data.messages_key=messages",
        "data.tools_key=tools",
        "data.enable_thinking_key=enable_thinking",
        "data.enable_thinking_default=False",
        "data.ignore_input_ids_mismatch=True",
        f"data.train_max_samples={args.train_max_samples}",
        f"data.val_max_samples={args.val_max_samples}",
        f"data.max_length={args.max_length}",
        f"data.truncation={args.truncation}",
        "data.pad_mode=no_padding",
        # 这里使用静态 batch，接线阶段更容易解释和排查 loss/token 数量。
        f"data.train_batch_size={args.train_batch_size}",
        f"data.micro_batch_size_per_gpu={args.micro_batch_size}",
        "data.use_dynamic_bsz=False",
        "data.num_workers=0",
        f"optim.lr={args.lr}",
        "engine=fsdp",
        # 8B 默认走 FSDP 分布式；offload 保留为稳妥默认值。
        "engine.model_dtype=bfloat16",
        "engine.param_offload=True",
        "engine.optimizer_offload=True",
        "engine.use_torch_compile=False",
        f"model.path={ROOT / args.model}",
        "+model.override_config.attn_implementation=sdpa",
        "model.use_remove_padding=True",
        "model.enable_activation_offload=True",
        f"trainer.default_local_dir={ROOT / args.save_path}",
        # W&B 中看到的 project/run name 就来自这两项。
        f"trainer.project_name={project}",
        f"trainer.experiment_name={args.experiment}",
        f"trainer.logger=[{args.logger}]",
        "trainer.total_epochs=1",
        f"trainer.total_training_steps={args.total_training_steps}",
        f"trainer.test_freq={args.test_freq}",
        f"trainer.save_freq={args.save_freq}",
        f"trainer.n_gpus_per_node={args.nproc_per_node}",
        f"trainer.nnodes={args.nnodes}",
        "trainer.resume_mode=disable",
        *args.overrides,
    ]
    if args.dry_run:
        # dry-run 只打印最终 verl 命令，适合教学时先看“脚本到底会启动什么”。
        print(" ".join(str(item) for item in cmd))
        return 0
    if not args.skip_loss_mask_check:
        # 启动 GPU 训练前先确认监督 token 没被 max_length 截没。
        _check_loss_masks(args)
    env = _env()
    # 这里才是真正启动 verl；check=False 是为了把原始退出码返回给外层 shell 脚本。
    return subprocess.run(cmd, cwd=ROOT, env=env, check=False).returncode


def _check_loss_masks(args: argparse.Namespace) -> None:
    """Fail fast when truncation removes all supervised tokens.

    MultiTurn SFT 只对 assistant gold tokens 计算 loss。如果 prompt 太长、max_length 太小，
    truncation 可能把 assistant 部分截掉，导致 loss_mask 全 0。这个 preflight 用 verl
    自己的 ``MultiTurnSFTDataset`` 读同一份 parquet，提前复现真实训练的数据处理路径。
    """
    from omegaconf import OmegaConf
    from verl.utils import hf_tokenizer
    from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset

    tokenizer = hf_tokenizer(str(ROOT / args.model))
    # 这份 config 和上面传给 verl.sft_trainer 的 data.* override 保持同口径。
    dataset_config = OmegaConf.create(
        {
            "messages_key": "messages",
            "tools_key": "tools",
            "enable_thinking_key": "enable_thinking",
            "enable_thinking_default": False,
            "ignore_input_ids_mismatch": True,
            "pad_mode": "no_padding",
            "truncation": args.truncation,
            "max_length": args.max_length,
            "shuffle": False,
        }
    )
    checks = [
        ("train", ROOT / args.train_file, args.train_max_samples),
        ("val", ROOT / args.val_file, args.val_max_samples),
    ]
    for split, path, max_samples in checks:
        # 直接实例化 verl dataset，确保检查的是“verl 实际会训练的样本”，不是我们自己猜的格式。
        dataset = MultiTurnSFTDataset(str(path), tokenizer, dataset_config, max_samples=max_samples)
        loss_counts = [_loss_token_count(dataset[index]["loss_mask"]) for index in range(len(dataset))]
        loss_sum = sum(loss_counts)
        if loss_sum <= 0:
            raise RuntimeError(
                f"{split} SFT loss_mask is empty after max_length={args.max_length}; "
                "increase --max-length or choose shorter rows before launching verl."
            )
        print(
            f"[sft preflight] {split}: rows={len(dataset)} "
            f"loss_tokens={loss_sum} min={min(loss_counts)} max={max(loss_counts)}"
        )


def _loss_token_count(loss_mask) -> int:
    """兼容普通 tensor 和 nested tensor 两种 loss_mask 表示。"""

    if getattr(loss_mask, "is_nested", False):
        return int(loss_mask.values().sum().item())
    return int(loss_mask.sum().item())


def _env() -> dict[str, str]:
    """构造子进程环境，确保 verl 能 import 本项目包并默认在线写 W&B。"""

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
    env.setdefault("WANDB_MODE", "online")
    return env


if __name__ == "__main__":
    raise SystemExit(main())
