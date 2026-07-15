"""Launch a verl GRPO run with the industrial AgentLoop.

这个脚本是 GRPO 的“Python 启动器”：
- 它不自己实现 PPO/GRPO 更新，训练循环交给 ``verl.trainer.main_ppo``。
- 它不直接调用 ``agent.runtime.run_agent_loop``，而是通过 verl 的 async AgentLoop
  接口注册 ``industrial_posttrain_agent``。
- 它负责把 prompt parquet、AgentLoop 配置、rollout/vLLM 参数、reward 关闭/接管方式
  翻译成 verl/Hydra override。
"""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
# 允许直接运行脚本时 import 本项目 train/agent/envs 包。
sys.path.insert(0, str(ROOT))

from agent.providers.factory import load_dotenv  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    # 数据是 prompt-only RLHFDataset；真实工具循环在 train/verl_agent_loop_adapter.py 在线执行。
    parser.add_argument("--model", default="models/original_model/Qwen3-8B")
    parser.add_argument("--train-file", default="data/rl/stage5/train.parquet")
    parser.add_argument("--val-file", default="data/rl/eval/train.parquet")
    parser.add_argument("--save-path", default="checkpoints/grpo/stage5_qwen3_8b")
    parser.add_argument("--experiment", default="grpo_stage5")
    parser.add_argument("--project", default=None)
    parser.add_argument("--total-training-steps", type=int, default=1)
    parser.add_argument("--train-max-samples", type=int, default=1)
    parser.add_argument("--val-max-samples", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--val-batch-size", type=int, default=1)
    parser.add_argument("--ppo-mini-batch-size", type=int, default=1)
    parser.add_argument("--ppo-micro-batch-size", type=int, default=1)
    parser.add_argument("--log-prob-micro-batch-size", type=int, default=1)
    parser.add_argument("--max-prompt-length", type=int, default=12288)
    parser.add_argument("--max-response-length", type=int, default=4096)
    parser.add_argument("--rollout-n", type=int, default=2)
    parser.add_argument("--rollout-gpu-memory-utilization", type=float, default=0.2)
    parser.add_argument("--agent-workers", type=int, default=1)
    parser.add_argument("--test-freq", type=int, default=-1)
    parser.add_argument("--logger", default="console,wandb")
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--n-gpus-per-node", type=int, default=64)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    # 读取 .env：W&B、DeepSeek/Ali judge key 等都从环境变量进入子进程。
    load_dotenv(ROOT / ".env")
    project = args.project or os.environ.get("WANDB_PROJECT", "industrial_posttrain_training")
    # 关键接线点：下面的 cmd 是 verl.main_ppo 的完整 Hydra override 列表。
    # GRPO 用 main_ppo + algorithm.adv_estimator=grpo，而不是 SFT trainer。
    cmd = [
        sys.executable,
        "-m",
        "verl.trainer.main_ppo",
        # 算法接线：使用 GRPO advantage，不启用 critic，也不启用独立 reward_model。
        "algorithm.adv_estimator=grpo",
        "algorithm.use_kl_in_reward=False",
        # 数据接线：train/grpo_builder.py 写出的 parquet 有 prompt/extra_info/reward_model。
        f"data.train_files={ROOT / args.train_file}",
        f"data.val_files={ROOT / args.val_file}",
        "data.prompt_key=prompt",
        # return_raw_chat=True 让 AgentLoop 拿到原始 chat messages，而不是只拿渲染后的 token。
        "data.return_raw_chat=True",
        "data.filter_overlong_prompts=False",
        "+data.filter_prompts=False",
        "data.truncation=error",
        f"data.train_max_samples={args.train_max_samples}",
        f"data.val_max_samples={args.val_max_samples}",
        f"data.train_batch_size={args.train_batch_size}",
        f"data.val_batch_size={args.val_batch_size}",
        f"data.max_prompt_length={args.max_prompt_length}",
        f"data.max_response_length={args.max_response_length}",
        "data.dataloader_num_workers=0",
        f"actor_rollout_ref.model.path={ROOT / args.model}",
        "+actor_rollout_ref.model.override_config.attn_implementation=sdpa",
        "actor_rollout_ref.model.use_remove_padding=True",
        "actor_rollout_ref.actor.optim.lr=1e-6",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={args.ppo_mini_batch_size}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={args.ppo_micro_batch_size}",
        "actor_rollout_ref.actor.use_kl_loss=True",
        "actor_rollout_ref.actor.kl_loss_coef=0.001",
        "actor_rollout_ref.actor.entropy_coeff=0",
        "actor_rollout_ref.actor.fsdp_config.param_offload=True",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",
        "actor_rollout_ref.actor.use_dynamic_bsz=False",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.mode=async",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        f"actor_rollout_ref.rollout.max_model_len={args.max_prompt_length + args.max_response_length}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={args.rollout_gpu_memory_utilization}",
        "actor_rollout_ref.rollout.enable_chunked_prefill=False",
        "actor_rollout_ref.rollout.enforce_eager=True",
        "actor_rollout_ref.rollout.free_cache_engine=True",
        f"actor_rollout_ref.rollout.n={args.rollout_n}",
        # multi_turn 接线：verl 负责 token 生成，本项目 AgentLoop 负责工具循环和 observation 回放。
        "actor_rollout_ref.rollout.multi_turn.enable=True",
        "actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1",
        "actor_rollout_ref.rollout.multi_turn.max_assistant_turns=8",
        "actor_rollout_ref.rollout.multi_turn.max_user_turns=8",
        "actor_rollout_ref.rollout.agent.default_agent_loop=industrial_posttrain_agent",
        # 这个 YAML 把注册名 industrial_posttrain_agent 解析到 train.verl_agent_loop_adapter.IndustrialPosttrainAgentLoop。
        f"actor_rollout_ref.rollout.agent.agent_loop_config_path={ROOT / 'configs/verl_agent_loop.yaml'}",
        f"actor_rollout_ref.rollout.agent.num_workers={args.agent_workers}",
        f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={args.log_prob_micro_batch_size}",
        "actor_rollout_ref.ref.fsdp_config.param_offload=True",
        f"actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu={args.log_prob_micro_batch_size}",
        "trainer.critic_warmup=0",
        "critic.enable=False",
        # reward_model 关闭：reward 由 AgentLoop 内部调用 train.verl_reward_adapter 返回给 verl。
        "reward_model.enable=False",
        f"trainer.default_local_dir={ROOT / args.save_path}",
        f"trainer.project_name={project}",
        f"trainer.experiment_name={args.experiment}",
        f"trainer.logger=[{args.logger}]",
        f"trainer.n_gpus_per_node={args.n_gpus_per_node}",
        f"trainer.nnodes={args.nnodes}",
        "trainer.total_epochs=1",
        f"trainer.total_training_steps={args.total_training_steps}",
        "trainer.save_freq=-1",
        "trainer.val_before_train=False",
        f"trainer.test_freq={args.test_freq}",
        "trainer.resume_mode=disable",
        *args.overrides,
    ]
    if args.dry_run:
        # dry-run 用于确认最终 Hydra overrides，特别适合检查 AgentLoop 是否接上。
        print(" ".join(str(item) for item in cmd))
        return 0
    env = _env()
    # 真正启动 verl GRPO；返回子进程原始退出码给一键 shell 脚本。
    return subprocess.run(cmd, cwd=ROOT, env=env, check=False).returncode


def _env() -> dict[str, str]:
    """构造 GRPO 子进程环境。

    VERL_RUN_ID 会被 AgentLoop 读取，用作 rollout artifact run 目录名。
    外层 run_agenticrl_stage.sh 通常会显式设置它；这里提供兜底。
    """

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
    env.setdefault("WANDB_MODE", "online")
    env.setdefault("VERL_RUN_ID", f"verl_grpo_stage5_{os.getpid()}")
    return env


if __name__ == "__main__":
    raise SystemExit(main())
