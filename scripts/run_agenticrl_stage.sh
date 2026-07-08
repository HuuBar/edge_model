#!/usr/bin/env bash
# 版权所有 © 2026 深圳途明智启科技有限公司。保留所有权利。
# 未经书面许可，任何单位或个人不得复制、传播、发布、转卖、改编、仿制或用于商业用途。
# 侵权必究。
#
# 一键启动 AgenticRL 训练。
#
# 当前项目里的 AgenticRL 具体实现是：
#   verl GRPO + custom AgentLoop + 本项目 tool runtime + verifier/LLM judge reward。
#
# 默认用途：
#   - 从 SFT 后的 模型开始。
#   - 使用 RL parquet 数据在线 rollout。
#   - rollout_n 默认为 8。
#   - 每条 rollout 都落 trajectory、score、token_trace、sandbox_final_state。
#   - 默认开启训练/推理不一致处理：rollout logprob + sequence-level TIS。
#
# 常用覆盖方式：
#   MODEL=models/my_sft_model_hf \
#   ROLLOUT_N=8 \
#   TOTAL_STEPS=166 \
#   bash scripts/run_agenticrl_stage.sh
#
# 额外 verl override 可以直接追加在脚本参数后：
#   bash scripts/run_agenticrl_stage.sh actor_rollout_ref.actor.optim.lr=5e-7 trainer.save_freq=10

set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Python 环境。默认使用单独的 verl 环境。
PYTHON="${PYTHON:-$ROOT/.venv-verl/bin/python}"

# GRPO 默认从 Qwen3-8B 启动；想接 SFT 冷启就把 MODEL 指向 run_sft_stage.sh 输出的 models/<run>_hf。
MODEL="${MODEL:-models/original_model/Qwen3-8B}"

# 分布式规模。默认 1 节点 x 64 GPU；多节点时可改成 NNODES=8 N_GPUS_PER_NODE=8。
NNODES="${NNODES:-1}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-64}"

# RL 数据。默认接路由切分：train=RL 池(2274)，val=EVAL 池(305 held-out，绝不进 train)。
TRAIN_FILE="${TRAIN_FILE:-data/rl/stage5/train.parquet}"
VAL_FILE="${VAL_FILE:-data/rl/eval/train.parquet}"

# 运行名。也会作为 VERL_RUN_ID，决定 rollout artifact 目录名。
RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${RUN_NAME:-agenticrl_stage_${RUN_TS}}"
export VERL_RUN_ID="${VERL_RUN_ID:-$RUN_NAME}"

# checkpoint 输出目录。默认带 timestamp，避免覆盖已有产物。
SAVE_PATH="${SAVE_PATH:-checkpoints/grpo/${RUN_NAME}}"

# 可选：把最终 GRPO actor checkpoint merge 成 HuggingFace 模型。
MERGE_HF="${MERGE_HF:-0}"
MERGED_MODEL_DIR="${MERGED_MODEL_DIR:-models/${RUN_NAME}_hf}"

# W&B project；.env 里已有 WANDB_PROJECT 时会优先使用。
PROJECT="${PROJECT:-${WANDB_PROJECT:-industrial_posttrain_training}}"
LOGGER="${LOGGER:-console,wandb}"

# 样本数量。当前默认跑完整训练集；val 默认 19 是当前 stage val split。
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:--1}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-19}"

# 当前 RL 池 train split 2273 条，batch_size=1 时 2273 step = 1 epoch（全部 case 各采一次）。
# 注意：GRPO 每步 ×ROLLOUT_N 次在线 rollout + judge，跑满很重；想先看曲线可 TOTAL_STEPS=500 覆盖。
TOTAL_STEPS="${TOTAL_STEPS:-2273}"

# GRPO rollout 数。用户要求的 n=8 在这里设置。
ROLLOUT_N="${ROLLOUT_N:-8}"

# batch 和 PPO micro batch。单卡安全默认值。
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-1}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-1}"
PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-1}"
LOG_PROB_MICRO_BATCH_SIZE="${LOG_PROB_MICRO_BATCH_SIZE:-1}"

# prompt + response 最大长度。两者之和会作为 vLLM max_model_len。
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-12288}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"

# vLLM 显存占用比例。8B 默认先保守设置，集群训练可按显存情况调高。
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.2}"

# AgentLoop worker 数。单卡先用 1，避免 LLM judge / tool runtime 并发过高。
AGENT_WORKERS="${AGENT_WORKERS:-1}"

# validation 和 checkpoint 频率。
# VAL_BEFORE_TRAIN=true 会先跑一次训练前验证，方便做 before/after 对比。
TEST_FREQ="${TEST_FREQ:-50}"
SAVE_FREQ="${SAVE_FREQ:-$TOTAL_STEPS}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-true}"
RESUME_MODE="${RESUME_MODE:-disable}"

# 训练/推理不一致处理。
# rollout 由 vLLM 生成，训练侧由 FSDP/PyTorch 计算 logprob，两条路径可能有偏差。
# 默认开启 rollout_correction：保存 rollout logprob，并做 sequence-level truncated IS。
ROLLOUT_CORRECTION="${ROLLOUT_CORRECTION:-1}"
ROLLOUT_IS="${ROLLOUT_IS:-sequence}"
ROLLOUT_IS_THRESHOLD="${ROLLOUT_IS_THRESHOLD:-2.0}"
ROLLOUT_IS_BATCH_NORMALIZE="${ROLLOUT_IS_BATCH_NORMALIZE:-false}"
ROLLOUT_BYPASS_MODE="${ROLLOUT_BYPASS_MODE:-false}"

# dry-run 只打印最终命令，不启动训练。
DRY_RUN="${DRY_RUN:-0}"

# 实时日志。脚本会尽量用 pseudo-TTY 保留 tqdm 进度条，再 tee 到日志文件。
LOG_DIR="${LOG_DIR:-runs/train_logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_NAME}.log}"

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-online}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

ensure_python() {
  if [[ "$PYTHON" == */* ]]; then
    [[ -x "$PYTHON" ]] && return 0
  elif command -v "$PYTHON" >/dev/null 2>&1; then
    return 0
  fi
  echo "[agenticrl] Python not found or not executable: $PYTHON" >&2
  exit 1
}

abs_path() {
  local path="$1"
  if [[ "$path" == /* ]]; then
    printf '%s\n' "$path"
  else
    printf '%s/%s\n' "$ROOT" "$path"
  fi
}

run_and_log() {
  mkdir -p "$(dirname "$LOG_FILE")"
  echo "[agenticrl] log_file=$LOG_FILE"
  echo "[agenticrl] command=$*" | tee -a "$LOG_FILE"

  set +e
  if command -v script >/dev/null 2>&1; then
    # script 提供 pseudo-TTY，让 verl 的 Training Progress tqdm 保持实时显示。
    local quoted
    printf -v quoted '%q ' "$@"
    script -qefc "$quoted" /dev/null 2>&1 | tee -a "$LOG_FILE"
    local status=${PIPESTATUS[0]}
    set -e
    return "$status"
  fi

  "$@" 2>&1 | tee -a "$LOG_FILE"
  local status=${PIPESTATUS[0]}
  set -e
  return "$status"
}

print_config() {
  cat <<EOF | tee -a "$LOG_FILE"
[agenticrl] config
  ROOT=$ROOT
  PYTHON=$PYTHON
  MODEL=$MODEL
  NNODES=$NNODES
  N_GPUS_PER_NODE=$N_GPUS_PER_NODE
  TRAIN_FILE=$TRAIN_FILE
  VAL_FILE=$VAL_FILE
  RUN_NAME=$RUN_NAME
  VERL_RUN_ID=$VERL_RUN_ID
  SAVE_PATH=$SAVE_PATH
  MERGE_HF=$MERGE_HF
  MERGED_MODEL_DIR=$MERGED_MODEL_DIR
  PROJECT=$PROJECT
  LOGGER=$LOGGER
  TRAIN_MAX_SAMPLES=$TRAIN_MAX_SAMPLES
  VAL_MAX_SAMPLES=$VAL_MAX_SAMPLES
  TOTAL_STEPS=$TOTAL_STEPS
  ROLLOUT_N=$ROLLOUT_N
  TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE
  VAL_BATCH_SIZE=$VAL_BATCH_SIZE
  PPO_MINI_BATCH_SIZE=$PPO_MINI_BATCH_SIZE
  PPO_MICRO_BATCH_SIZE=$PPO_MICRO_BATCH_SIZE
  LOG_PROB_MICRO_BATCH_SIZE=$LOG_PROB_MICRO_BATCH_SIZE
  MAX_PROMPT_LENGTH=$MAX_PROMPT_LENGTH
  MAX_RESPONSE_LENGTH=$MAX_RESPONSE_LENGTH
  ROLLOUT_GPU_MEMORY_UTILIZATION=$ROLLOUT_GPU_MEMORY_UTILIZATION
  AGENT_WORKERS=$AGENT_WORKERS
  TEST_FREQ=$TEST_FREQ
  SAVE_FREQ=$SAVE_FREQ
  VAL_BEFORE_TRAIN=$VAL_BEFORE_TRAIN
  RESUME_MODE=$RESUME_MODE
  ROLLOUT_CORRECTION=$ROLLOUT_CORRECTION
  ROLLOUT_IS=$ROLLOUT_IS
  ROLLOUT_IS_THRESHOLD=$ROLLOUT_IS_THRESHOLD
  ROLLOUT_IS_BATCH_NORMALIZE=$ROLLOUT_IS_BATCH_NORMALIZE
  ROLLOUT_BYPASS_MODE=$ROLLOUT_BYPASS_MODE
  DRY_RUN=$DRY_RUN
EOF
}

ensure_python
mkdir -p "$LOG_DIR"
print_config

cmd=(
  # 这里先进入本项目 Python 启动器；scripts/train_grpo_verl.py 再拼出 verl.trainer.main_ppo 命令。
  # 这样一键脚本只管环境变量/日志/后处理，Python 启动器集中维护 verl overrides。
  "$PYTHON" "scripts/train_grpo_verl.py"
  --model "$MODEL"
  --train-file "$TRAIN_FILE"
  --val-file "$VAL_FILE"
  --save-path "$SAVE_PATH"
  --experiment "$RUN_NAME"
  --project "$PROJECT"
  --total-training-steps "$TOTAL_STEPS"
  --train-max-samples "$TRAIN_MAX_SAMPLES"
  --val-max-samples "$VAL_MAX_SAMPLES"
  --train-batch-size "$TRAIN_BATCH_SIZE"
  --val-batch-size "$VAL_BATCH_SIZE"
  --ppo-mini-batch-size "$PPO_MINI_BATCH_SIZE"
  --ppo-micro-batch-size "$PPO_MICRO_BATCH_SIZE"
  --log-prob-micro-batch-size "$LOG_PROB_MICRO_BATCH_SIZE"
  --max-prompt-length "$MAX_PROMPT_LENGTH"
  --max-response-length "$MAX_RESPONSE_LENGTH"
  --rollout-n "$ROLLOUT_N"
  --rollout-gpu-memory-utilization "$ROLLOUT_GPU_MEMORY_UTILIZATION"
  --agent-workers "$AGENT_WORKERS"
  --test-freq "$TEST_FREQ"
  --logger "$LOGGER"
  --nnodes "$NNODES"
  --n-gpus-per-node "$N_GPUS_PER_NODE"
)

if [[ "$DRY_RUN" == "1" ]]; then
  cmd+=(--dry-run)
fi

# scripts/train_grpo_verl.py 里有一些安全默认值；这里用追加 override 覆盖成一键训练默认。
# 这些 override 会原样透传给 verl/Hydra，是“一键脚本默认策略”真正生效的位置。
cmd+=(
  "trainer.save_freq=${SAVE_FREQ}"
  "trainer.val_before_train=${VAL_BEFORE_TRAIN}"
  "trainer.resume_mode=${RESUME_MODE}"
)

if [[ "$ROLLOUT_CORRECTION" == "1" ]]; then
  cmd+=(
    # 训练/推理不一致处理：rollout 用 vLLM，训练 logprob 用 FSDP/PyTorch。
    # 打开 calculate_log_probs 后，verl 可以做 sequence-level truncated importance sampling。
    "actor_rollout_ref.rollout.calculate_log_probs=True"
    "algorithm.rollout_correction.rollout_is=${ROLLOUT_IS}"
    "algorithm.rollout_correction.rollout_is_threshold=${ROLLOUT_IS_THRESHOLD}"
    "algorithm.rollout_correction.rollout_is_batch_normalize=${ROLLOUT_IS_BATCH_NORMALIZE}"
    "algorithm.rollout_correction.bypass_mode=${ROLLOUT_BYPASS_MODE}"
  )
fi

# 允许调用方追加任意 verl/Hydra override。
# 例如：bash scripts/run_agenticrl_stage.sh actor_rollout_ref.actor.optim.lr=5e-7
cmd+=("$@")

# run_and_log 会尽量用 pseudo-TTY 保留 verl Training Progress tqdm，并同步写日志文件。
run_and_log "${cmd[@]}"
train_status=$?
if [[ "$train_status" -ne 0 ]]; then
  echo "[agenticrl] training failed: exit_code=$train_status" | tee -a "$LOG_FILE"
  exit "$train_status"
fi

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[agenticrl] dry-run finished; no checkpoint, rollout, or merge was produced." | tee -a "$LOG_FILE"
  exit 0
fi

save_abs="$(abs_path "$SAVE_PATH")"
latest_file="$save_abs/latest_checkpointed_iteration.txt"
if [[ -f "$latest_file" ]]; then
  final_step="$(tr -d '[:space:]' < "$latest_file")"
else
  final_step="$TOTAL_STEPS"
fi
final_ckpt="$save_abs/global_step_${final_step}"

if [[ ! -d "$final_ckpt" ]]; then
  echo "[agenticrl] final checkpoint not found: $final_ckpt" | tee -a "$LOG_FILE"
  exit 1
fi

echo "[agenticrl] final_checkpoint=$final_ckpt" | tee -a "$LOG_FILE"
echo "[agenticrl] rollout_dir=$ROOT/data/rollouts_verl/$VERL_RUN_ID" | tee -a "$LOG_FILE"

# 生成本地 rollout 分数汇总，便于训练后直接看每条 rollout 的 reward。
rollout_dir="data/rollouts_verl/$VERL_RUN_ID"
summary_dir="data/evals/${RUN_NAME}"
if [[ -f "$rollout_dir/scores.jsonl" ]]; then
  summary_cmd=(
    # 训练后从 rollout_dir/scores.jsonl 生成本地汇总，便于查看每条 rollout 的 reward 分布。
    "$PYTHON" "scripts/summarize_runs.py"
    --kind verl
    --input "$rollout_dir"
    --out "$summary_dir"
  )
  run_and_log "${summary_cmd[@]}"
  echo "[agenticrl] summary_dir=$ROOT/$summary_dir" | tee -a "$LOG_FILE"
else
  echo "[agenticrl] scores.jsonl not found; skip summary: $ROOT/$rollout_dir/scores.jsonl" | tee -a "$LOG_FILE"
fi

if [[ "$MERGE_HF" == "1" ]]; then
  actor_ckpt="$final_ckpt/actor"
  if [[ ! -d "$actor_ckpt" ]]; then
    echo "[agenticrl] actor checkpoint not found for merge: $actor_ckpt" | tee -a "$LOG_FILE"
    exit 1
  fi
  merged_abs="$(abs_path "$MERGED_MODEL_DIR")"
  merge_cmd=(
    # GRPO checkpoint 的 actor 子目录才是需要 merge 的策略模型。
    "$PYTHON" -m verl.model_merger merge
    --backend fsdp
    --local_dir "$actor_ckpt"
    --target_dir "$merged_abs"
  )
  run_and_log "${merge_cmd[@]}"
  merge_status=$?
  if [[ "$merge_status" -ne 0 ]]; then
    echo "[agenticrl] merge failed: exit_code=$merge_status" | tee -a "$LOG_FILE"
    exit "$merge_status"
  fi
  echo "[agenticrl] merged_hf_model=$merged_abs" | tee -a "$LOG_FILE"
fi

echo "[agenticrl] done" | tee -a "$LOG_FILE"
