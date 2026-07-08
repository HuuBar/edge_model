#!/usr/bin/env bash
# 版权所有 © 2026 深圳途明智启科技有限公司。保留所有权利。
# 未经书面许可，任何单位或个人不得复制、传播、发布、转卖、改编、仿制或用于商业用途。
# 侵权必究。
#
# 一键启动 SFT 训练。
#
# 默认用途：
#   - 使用当前已构造好的 stage SFT parquet 数据。
#   - 从 Qwen3-8B base model 开始做 SFT。
#   - 实时显示 verl 原生训练进度和 loss 日志。
#   - 训练成功后默认把 FSDP checkpoint merge 成 HuggingFace 模型目录。
#
# 常用覆盖方式：
#   MODEL=models/original_model/Qwen3-8B \
#   NNODES=1 \
#   NPROC_PER_NODE=64 \
#   TOTAL_STEPS=166 \
#   RUN_NAME=sft_qwen3_8b_stage_$(date +%Y%m%d_%H%M%S) \
#   bash scripts/run_sft_stage.sh
#
# 额外 verl override 可以直接追加在脚本参数后：
#   bash scripts/run_sft_stage.sh trainer.save_freq=20 optim.lr=5e-6

set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Python 环境。默认使用单独的 verl 环境。
PYTHON="${PYTHON:-$ROOT/.venv-verl/bin/python}"

# 模型路径。默认 Qwen3-8B。
MODEL="${MODEL:-models/original_model/Qwen3-8B}"

# 分布式规模。默认 1 节点 x 64 GPU；多节点时设置 NNODES/NPROC_PER_NODE/NODE_RANK/MASTER_ADDR。
NNODES="${NNODES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-64}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

# SFT 数据。默认使用发布包内已构造好的 stage5 SFT parquet；不修改 batch 源数据。
TRAIN_FILE="${TRAIN_FILE:-data/sft/stage5/train.parquet}"
VAL_FILE="${VAL_FILE:-data/sft/stage5/val.parquet}"

# 运行名。用于 W&B experiment、checkpoint 目录、日志文件。
RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${RUN_NAME:-sft_stage_${RUN_TS}}"

# checkpoint 输出目录。默认带 timestamp，避免覆盖已有产物。
SAVE_PATH="${SAVE_PATH:-checkpoints/sft/${RUN_NAME}}"

# 训练后 HF merged model 输出目录。
MERGED_MODEL_DIR="${MERGED_MODEL_DIR:-models/${RUN_NAME}_hf}"

# W&B project；.env 里已有 WANDB_PROJECT 时会优先使用。
PROJECT="${PROJECT:-${WANDB_PROJECT:-industrial_posttrain_training}}"
LOGGER="${LOGGER:-console,wandb}"

# 样本数量。-1 表示使用对应 parquet 的全部样本。
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:--1}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:--1}"

# 总训练 step。当前 SFT 池 train split 121 条，batch_size=1 时 121 step = 1 epoch（跑满全部 case）。
TOTAL_STEPS="${TOTAL_STEPS:-121}"

# SFT 验证频率。设为 20 可以看到验证 loss 曲线；设为 -1 则只在最后验证。
TEST_FREQ="${TEST_FREQ:-20}"

# checkpoint 保存频率。-1 表示只保存最终 checkpoint；设为 20 会额外保存中间点。
SAVE_FREQ="${SAVE_FREQ:--1}"

# Qwen3 当前样本较长，12288 是已验证不会截掉监督 token 的安全长度。
MAX_LENGTH="${MAX_LENGTH:-12288}"
TRUNCATION="${TRUNCATION:-left}"

# 8B 训练时可继续从 1 开始，显存允许再调大。
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"

# SFT 学习率。
LR="${LR:-1e-5}"

# 是否跳过 loss mask preflight。默认不跳过，避免 max_length 截断导致无监督 token。
SKIP_LOSS_MASK_CHECK="${SKIP_LOSS_MASK_CHECK:-0}"

# 训练成功后是否自动 merge 成 HF 模型目录。
MERGE_HF="${MERGE_HF:-1}"

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
  echo "[sft] Python not found or not executable: $PYTHON" >&2
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
  echo "[sft] log_file=$LOG_FILE"
  echo "[sft] command=$*" | tee -a "$LOG_FILE"

  set +e
  if command -v script >/dev/null 2>&1; then
    # script 提供 pseudo-TTY，让 tqdm/进度条保持实时显示；tee 同时保存日志。
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
[sft] config
  ROOT=$ROOT
  PYTHON=$PYTHON
  MODEL=$MODEL
  NNODES=$NNODES
  NPROC_PER_NODE=$NPROC_PER_NODE
  NODE_RANK=$NODE_RANK
  MASTER_ADDR=$MASTER_ADDR
  MASTER_PORT=$MASTER_PORT
  TRAIN_FILE=$TRAIN_FILE
  VAL_FILE=$VAL_FILE
  RUN_NAME=$RUN_NAME
  SAVE_PATH=$SAVE_PATH
  MERGED_MODEL_DIR=$MERGED_MODEL_DIR
  PROJECT=$PROJECT
  LOGGER=$LOGGER
  TRAIN_MAX_SAMPLES=$TRAIN_MAX_SAMPLES
  VAL_MAX_SAMPLES=$VAL_MAX_SAMPLES
  TOTAL_STEPS=$TOTAL_STEPS
  TEST_FREQ=$TEST_FREQ
  SAVE_FREQ=$SAVE_FREQ
  MAX_LENGTH=$MAX_LENGTH
  TRUNCATION=$TRUNCATION
  TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE
  MICRO_BATCH_SIZE=$MICRO_BATCH_SIZE
  LR=$LR
  MERGE_HF=$MERGE_HF
  DRY_RUN=$DRY_RUN
EOF
}

ensure_python
mkdir -p "$LOG_DIR"
print_config

cmd=(
  # 这里不是直接 `python -m verl...`，而是先进入本项目 Python 启动器；
  # scripts/train_sft.py 会做 loss-mask preflight，再拼出 verl.trainer.sft_trainer 命令。
  "$PYTHON" "scripts/train_sft.py"
  --model "$MODEL"
  --train-file "$TRAIN_FILE"
  --val-file "$VAL_FILE"
  --save-path "$SAVE_PATH"
  --experiment "$RUN_NAME"
  --project "$PROJECT"
  --train-max-samples "$TRAIN_MAX_SAMPLES"
  --val-max-samples "$VAL_MAX_SAMPLES"
  --total-training-steps "$TOTAL_STEPS"
  --test-freq "$TEST_FREQ"
  --save-freq "$SAVE_FREQ"
  --max-length "$MAX_LENGTH"
  --truncation "$TRUNCATION"
  --train-batch-size "$TRAIN_BATCH_SIZE"
  --micro-batch-size "$MICRO_BATCH_SIZE"
  --lr "$LR"
  --logger "$LOGGER"
  --nnodes "$NNODES"
  --nproc-per-node "$NPROC_PER_NODE"
  --node-rank "$NODE_RANK"
  --master-addr "$MASTER_ADDR"
  --master-port "$MASTER_PORT"
)

if [[ "$SKIP_LOSS_MASK_CHECK" == "1" ]]; then
  cmd+=(--skip-loss-mask-check)
fi

if [[ "$DRY_RUN" == "1" ]]; then
  cmd+=(--dry-run)
fi

# 允许调用方追加任意 verl/Hydra override。
# 例如：bash scripts/run_sft_stage.sh optim.lr=5e-6 trainer.save_freq=20
cmd+=("$@")

# run_and_log 会尽量用 pseudo-TTY 保留 tqdm 进度条，并把完整 stdout/stderr 写入 LOG_FILE。
run_and_log "${cmd[@]}"
train_status=$?
if [[ "$train_status" -ne 0 ]]; then
  echo "[sft] training failed: exit_code=$train_status" | tee -a "$LOG_FILE"
  exit "$train_status"
fi

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[sft] dry-run finished; no checkpoint or merge was produced." | tee -a "$LOG_FILE"
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
  echo "[sft] final checkpoint not found: $final_ckpt" | tee -a "$LOG_FILE"
  exit 1
fi

echo "[sft] final_checkpoint=$final_ckpt" | tee -a "$LOG_FILE"

if [[ "$MERGE_HF" == "1" ]]; then
  merged_abs="$(abs_path "$MERGED_MODEL_DIR")"
  merge_cmd=(
    # verl SFT 默认保存 FSDP checkpoint；推理/继续训练常用 HF 格式，所以这里训练后自动 merge。
    "$PYTHON" -m verl.model_merger merge
    --backend fsdp
    --local_dir "$final_ckpt"
    --target_dir "$merged_abs"
  )
  run_and_log "${merge_cmd[@]}"
  merge_status=$?
  if [[ "$merge_status" -ne 0 ]]; then
    echo "[sft] merge failed: exit_code=$merge_status" | tee -a "$LOG_FILE"
    exit "$merge_status"
  fi
  echo "[sft] merged_hf_model=$merged_abs" | tee -a "$LOG_FILE"
fi

echo "[sft] done" | tee -a "$LOG_FILE"
