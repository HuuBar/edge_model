#!/usr/bin/env bash
# 版权所有 © 2026 深圳途明智启科技有限公司。保留所有权利。
# 未经书面许可，任何单位或个人不得复制、传播、发布、转卖、改编、仿制或用于商业用途。
# 侵权必究。
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/.venv-verl/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  echo "[data] Python not found: $PYTHON" >&2
  echo "[data] run: bash scripts/setup_env.sh" >&2
  exit 1
fi

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

"$PYTHON" scripts/build_sft.py \
  --batch data/batches/sft \
  --out data/sft/stage5 \
  --val-every 10

"$PYTHON" scripts/build_grpo.py \
  --batch data/batches/rl \
  --out data/rl/stage5 \
  --rollout-root data/rollouts_verl \
  --val-every 2274

"$PYTHON" scripts/build_grpo.py \
  --batch data/batches/eval \
  --out data/rl/eval \
  --rollout-root data/rollouts_verl \
  --val-every 305

echo "[data] generated:"
echo "  data/sft/stage5/train.parquet"
echo "  data/sft/stage5/val.parquet"
echo "  data/rl/stage5/train.parquet"
echo "  data/rl/eval/train.parquet"
