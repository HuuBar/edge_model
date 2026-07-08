#!/usr/bin/env bash
# 版权所有 © 2026 深圳途明智启科技有限公司。保留所有权利。
# 未经书面许可，任何单位或个人不得复制、传播、发布、转卖、改编、仿制或用于商业用途。
# 侵权必究。
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv-verl}"

"$PYTHON_BIN" -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install -U pip setuptools wheel

"$VENV_DIR/bin/python" -m pip install -e ".[monitoring]"
"$VENV_DIR/bin/python" -m pip install -e "verl/upstream[vllm]"
"$VENV_DIR/bin/python" -m pip install pandas pyarrow transformers accelerate sentencepiece

echo "[setup] environment ready: $ROOT/$VENV_DIR"
echo "[setup] next: cp .env.example .env, fill keys, then run training scripts"
