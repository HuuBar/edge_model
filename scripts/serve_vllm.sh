#!/usr/bin/env bash
# 版权所有 © 2026 深圳途明智启科技有限公司。保留所有权利。
# 未经书面许可，任何单位或个人不得复制、传播、发布、转卖、改编、仿制或用于商业用途。
# 侵权必究。
# 启动本地 vLLM（Qwen3-8B）—— 固化 native tool calling 所需的 flag。
#
# ⚠️ 必带 --enable-auto-tool-choice --tool-call-parser hermes：
#    .env 已设 VLLM_NATIVE_TOOL_CALLING=true（rollout 走原生工具格式，与 Qwen3 训练格式一致）。
#    若不带这两个 flag 启动，server 会拒绝 tool_choice，整条 rollout 链路报 400。
#    （2026-06-20 实测：旧启动命令没带 flag → native 不可用 → 当时探针退回自定义文本菜单，
#     压低了 Qwen 基线；切 native 后须由本脚本保证 flag 不丢。）
#
# 用法：bash scripts/serve_vllm.sh        （前台）
#       setsid nohup bash scripts/serve_vllm.sh > /tmp/vllm.log 2>&1 < /dev/null &   （后台 detached）

set -euo pipefail

MODEL="${VLLM_MODEL:-models/original_model/Qwen3-8B}"
PORT="${VLLM_PORT:-8000}"
MAXLEN="${VLLM_MAX_MODEL_LEN:-32768}"
GPU_UTIL="${VLLM_GPU_UTIL:-0.85}"
PARSER="${VLLM_TOOL_PARSER:-hermes}"   # Qwen3 用 hermes 工具解析器

exec vllm serve "$MODEL" \
  --port "$PORT" \
  --served-model-name "$MODEL" \
  --max-model-len "$MAXLEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --enable-auto-tool-choice \
  --tool-call-parser "$PARSER"
