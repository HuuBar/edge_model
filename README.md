<!--
版权所有 © 2026 深圳途明智启科技有限公司。保留所有权利。
未经书面许可，任何单位或个人不得复制、传播、发布、转卖、改编、仿制或用于商业用途。
侵权必究。
-->

# Industrial Post-Training Training Release

这是一个干净训练包，只保留能启动训练的代码和配置。默认训练模型采用 Qwen3-8B。

不包含：

- 训练数据集（`data/` 单独打包，见下方「数据集下载」）
- 模型权重
- checkpoint
- W&B 历史记录
- rollout 历史结果
- 展示页或后台演示服务

## 数据集下载

训练数据集（`data/` 目录，约 580MB）体积较大，单独打包，请从飞书下载：


下载得到的压缩包为 `industrial_posttrain_data.zip`。把它放到项目根目录（即本 README 所在目录 `industrial_posttrain_training_release/`）下解压：

```bash
cd industrial_posttrain_training_release
unzip industrial_posttrain_data.zip
```

解压后会在项目根目录生成 `data/` 目录，最终结构应为：

```text
industrial_posttrain_training_release/
├── data/
│   ├── batches/        训练数据源文件
│   ├── sft/stage5/     已构造好的 SFT parquet
│   └── rl/             已构造好的 GRPO train/eval parquet
├── agent/
├── train/
└── ...
```

确认下面四个 parquet 存在即说明数据集放置正确：

```text
data/sft/stage5/train.parquet
data/sft/stage5/val.parquet
data/rl/stage5/train.parquet
data/rl/eval/train.parquet
```

如需重新构造数据，见第 4 节。

## 目录

```text
agent/                     Agent runtime、provider、verifier
envs/                      售后工具 runtime 和 sandbox
train/                     SFT/GRPO 数据构造与 verl AgentLoop adapter
scripts/                   一键数据构造、SFT、GRPO 启动脚本
configs/                   verl AgentLoop 配置
data/batches/              训练数据源文件
data/sft/stage5/           已构造好的 SFT parquet
data/rl/stage5/            已构造好的 GRPO train parquet
data/rl/eval/              已构造好的 GRPO eval parquet
verl/upstream/             固定随包的 verl 源码
models/original_model/     放初始模型权重，压缩包内为空
checkpoints/               训练输出目录，压缩包内为空
runs/train_logs/           训练日志目录，压缩包内为空
```

## 1. 创建训练环境

```bash
cd industrial_posttrain_training_release
bash scripts/setup_env.sh
```

如果机器上已经有可用的 verl/vLLM 环境，可以不执行上面的安装脚本，启动训练时显式传入：

```bash
PYTHON=/path/to/python bash scripts/run_sft_stage.sh
```

## 2. 配置环境变量

```bash
cp .env.example .env
```

至少填写：

```text
WANDB_API_KEY=你的 wandb key
QWEN_API_KEY=你的通义千问兼容接口 key
```

SFT 不需要 LLM judge。GRPO reward 默认需要 `VERIFIER_PROVIDER=qwen` 和 `QWEN_API_KEY`。

如果只想先本地检查接线，不想调用 judge，可以在 `.env` 里设置：

```text
VERIFIER_PROVIDER=none
```

真实训练建议使用真实 judge。

## 3. 下载初始模型

默认训练从 Qwen3-8B 初始模型开始，模型目录约定为：

```text
models/original_model/Qwen3-8B
```

推荐用随包脚本下载：

```bash
bash scripts/download_qwen3_8b.sh
```

也可以手动下载：

```bash
.venv-verl/bin/python -m pip install huggingface_hub
.venv-verl/bin/python -m huggingface_hub.commands.huggingface_cli download Qwen/Qwen3-8B \
  --local-dir models/original_model/Qwen3-8B \
  --local-dir-use-symlinks False
```

## 4. 重新构造训练数据

数据集压缩包里已经带好 parquet（见「数据集下载」）。需要复现数据构造时运行：

```bash
bash scripts/build_training_data.sh
```

期望输出：

```text
data/sft/stage5/train.parquet       121 rows
data/sft/stage5/val.parquet          14 rows
data/rl/stage5/train.parquet       2273 rows
data/rl/eval/train.parquet          304 rows
```

## 5. 一键启动 SFT

```bash
bash scripts/run_sft_stage.sh
```

常用调试参数：

```bash
TOTAL_STEPS=1 TRAIN_MAX_SAMPLES=1 VAL_MAX_SAMPLES=1 LOGGER=console MERGE_HF=0 \
  bash scripts/run_sft_stage.sh
```

默认输出：

```text
checkpoints/sft/<run_name>/
models/<run_name>_hf/
runs/train_logs/<run_name>.log
```

SFT 默认从 `models/original_model/Qwen3-8B` 开始，不会从已有 checkpoint resume。分布式默认规模是 `NNODES=1`、`NPROC_PER_NODE=64`。

## 6. 一键启动 AgenticRL / GRPO

```bash
bash scripts/run_agenticrl_stage.sh
```

常用调试参数：

```bash
TOTAL_STEPS=1 TRAIN_MAX_SAMPLES=1 VAL_MAX_SAMPLES=1 ROLLOUT_N=1 LOGGER=console TEST_FREQ=-1 SAVE_FREQ=1 \
  bash scripts/run_agenticrl_stage.sh
```

默认输出：

```text
checkpoints/grpo/<run_name>/
data/rollouts_verl/<run_name>/
data/evals/<run_name>/
runs/train_logs/<run_name>.log
```

GRPO 默认也从 `models/original_model/Qwen3-8B` 开始，不依赖 SFT 输出。分布式默认规模是 `NNODES=1`、`N_GPUS_PER_NODE=64`。

## 7. 分布式规模和模型覆盖

默认总 GPU 数为 64。单机 64 卡可直接使用默认值；多机 8x8 时这样覆盖：

```bash
NNODES=8 NPROC_PER_NODE=8 NODE_RANK=<0-7> MASTER_ADDR=<rank0_host> \
  bash scripts/run_sft_stage.sh

NNODES=8 N_GPUS_PER_NODE=8 \
bash scripts/run_agenticrl_stage.sh
```

下载其它模型到本地目录后，也可以用 `MODEL=/path/to/model` 覆盖默认模型路径。

## 8. 训练过程看什么

SFT 主要看：

- `train/loss`
- `val/loss`
- learning rate
- checkpoint 是否正常保存

GRPO 主要看：

- `critic/rewards/mean`
- `critic/score/mean`
- `actor/loss`
- `actor/entropy`
- `actor/grad_norm`
- `response_length/mean`
- `num_turns/mean`
- `val-core/*/reward/mean@1`
- `val-aux/*/parse_error`
- `val-aux/*/tool_error_llm`
- `val-aux/*/max_step_hit`

每条 GRPO rollout 的轨迹和分数会写入：

```text
data/rollouts_verl/<run_name>/
```

训练结束后的汇总会写入：

```text
data/evals/<run_name>/
```
