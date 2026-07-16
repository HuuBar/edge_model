# WiFi Agent 训练 Pipeline 设计 v1

> 覆盖数据处理 + SFT训练 + RL训练 + 评测的全链路设计
> 基于当前数据现状（8821条原始数据）和NPU环境（4卡910B4）

---

## 一、整体架构

```
┌──────────────────────────────────────────────────────────────────────┐
│                        原始数据（8821条）                              │
│  query_en/ar/cn + intent_label + rag + react_label + categories      │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌─────────┐ ┌─────────┐ ┌─────────────┐
        │ A级数据  │ │ B级数据  │ │  C级数据     │
        │ ~1063条 │ │ ~4123条 │ │   ~2319条   │
        │ API类   │ │ 澄清类  │ │  QA类       │
        └────┬────┘ └────┬────┘ └──────┬──────┘
             │           │              │
             └───────────┼──────────────┘
                         ▼
              ┌──────────────────┐
              │  数据处理Pipeline │
              │  1. 工具名映射    │
              │  2. 四件套转换    │
              │  3. 场景分类      │
              │  4. 训练/测试划分 │
              │  5. 数据增强      │
              └────────┬─────────┘
                       ▼
              ┌──────────────────┐
              │  训练数据输出     │
              │  train.parquet   │
              │  val.parquet     │
              │  test.parquet    │
              └────────┬─────────┘
                       ▼
        ┌──────────────────────────────┐
        │       训练 Pipeline           │
        │  ┌─────────┐  ┌──────────┐  │
        │  │ SFT训练  │  │ RL训练    │  │
        │  │ 4卡NPU  │  │ 4卡NPU   │  │
        │  │ DeepS.  │  │ 最小实现  │  │
        │  └────┬────┘  └────┬─────┘  │
        │       └─────┬──────┘        │
        │             ▼               │
        │      ┌──────────┐          │
        │      │ 评测Pipeline│         │
        │      │ AgentLoop  │         │
        │      │ + Verifier │         │
        │      └──────────┘          │
        └──────────────────────────────┘
```

---

## 二、数据处理 Pipeline

### 2.1 阶段1：数据分级与筛选

**输入**: 原始JSON文件（21个文件，8821条）  
**输出**: 分级后的数据子集  
**脚本**: `data_pipeline/stage1_filter.py`

```python
# 分级逻辑
A级 = categories只含基础API工具（wifi.get_info, wifi.open等）
B级 = 含澄清类意图（clarification, continuation）
C级 = 无工具调用的纯QA（react_label为空或只有final_answer）
条件依赖 = dataset字段含"条件依赖"或"步骤间依赖"
丢弃 = 业务无关/闲聊/结构损坏
```

**当前策略**: 条件依赖类（952条）暂不处理，先用A+B+C级数据跑通pipeline。

### 2.2 阶段2：工具名映射（最关键！）

**输入**: 原始JSON中的旧工具名  
**输出**: 映射后的新工具名  
**脚本**: `data_pipeline/stage2_tool_mapping.py`

旧数据中的工具名 → 新代码工具名映射表：

| 旧名（数据中使用） | 新名（代码中定义） | 类型 |
|-------------------|-------------------|------|
| `get_traffic_statistics` | `data.get_usage` | 读 |
| `set_data_limit` | `data.set_limit` | 写 |
| `set_data_alert_threshold` | `data.set_alert_threshold` | 写 |
| `set_wifi_name` / `set_wifi_ssid` | `wifi.set_config` | 写 |
| `switch_wifi_broadcast` / `hide_ssid` | `wifi.hide_ssid` | 写 |
| `set_wifi_channel` | `wifi.set_channel` | 写 |
| `set_wifi_bandwidth` | `wifi.set_bandwidth` | 写 |
| `switch_5G_mode` | `wifi.switch_5g_mode` | 写 |
| `switch_5G_priority` | `wifi.switch_5g_priority` | 写 |
| `get_wifi_info` / `get_wifi_config` | `wifi.get_info` | 读 |
| `list_wifi_clients` / `get_client_list` | `wifi.list_clients` | 读 |
| `get_device_info` / `get_serial_number` / `get_antenna_type` | `device.get_info` | 读 |
| `get_network_status` / `get_ethernet_speed` / `get_frequency_band` | `network.get_status` | 读 |
| `get_network_settings` / `get_dns_info` / `get_module_switch` | `network.get_settings` | 读 |
| `get_system_logs` | `system.get_logs` | 读 |
| `search_policy` / `get_policy` | `policy.search` | 读 |
| `set_network_ip_mode` | `network.set_ip_mode` | 写 |
| `set_network_ip_pool` | `network.set_ip_pool` | 写 |
| `restart_device` | `device.restart` | 写 |
| `change_password` / `change_user_password` | `user.change_password` | 写 |
| `wifi.open` / `open_wifi` / `enable_wifi` | `wifi.open` | 写 |
| `wifi.close` / `close_wifi` / `disable_wifi` | `wifi.close` | 写 |

**处理C级数据（纯QA）**: C级数据无工具调用，转换为"inform/deny"类case，verifier_spec中required_side_effects为空，只保留required_response_points。

### 2.3 阶段3：原始JSON → 四件套转换

**输入**: 映射后的单条原始JSON  
**输出**: case.json + env_snapshot.json + gold.json + verifier_spec.json  
**脚本**: `data_pipeline/stage3_convert_to_quartet.py`

#### 3a. case.json

```json
{
  "case_id": "WIFI_001",
  "customer_message": "Check my data. If under 30GB, set limit to 60GB.",
  "entities": {"device_id": "DEV_001"},
  "primary_intent": "set_data_limit",
  "language": "EN"
}
```

#### 3b. env_snapshot.json

从原始数据的 `tool_result` 和 `rag.knowledge` 反推设备状态：

```json
{
  "case_id": "WIFI_001",
  "reference_now": "2026-07-15T10:00:00",
  "readonly_tables": {
    "device_info": {
      "DEV_001": {"device_id": "DEV_001", "model": "HW-5G-CPE-Pro", ...}
    },
    "wifi_config": {
      "DEV_001": {"ssid": "MyWiFi", "enabled": true, ...}
    },
    "network_status": {
      "DEV_001": {"connected": true, "signal_strength": -75, ...}
    },
    "data_usage": {
      "DEV_001": {"total_download_mb": 17700, "current_month_download_mb": 3200, ...}
    },
    "connected_clients": {},
    "network_settings": {},
    "dhcp_leases": {},
    "system_logs": {},
    "policies": [
      {"policy_id": "P_DATA_LIMIT", "topic": "data_limit", "action_allowed": true}
    ]
  },
  "policies": [...]
}
```

**关键**: data_usage中的流量值从react_label的tool_result中提取（如"总共17.3GB,当月3.2GB"）。

#### 3c. gold.json

从 `react_label_en` 直接映射：

```json
{
  "case_id": "WIFI_001",
  "gold_trajectory": {
    "namespace_id": "gold:WIFI_001:gold_001",
    "parsed_actions": [
      {
        "step": 1,
        "tool_call_id": "tc_1",
        "name": "data.get_usage",
        "arguments": {},
        "timestamp": "2026-07-15T10:00:01"
      },
      {
        "step": 2,
        "tool_call_id": "tc_2",
        "name": "data.set_limit",
        "arguments": {"limit_mb": 61440, "target": "device"},
        "timestamp": "2026-07-15T10:00:02"
      }
    ],
    "tool_observations": [
      {"tool_call_id": "tc_1", "tool_name": "data.get_usage", "ok": true,
       "result": {"total_download_mb": 17700, "current_month_download_mb": 3200, ...}},
      {"tool_call_id": "tc_2", "tool_name": "data.set_limit", "ok": true,
       "result": {"limit_id": "DL_001", "limit_mb": 61440, "status": "applied"}}
    ],
    "final_text": "I've checked your usage (3.2GB this month) and set the data limit to 60GB.",
    "tool_errors": []
  }
}
```

#### 3d. verifier_spec.json

根据工具组合自动生成：

```json
{
  "policy_required": false,
  "evidence_required": true,
  "required_read_tools": ["data.get_usage"],
  "allowed_write_tools": ["data.set_limit"],
  "required_side_effects": [
    {"id": "set_limit", "tool": "data.set_limit",
     "required_correct": {"limit_mb": "data_usage.current_month_download_mb"}}
  ],
  "forbidden_side_effects": [],
  "required_response_points": [
    {"id": "confirm_usage", "description": "告知用户当前流量使用量"},
    {"id": "confirm_limit", "description": "确认已设置新的流量限额"}
  ],
  "forbidden_text_points": [],
  "max_steps": 6,
  "version": "verifier_simple_v1"
}
```

### 2.4 阶段4：场景分类

**输入**: 四件套数据  
**输出**: 带场景标签的数据  
**脚本**: `data_pipeline/stage4_classify.py`

分类维度（从原始数据提取）：

| 维度 | 来源字段 | 类别示例 |
|------|---------|---------|
| 主意图 | `primary_intent` / `categories[0]` | wifi_config, data_limit, network_settings |
| 子意图 | `categories` 组合 | wifi_open, data_alert, password_change |
| 语言 | `language` | EN / AR / CN |
| 轮次 | `dataset` 含"单轮/多轮" | single_turn / multi_turn |
| 复杂度 | 工具调用数量 | simple(1-2) / medium(3-4) / complex(5+) |
| 是否需要澄清 | `dataset` 含"澄清" | clarification / direct |

### 2.5 阶段5：训练/测试/验证集划分

**策略**: 按场景分类分层划分，**不随机**

```python
# 划分策略
def stratified_split(data, test_ratio=0.1, val_ratio=0.1):
    """按场景分类分层划分"""
    # 1. 按主意图分组
    groups = groupby(data, key=lambda x: x["primary_intent"])
    
    # 2. 每组内按复杂度排序
    for intent, items in groups:
        items.sort(key=lambda x: x["complexity"])
        
        # 3. 每组取最后test_ratio%作为测试集（选最难的）
        #    取中间val_ratio%作为验证集
        #    其余作为训练集
        n = len(items)
        test_count = max(1, int(n * test_ratio))
        val_count = max(1, int(n * val_ratio))
        
        test_set.extend(items[-test_count:])      # 最难的进测试
        val_set.extend(items[-test_count-val_count:-test_count])  # 中间进验证
        train_set.extend(items[:-test_count-val_count])  # 其余训练
    
    return train_set, val_set, test_set
```

**为什么要这样划分**:
- 测试集选最难的样本 → 更能反映模型上限
- 每类意图都有代表 → 避免某类场景完全没测到
- 验证集用于early stopping和超参调优

### 2.6 阶段6：四件套 → SFT训练格式

**输入**: 四件套数据  
**输出**: train.parquet / val.parquet / test.parquet  
**脚本**: `data_pipeline/stage6_build_sft_parquet.py`

```python
def build_sft_row(case, env_snapshot, gold, verifier_spec, tool_schemas):
    """构建一条SFT训练样本"""
    messages = [
        # Step 1: System prompt
        {"role": "system", "content": render_prompt("system.txt", {})},
        # Step 2: User query
        {"role": "user", "content": render_prompt("step_user.txt", {
            "case": _case_context(case)
        })},
    ]
    
    # Step 3-N: 重放gold trajectory的每个tool call + observation
    observations_by_id = {obs["tool_call_id"]: obs 
                          for obs in gold["tool_observations"]}
    for action in gold["parsed_actions"]:
        tc_id = action["tool_call_id"]
        # Assistant tool call (XML格式)
        tool_call_json = json.dumps({
            "name": action["name"],
            "arguments": action.get("arguments", {})
        }, ensure_ascii=False)
        messages.append({
            "role": "assistant",
            "content": f'<tool_call>{tool_call_json}</tool_call>'
        })
        # Tool observation
        obs = observations_by_id.get(tc_id, {"ok": False, "error": "missing"})
        messages.append({
            "role": "tool",
            "content": json.dumps(project_observation_for_model(obs), ensure_ascii=False)
        })
    
    # Final: Assistant final answer
    messages.append({
        "role": "assistant",
        "content": gold["final_text"]
    })
    
    return {
        "messages": messages,
        "tools": tool_schemas,
        "case_id": case["case_id"],
        "primary_intent": case.get("primary_intent"),
        "split": "train",  # or val/test
    }
```

### 2.7 阶段7：数据增强（迭代反馈）

**触发条件**: 评测后收集bad cases

```python
def enhance_from_feedback(bad_cases, model_provider, tool_factory):
    """从评测反馈生成增强数据"""
    enhanced = []
    for case in bad_cases:
        # 1. 用当前模型重新生成trajectory
        new_trajectory = run_agent_loop(
            case=case,
            env_snapshot=case["env_snapshot"],
            model_provider=model_provider,
            tool_factory=tool_factory,
            max_steps=8
        )
        # 2. 用verifier打分
        score = score_trajectory(
            case=case, env_snapshot=case["env_snapshot"],
            verifier_spec=case["verifier_spec"],
            executed_trajectory=new_trajectory
        )
        # 3. 如果分数高于原始gold → 替换为新的gold
        if score["reward"] > case["original_gold_reward"]:
            enhanced.append({
                "case": case,
                "new_gold": new_trajectory,
                "new_score": score
            })
    return enhanced
```

---

## 三、训练 Pipeline

### 3.1 SFT 最小实现（不依赖VERL）

**方案**: 使用 `transformers.Trainer` + `accelerate` 做NPU多卡训练

**脚本**: `training/sft_train.py`

```python
# 核心训练流程
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, 
    TrainingArguments, Trainer,
    DataCollatorForSeq2Seq
)
from accelerate import Accelerator

def train_sft(
    model_path: str = "/home/edgeModelWorkspace/origin_model/Qwen3-4B",
    data_path: str = "data/output/train.parquet",
    output_dir: str = "models/sft_qwen3_4b_wifi",
    num_epochs: int = 3,
    batch_size: int = 2,      # 4B FP16在910B上单卡batch=2
    gradient_accumulation: int = 4,
    learning_rate: float = 2e-5,
    npu_devices: list[int] = [2, 3, 6, 7],  # 4卡
):
    # 1. 加载模型和分词器
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",  # DeepSpeed会自动覆盖
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # 2. 加载训练数据
    dataset = load_parquet_dataset(data_path)
    
    # 3. 数据预处理：messages → token_ids
    def preprocess(example):
        text = tokenizer.apply_chat_template(
            example["messages"], 
            tokenize=False,
            add_generation_prompt=False
        )
        return tokenizer(text, truncation=True, max_length=2048, padding="max_length")
    
    processed = dataset.map(preprocess, remove_columns=dataset.column_names)
    
    # 4. DeepSpeed配置（4卡NPU）
    ds_config = {
        "train_batch_size": batch_size * len(npu_devices) * gradient_accumulation,
        "gradient_accumulation_steps": gradient_accumulation,
        "optimizer": {"type": "AdamW", "params": {"lr": learning_rate}},
        "fp16": {"enabled": True},
        "zero_optimization": {
            "stage": 2,  # ZeRO-2适合4卡
            "offload_optimizer": {"device": "none"},  # NPU不offload
        },
    }
    
    # 5. 训练参数
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        logging_steps=10,
        save_steps=500,
        eval_steps=500,
        deepspeed=ds_config,
        # NPU特定
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )
    
    # 6. 训练
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=processed,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    )
    trainer.train()
    
    # 7. 保存
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
```

### 3.2 多卡DeepSpeed配置

**文件**: `training/ds_config_zero2.json`

```json
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": 2,
  "gradient_accumulation_steps": 4,
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 2e-5,
      "betas": [0.9, 0.999],
      "eps": 1e-8,
      "weight_decay": 0.01
    }
  },
  "scheduler": {
    "type": "WarmupLR",
    "params": {
      "warmup_min_lr": 0,
      "warmup_max_lr": 2e-5,
      "warmup_num_steps": 100
    }
  },
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 1000,
    "initial_scale_power": 16
  },
  "zero_optimization": {
    "stage": 2,
    "allgather_partitions": true,
    "allgather_bucket_size": 2e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 2e8
  },
  "gradient_clipping": 1.0,
  "wall_clock_breakdown": false
}
```

### 3.3 RL 最小实现（REINFORCE简化版）

**方案**: 不依赖VERL，用自实现的REINFORCE + 在线rollout

**脚本**: `training/rl_train_simple.py`

```python
"""简化版RL训练（REINFORCE with baseline）"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

def train_rl_simple(
    model_path: str,
    cases: list[dict],        # 评测用cases
    env_snapshots: list[dict],
    verifier_specs: list[dict],
    num_epochs: int = 10,
    rollout_n: int = 4,       # 每条case采样4条trajectory
    lr: float = 1e-6,
):
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        for case, env, spec in zip(cases, env_snapshots, verifier_specs):
            # 1. 采样N条trajectory
            rollouts = []
            for _ in range(rollout_n):
                trajectory = run_agent_loop(
                    case=case, env_snapshot=env,
                    model_provider=LocalHFProvider(model_path),
                    tool_factory=ToolFactory(), max_steps=8
                )
                score = score_trajectory(
                    case=case, env_snapshot=env,
                    verifier_spec=spec, executed_trajectory=trajectory
                )
                rollouts.append({"trajectory": trajectory, "reward": score["reward"]})
            
            # 2. 计算baseline（平均reward）
            rewards = [r["reward"] for r in rollouts]
            baseline = sum(rewards) / len(rewards)
            
            # 3. REINFORCE梯度更新
            for rollout in rollouts:
                advantage = rollout["reward"] - baseline
                if advantage > 0:  # 只更新好的trajectory
                    # 从trajectory中提取prompt+response
                    messages = build_messages_from_trajectory(rollout["trajectory"])
                    text = tokenizer.apply_chat_template(messages, tokenize=False)
                    inputs = tokenizer(text, return_tensors="pt").to(model.device)
                    
                    # 计算log_prob
                    outputs = model(**inputs, labels=inputs["input_ids"])
                    loss = outputs.loss * (-advantage)  # 梯度上升
                    
                    loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
        
        # 保存checkpoint
        model.save_pretrained(f"models/rl_epoch_{epoch}")
```

### 3.4 评测 Pipeline

**脚本**: `training/evaluate.py`

```python
def evaluate_model(model_path, test_cases, test_envs, test_specs):
    """评测模型在测试集上的表现"""
    provider = LocalHFProvider(model_path, npu_device=2, torch_dtype="float16")
    tool_factory = ToolFactory()
    results = []
    
    for case, env, spec in zip(test_cases, test_envs, test_specs):
        # 1. 跑Agent Loop
        trajectory = run_agent_loop(
            case=case, env_snapshot=env,
            provider=provider, tool_factory=tool_factory, max_steps=spec["max_steps"]
        )
        
        # 2. Verifier打分
        score = score_trajectory(
            case=case, env_snapshot=env, verifier_spec=spec,
            executed_trajectory=trajectory,
            sandbox_final_state=trajectory.get("sandbox_final_state")
        )
        
        # 3. 记录结果
        results.append({
            "case_id": case["case_id"],
            "reward": score["reward"],
            "raw_reward": score["raw_reward"],
            "subscores": score["subscores"],
            "active_caps": score["active_caps"],
            "num_tool_calls": len(trajectory.get("parsed_actions", [])),
            "has_wifi_get_info": any(a.get("name") == "wifi.get_info" 
                                     for a in trajectory.get("parsed_actions", [])),
            "has_wifi_open": any(a.get("name") == "wifi.open" 
                                 for a in trajectory.get("parsed_actions", [])),
        })
    
    # 4. 汇总统计
    rewards = [r["reward"] for r in results]
    return {
        "mean_reward": sum(rewards) / len(rewards),
        "median_reward": sorted(rewards)[len(rewards)//2],
        "tool_call_rate": sum(1 for r in results if r["num_tool_calls"] > 0) / len(results),
        "correct_sequence_rate": sum(1 for r in results 
                                     if r["has_wifi_get_info"] and r["has_wifi_open"]) / len(results),
        "details": results
    }
```

---

## 四、目录结构

```
edge_model/
├── data_pipeline/                    # 数据处理Pipeline
│   ├── __init__.py
│   ├── stage1_filter.py              # 阶段1: 数据分级筛选
│   ├── stage2_tool_mapping.py        # 阶段2: 工具名映射
│   ├── stage3_convert_to_quartet.py  # 阶段3: 四件套转换
│   ├── stage4_classify.py            # 阶段4: 场景分类
│   ├── stage5_split.py               # 阶段5: 训练/测试划分
│   ├── stage6_build_sft_parquet.py   # 阶段6: 构建SFT训练数据
│   ├── stage7_enhance.py             # 阶段7: 数据增强
│   ├── tool_name_mapping.py          # 工具名映射表
│   └── run_all.py                    # 一键运行全部pipeline
│
├── training/                         # 训练Pipeline
│   ├── sft_train.py                  # SFT训练（最小实现）
│   ├── rl_train_simple.py            # RL训练（REINFORCE简化）
│   ├── evaluate.py                   # 评测
│   ├── ds_config_zero2.json          # DeepSpeed配置
│   ├── ds_config_zero3.json          # DeepSpeed配置（8B用）
│   └── launch_sft.sh                 # SFT启动脚本
│
├── data/                             # 数据目录
│   ├── raw/                          # 原始JSON数据
│   ├── processed/                    # 处理后数据
│   │   ├── quartets/                 # 四件套（case/env/gold/verifier）
│   │   ├── classified/               # 分类后数据
│   │   └── splits/                   # 训练/测试/验证集
│   └── output/                       # 最终训练数据
│       ├── sft_train.parquet
│       ├── sft_val.parquet
│       └── sft_test.parquet
│
└── docs/
    └── pipeline_design_v1.md         # 本文档
```

---

## 五、实施路线图

### Phase 1: 数据Pipeline打通（1-2天）

| # | 任务 | 文件 | 优先级 |
|---|------|------|--------|
| 1 | 实现工具名映射表 | `tool_name_mapping.py` | P0 |
| 2 | 实现阶段1-3（筛选+映射+四件套转换） | `stage1-3` | P0 |
| 3 | 用A级数据（1063条）验证转换 | `run_all.py` | P0 |
| 4 | 实现阶段4-5（分类+划分） | `stage4-5` | P1 |
| 5 | 实现阶段6（parquet输出） | `stage6` | P1 |
| 6 | 端到端验证（原始JSON → parquet） | `run_all.py` | P1 |

### Phase 2: SFT训练打通（2-3天）

| # | 任务 | 文件 | 优先级 |
|---|------|------|--------|
| 7 | 实现SFT训练脚本（单卡） | `sft_train.py` | P0 |
| 8 | 用0.6B/1.7B验证训练能跑通 | `launch_sft.sh` | P0 |
| 9 | DeepSpeed多卡配置 | `ds_config_zero2.json` | P0 |
| 10 | 4卡NPU训练4B模型 | `launch_sft.sh` | P1 |
| 11 | 评测脚本 | `evaluate.py` | P1 |
| 12 | 训练前后对比评测 | - | P1 |

### Phase 3: RL训练打通（3-5天）

| # | 任务 | 文件 | 优先级 |
|---|------|------|--------|
| 13 | 实现简化版REINFORCE | `rl_train_simple.py` | P0 |
| 14 | 在线rollout + verifier打分 | - | P0 |
| 15 | 4卡NPU RL训练 | - | P1 |
| 16 | 评测对比 | `evaluate.py` | P1 |

### Phase 4: 数据增强与迭代（持续）

| # | 任务 | 文件 | 优先级 |
|---|------|------|--------|
| 17 | 实现数据增强 | `stage7_enhance.py` | P2 |
| 18 | B级数据加入训练 | - | P2 |
| 19 | C级数据加入训练 | - | P2 |
| 20 | 条件依赖数据重建后加入 | V4脚本 | P3 |

---

## 六、关键决策

1. **条件依赖数据**: 暂不处理，等其他数据跑通pipeline后再回来修复
2. **多语言**: 先只用英文（query_en + react_label_en），阿/中后续迭代
3. **训练框架**: SFT用transformers Trainer + DeepSpeed（不依赖VERL），RL用简化版REINFORCE
4. **模型选择**: SFT先用4B（效果/速度平衡），验证pipeline后推广到8B
5. **评测标准**: 以mean_reward + tool_call_rate + correct_sequence_rate为核心指标
