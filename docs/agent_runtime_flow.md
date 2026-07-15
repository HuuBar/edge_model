# WiFi Agent Runtime 完整流程

## 1. 整体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                     Agent Runtime 入口                           │
│              run_agent_loop(case, env, provider)                 │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 0: 环境初始化                                               │
│ ├─ 从env_snapshot构建SandboxState（5个台账）                    │
│ ├─ 从tool_factory获取22个WiFi工具定义                            │
│ ├─ 从system prompt加载工具描述+XML格式说明                       │
│ └─ 构建初始messages列表                                          │
│    [{"role":"system","content":system_prompt},                   │
│     {"role":"user","content":user_query}]                        │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 1: 大模型推理（generate）                                    │
│ ├─ provider.generate(messages, tools=TOOLS)                     │
│ ├─ tokenizer.apply_chat_template(messages, tools=TOOLS)         │
│ │   └─ Qwen3 chat template渲染tools为特殊token                   │
│ ├─ model.generate() → 输出文本                                  │
│ └─ 解析文本中的<tool_call>{...}</tool_call>                      │
└──────────┬──────────────────────────────┬───────────────────────┘
           │ 发现tool_call                 │ 无tool_call
           ▼                              ▼
┌──────────────────┐            ┌──────────────────────────┐
│ Step 2: 工具执行  │            │ Step N: 返回Final Answer  │
│ ├─ 解析工具名     │            │ 模型直接给出最终回答        │
│ ├─ 解析参数       │            │ （当前4B/8B的行为）        │
│ ├─ 校验权限       │            └──────────────────────────┘
│ ├─ 执行handler   │
│ └─ 写sandbox台账  │
└──────────┬────────┘
           │
           ▼
┌──────────────────────────────────────┐
│ Step 3: 构建Observation               │
│ ├─ 工具执行结果 → JSON                │
│ └─ 添加到messages:                    │
│    {"role":"tool",                    │
│     "content": {"result": "..."}}    │
└──────────┬───────────────────────────┘
           │
           ▼
┌──────────────────────────────────────┐
│ Step 4: 循环判断                      │
│ ├─ steps < max_steps ? → 回到Step 1  │
│ └─ steps >= max_steps → 返回最终结果  │
└──────────────────────────────────────┘
```

## 2. 模型推理内部流程（provider.generate）

```
LocalHFProvider.generate(messages, tools)
│
├─ _load() ───────────────────────────────────────────┐
│  ├─ 清理NPU显存 (gc + empty_cache + synchronize)    │
│  ├─ 自动选择精度：4B→FP16, 8B→BF16                  │
│  ├─ AutoTokenizer.from_pretrained()                  │
│  └─ AutoModelForCausalLM.from_pretrained()           │
│     └─ device_map={"": "npu:X"} 指定空闲卡          │
│                                                      │
├─ tokenizer.apply_chat_template(                     │
│      messages,                                       │
│      tools=tools,          ← 22个WiFi工具定义        │
│      add_generation_prompt=True                      │
│   )                                                  │
│   └─ 渲染为Qwen3 chat格式：                          │
│      <|im_start|>system...tools...<|im_end|>        │
│      <|im_start|>user...<|im_end|>                  │
│      <|im_start|>assistant                          │
│                                                      │
├─ model.generate()                                    │
│   └─ NPU推理 → 生成token序列                         │
│                                                      │
└─ tokenizer.decode()                                  │
    └─ 文本："<tool_call>{"tool":"wifi.get_info"}...   │
        或 "您的WiFi已恢复..."                         │
```

## 3. 工具调用解析与执行

```
解析回复文本
│
├─ 正则匹配：<tool_call>(.*?)</tool_call>
│
├─ 提取JSON：{"tool": "wifi.get_info", "params": {...}}
│
├─ 权限检查 ──────────────────────────────────────────┐
│  ├─ 工具是否在允许列表？                              │
│  ├─ 参数类型是否正确？                                │
│  └─ 是否需要先执行read工具？                          │
│                                                      │
├─ 分发执行 ──────────────────────────────────────────┤
│  │                                                    │
│  ├─ 读工具（8个）                                    │
│  │  ├─ wifi.get_info → 查wifi_config表               │
│  │  ├─ device.get_info → 查device_info表             │
│  │  ├─ network.get_status → 查network_status表       │
│  │  └─ ...                                           │
│  │                                                    │
│  └─ 写工具（14个）                                   │
│     ├─ wifi.open → 改wifi_config.enabled=true        │
│     │                + 写switch_log台账               │
│     ├─ wifi.set_channel → 校验信道范围(1-14/36-165)   │
│     └─ ...                                           │
│                                                      │
└─ 返回结果 → 加入messages进入下一轮                   │
```

## 4. 当前模型能力对比

| 模型 | 加载 | 正常输出 | 调用工具 | 原因 |
|------|------|---------|---------|------|
| 0.6B | ✅ | ✅ | ❌ 随机 | 能力太弱 |
| **1.7B** | ✅ | ✅ | **✅ 偶尔** | **小模型随机探索碰巧输出XML** |
| 4B | ✅ | ✅ | ❌ | 未被SFT训练，不会XML格式 |
| 4B-Instruct | ✅ | ✅ | ❌ | 同上，instruction tuning不含工具 |
| 8B | ✅ | ✅ | ❌ | 同上，更倾向自然语言 |

## 5. SFT训练路径（下一步）

```
Phase 1: 数据生成（用1.7B）
├─ 用1.7B跑100个WiFi case
├─ 筛选正确调工具的轨迹（如TEST_WIFI_NOT_FOUND_001）
└─ 提取 (messages_history, assistant_reply_with_tool_call) 训练对

Phase 2: SFT训练（训练4B）
├─ 加载Qwen3-4B基础权重
├─ 用Phase 1的数据做监督微调
├─ 学习目标：输出正确的<tool_call>XML格式
└─ 保存SFT后的4B模型

Phase 3: 验证
├─ 用SFT后的4B重新跑测试
├─ 预期：4B能稳定调用工具（比1.7B更可靠）
└─ 最终：4B作为主力部署模型
```

## 6. NPU资源管理

```
8张910B4 NPU (每张32GB)
│
├─ NPU 0,1,4,5: VLLM服务进程（约27GB/卡）→ 不可用
│
└─ NPU 2,3,6,7: 空闲（约30GB/卡）→ 可用
    │
    ├─ NPU 2: Qwen3-4B (FP16, 7.5GB)
    ├─ NPU 3: Qwen3-8B (BF16, 15.3GB)
    ├─ NPU 6: Qwen3-4B-Instruct (FP16, 7.5GB)
    └─ NPU 7: 预留
```
