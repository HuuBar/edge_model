# WiFi Agent Runtime 深度详解 —— 代码级完全指南

> 本文档逐行对应代码，讲解 Agent Runtime 的每个步骤在哪些文件、哪些函数中实现。
> 文末附可直接运行的测试脚本，会打印台账变化和工具调用全过程。

---

## 目录

1. [整体架构与数据流](#1-整体架构与数据流)
2. [Step 0: 环境初始化 —— 台账、工具、Prompt 如何拼装](#2-step-0-环境初始化)
3. [Step 1: 模型推理 —— provider.generate 内部发生了什么](#3-step-1-模型推理)
4. [Step 2: 工具调用解析 —— `<tool_call>` XML 如何变成可执行动作](#4-step-2-工具调用解析)
5. [Step 3: 工具执行 —— 读工具 vs 写工具的本质区别](#5-step-3-工具执行)
6. [Step 4: Observation 回放 —— 工具结果怎么回到消息里](#6-step-4-observation-回放)
7. [完整测试脚本：一步步打印台账变化和工具调用](#7-完整测试脚本)
8. [附录：核心代码速查表](#8-附录核心代码速查表)

---

## 1. 整体架构与数据流

### 1.1 核心模块与文件对应

```
edge_model/
|
|-- agent/
|   |-- runtime.py              # 主循环 run_agent_loop() —— Agent Loop  orchestrator
|   |-- observations.py         # observation_message() —— 工具结果包装成 role="tool" 消息
|   |-- prompts/
|   |   |-- system.txt          # System Prompt（身份+规则+工具调用格式+XML格式说明）
|   |   |-- step_user.txt       # 每步用户消息模板（Jinja2，含case上下文）
|   |   |-- tool_error_feedback.txt  # 工具调用格式错误时的反馈模板
|   |   |-- templates.py        # render_prompt() / prompt_hash() —— Prompt渲染工具
|   |-- providers/
|   |   |-- local_hf_provider.py # LocalHFProvider —— NPU本地推理（含FP16/BF16自动选择）
|   |-- trajectory.py           # Trajectory 类 —— 记录整条rollout轨迹
|
|-- envs/
|   |-- toolfactory.py          # ToolFactory 类 —— 工具注册表+统一执行execute()
|   |-- sandbox_state.py        # SandboxState 类 —— 可写台账管理（5个台账）
|   |-- schemas.py              # ToolDefinition / ToolArg / ToolExecutionError 数据结构
|   |-- toollist/
|   |   |-- common.py           # 工具共享实现：TOOL_SPECS + _read_handler + _write_handler
|   |   |-- wifi_get_info.py    # 薄包装：TOOL = make_tool("wifi.get_info")
|   |   |-- wifi_set_config.py  # 薄包装：TOOL = make_tool("wifi.set_config")
|   |   |-- ...（共22个薄包装）
|
|-- schemas/
|   |-- env_schema.py           # EnvSnapshotSchema / SANDBOX_KEYS / READONLY_TABLES / default_sandbox()
```

### 1.2 数据流全景图

```
                    +---------------------------+
                    |  case（工单）              |
                    |  case_id + customer_message|
                    +------------+--------------+
                                 |
                    +------------v--------------+
                    |  env_snapshot（世界状态）   |
                    |  ├─ readonly_tables（9张只读表）|
                    |  ├─ policies（策略规则）      |
                    |  ├─ sandbox_initial（5个空台账）|
                    |  └─ tool_faults（故障注入）   |
                    +------------+--------------+
                                 |
+--------------------------------v----------------------------------------+
|                           run_agent_loop()                               |
|  文件: agent/runtime.py  函数: run_agent_loop()  行号: 180-421          |
|                                                                          |
|  Step 0: 初始化                                                          |
|    ├─ SandboxState.from_env_snapshot() ── 创建可写沙盒                    |
|    ├─ tool_factory.tool_schemas() ── 生成22个工具的OpenAI schema          |
|    ├─ render_prompt("system.txt") ── 加载System Prompt                  |
|    └─ render_prompt("step_user.txt") ── 渲染首条用户消息                  |
|                                                                          |
|  Step 1: 循环（for step in max_steps）                                    |
|    ├─ provider.generate(messages, tools=tool_schemas)                    |
|    │     文件: agent/providers/local_hf_provider.py                       |
|    │     └─ model.generate() → 输出文本                                  |
|    │                                                                      |
|    ├─ parse_tool_calls(raw_text) ── 解析<tool_call>XML                   |
|    │     文件: agent/runtime.py  函数: parse_tool_calls()  行号: 48      |
|    │                                                                      |
|    ├─ 分支判断:                                                           |
|    │   A) parse_error → 构造错误反馈 → continue                          |
|    │   B) 无tool_call → final_text → break（自然终止）                    |
|    │   C) 有tool_call → 进入Step 3                                       |
|    │                                                                      |
|    ├─ tool_factory.execute(name, args, env, sandbox, context)            |
|    │     文件: envs/toolfactory.py  函数: execute()  行号: 154           |
|    │     ├─ 读工具 → _read_handler() → 查只读表 → 无副作用               |
|    │     └─ 写工具 → _write_handler() → 校验 → 落台账 → 审计日志          |
|    │                                                                      |
|    ├─ observation_message(result) ── 包装成 role="tool" 消息             |
|    │     文件: agent/observations.py  行号: 53                           |
|    │                                                                      |
|    └─ messages.append(tool_msg) ── 回放给下一步模型                        |
|                                                                          |
|  Step N: 导出                                                            |
|    ├─ trajectory.to_dict() ── 完整轨迹（含prompt_history/tool_calls/     |
|    │                          observations/final_text/sandbox_state）    |
|    └─ sandbox.export() ── sandbox_final_state（5个台账的最终内容）       |
+--------------------------------------------------------------------------+
```

---

## 2. Step 0: 环境初始化

### 2.1 只读台账 —— 9张表定义在哪里

**代码位置**: `schemas/env_schema.py`  行号: 55-65

```python
# schemas/env_schema.py

READONLY_TABLES = [
    "device_info",        # 设备基本信息（型号、固件版本、IMEI、运行时长等）
    "wifi_config",        # WiFi当前配置（SSID、密码、信道、频段、隐藏状态、加密方式等）
    "network_status",     # 网络实时状态（连接状态、信号强度、上下行速率、延迟、丢包率等）
    "connected_clients",  # 已连接客户端列表（MAC地址、IP、设备名称、连接时长、实时流量等）
    "data_usage",         # 流量使用统计（总流量、各客户端流量、本月/今日用量、剩余额度等）
    "network_settings",   # 网络高级设置（MTU、IPv6开关、UPnP、端口映射、防火墙规则等）
    "dhcp_leases",        # DHCP租约表（MAC-IP映射、租约获取时间、到期时间、主机名等）
    "system_logs",        # 系统日志（近期事件、告警、错误、操作记录等时间序列数据）
    "policies",           # 策略/规则配置（客服可操作范围、自动限速规则、黑白名单等）
]
```

这些表在 `env_snapshot["readonly_tables"]` 中，**模型只能读、不能改**。读工具（如 `wifi.get_info`）就是从这些表里查数据。

### 2.2 可写台账 —— 5个沙盒台账定义在哪里

**代码位置**: `schemas/env_schema.py`  行号: 33-39 和 68-83

```python
# schemas/env_schema.py

SANDBOX_KEYS = [
    "wifi_config_log",    # WiFi配置变更台账（wifi.set_config/set_channel/set_bandwidth等落账）
    "switch_log",         # 开关/模式切换台账（wifi.open/close/switch_5g_mode等落账）
    "data_limit_log",     # 流量限制操作台账（data.set_limit/set_alert_threshold等落账）
    "ip_config_log",      # IP配置变更台账（network.set_ip_mode/set_ip_pool等落账）
    "operation_log",      # 通用运维操作台账（device.restart/user.change_password等落账）
]

def default_sandbox() -> dict[str, Any]:
    """5个台账初始都是空列表（事件流语义：每次写append一条记录）"""
    return {
        "wifi_config_log": [],
        "switch_log": [],
        "data_limit_log": [],
        "ip_config_log": [],
        "operation_log": [],
    }
```

**事件流语义**: 每次写操作产生一条独立记录，append到对应列表。不存在"覆盖"，保留完整操作历史。

### 2.3 写工具→台账的映射

**代码位置**: `envs/sandbox_state.py`  行号: 39-101

```python
# envs/sandbox_state.py

WRITE_TOOL_FACTORS = {
    # wifi_config_log 台账 ← 4个写工具
    "wifi.set_config":     {"ledger": "wifi_config_log", "fact": "wifi_config_set"},
    "wifi.set_channel":    {"ledger": "wifi_config_log", "fact": "wifi_channel_set"},
    "wifi.set_bandwidth":  {"ledger": "wifi_config_log", "fact": "wifi_bandwidth_set"},
    "wifi.hide_ssid":      {"ledger": "wifi_config_log", "fact": "wifi_ssid_hidden"},
    # switch_log 台账 ← 4个写工具
    "wifi.open":           {"ledger": "switch_log",      "fact": "wifi_opened"},
    "wifi.close":          {"ledger": "switch_log",      "fact": "wifi_closed"},
    "wifi.switch_5g_mode":     {"ledger": "switch_log", "fact": "cellular_5g_mode_switched"},
    "wifi.switch_5g_priority": {"ledger": "switch_log", "fact": "wifi_5g_priority_switched"},
    # data_limit_log 台账 ← 2个写工具
    "data.set_limit":          {"ledger": "data_limit_log", "fact": "data_limit_set"},
    "data.set_alert_threshold":{"ledger": "data_limit_log", "fact": "data_alert_threshold_set"},
    # ip_config_log 台账 ← 2个写工具
    "network.set_ip_mode": {"ledger": "ip_config_log", "fact": "ip_mode_set"},
    "network.set_ip_pool": {"ledger": "ip_config_log", "fact": "ip_pool_set"},
    # operation_log 台账 ← 2个写工具
    "device.restart":      {"ledger": "operation_log", "fact": "device_restarted"},
    "user.change_password":{"ledger": "operation_log", "fact": "password_changed"},
}
```

**共13个写工具 → 5个台账**。每个写工具知道：
- 自己的记录落哪个 `ledger`（台账）
- 自己的 `fact` key是什么（verifier用来判断"这个动作是否已发生"）

### 2.4 工具定义 —— 22个工具在哪里注册

**第一层: 工具模块列表**

**代码位置**: `envs/toolfactory.py`  行号: 36-65

```python
# envs/toolfactory.py

TOOL_MODULES = [
    # 读工具（8个）
    "envs.toollist.wifi_get_info",        # 读取WiFi配置
    "envs.toollist.wifi_list_clients",    # 读取客户端列表
    "envs.toollist.device_get_info",      # 读取设备信息
    "envs.toollist.data_get_usage",       # 读取流量统计
    "envs.toollist.network_get_status",   # 读取网络状态
    "envs.toollist.network_get_settings", # 读取网络设置
    "envs.toollist.system_get_logs",      # 读取系统日志
    "envs.toollist.policy_search",        # 检索客服策略
    # 写工具（13个）
    "envs.toollist.wifi_set_config",      # 设置SSID/密码
    "envs.toollist.wifi_set_channel",     # 设置信道
    "envs.toollist.wifi_set_bandwidth",   # 设置带宽
    "envs.toollist.wifi_hide_ssid",       # 隐藏SSID
    "envs.toollist.wifi_open",            # 开启WiFi
    "envs.toollist.wifi_close",           # 关闭WiFi
    "envs.toollist.wifi_switch_5g_mode",      # 切换蜂窝5G模式
    "envs.toollist.wifi_switch_5g_priority",  # 切换WiFi 5GHz优选
    "envs.toollist.data_set_limit",       # 设置流量上限
    "envs.toollist.data_set_alert_threshold", # 设置告警阈值
    "envs.toollist.network_set_ip_mode",  # 设置IP模式
    "envs.toollist.network_set_ip_pool",  # 设置DHCP地址池
    "envs.toollist.device_restart",       # 重启设备
    "envs.toollist.user_change_password", # 修改管理密码
]
```

**第二层: 工具元数据（描述/权限/参数）**

**代码位置**: `envs/toollist/common.py`  行号: 664-905

```python
# envs/toollist/common.py

TOOL_SPECS = {
    "wifi.get_info": {
        "description": "读取WiFi当前配置，包括SSID、密码...",
        "permissions": ("read",),           # ← 读权限
        "args": {},                          # ← 无参数
    },
    "wifi.open": {
        "description": "开启WiFi广播...",
        "permissions": ("sandbox_write",),  # ← 写权限（会落台账）
        "args": {
            "band": arg("string", False, "要开启的频段：all/2.4G/5G"),
        },
    },
    "wifi.set_config": {
        "description": "设置WiFi的SSID、密码和加密方式...",
        "permissions": ("sandbox_write", "irreversible_action"),  # ← 写+不可逆
        "args": {
            "ssid": arg("string", True, "新的WiFi名称，1-32字节"),       # ← 必填
            "password": arg("string", False, "新密码，至少8位"),         # ← 可选
            "encryption": arg("string", False, "加密方式"),
            "band": arg("string", False, "适用频段"),
        },
    },
    # ... 共22个工具的完整定义
}
```

**第三层: 薄包装文件**

**代码位置**: `envs/toollist/wifi_get_info.py`（每个工具一个文件，只有2行）

```python
# envs/toollist/wifi_get_info.py
from envs.toollist.common import make_tool
TOOL = make_tool("wifi.get_info")  # ← 从TOOL_SPECS取出定义，组装成ToolDefinition
```

**工厂函数 make_tool**

**代码位置**: `envs/toollist/common.py`  行号: 929-951

```python
def make_tool(tool_name: str) -> ToolDefinition:
    spec = TOOL_SPECS[tool_name]
    
    def handler(args, env_snapshot, sandbox, context):
        # 统一入口：按权限分流到读handler或写handler
        return execute_named_tool(tool_name, args, env_snapshot, sandbox, context)
    
    return ToolDefinition(
        name=tool_name,
        description=spec["description"],
        permissions=spec["permissions"],
        args=spec["args"],
        handler=handler,
    )
```

### 2.5 System Prompt 如何拼装

**代码位置**: `agent/prompts/system.txt`（完整文件见上文）

System Prompt 包含：
1. **身份定义**: "你是一个WiFi随身设备客服Agent"
2. **问题类型枚举**: WiFi连接/网速/配置/流量/设备管理/网络设置/客户端管理
3. **行为规则**: 每步只输出一个tool call、先调查再回复、工具调用格式等
4. **XML格式说明**: `<tool_call>{"name": "...", "arguments": {...}}</tool_call>`
5. **安全约束**: device.restart预告断网、wifi.close高风险等

**拼装入口**:

**代码位置**: `agent/runtime.py`  行号: 239

```python
system_text = render_prompt("system.txt", {})  # Jinja2渲染，当前无变量
```

### 2.6 初始 Messages 如何拼装

**代码位置**: `agent/runtime.py`  行号: 246-249

```python
messages = [
    {"role": "system", "content": system_text},   # ← 身份+规则+XML格式
    {"role": "user",   "content": step_text},     # ← 客户消息+可见上下文
]
```

`step_text` 由 `render_prompt("step_user.txt", {"case": _case_context(case)})` 渲染，包含：
- ticket_id（脱敏后的case id）
- customer_message（客户原始诉求）
- 其他可见上下文（customer_id、market等）

---

## 3. Step 1: 模型推理

### 3.1 生成入口

**代码位置**: `agent/runtime.py`  行号: 279

```python
output = provider.generate(
    messages,                          # 当前完整对话历史
    sampling_config=sampling_config,   # temperature=0.7, top_p=0.9等
    tools=tool_schemas,                # 22个工具的OpenAI function schema
)
```

### 3.2 provider.generate 内部流程

**代码位置**: `agent/providers/local_hf_provider.py`  行号: 169-233

```python
def generate(self, messages_or_prompt, sampling_config=None, tools=None):
    # 1. 惰性加载模型（首次调用时）
    self._load()  # → AutoTokenizer + AutoModelForCausalLM.from_pretrained()
    
    # 2. 用chat template渲染messages+tools为模型输入文本
    prompt = self._tokenizer.apply_chat_template(
        messages,              # 完整消息列表
        tools=tools,           # 22个工具schema
        tokenize=False,        # 返回字符串而非token ids
        add_generation_prompt=True,  # 追加<|im_start|>assistant引导生成
    )
    # 输出示例:
    # <|im_start|>system
    # 你是一个WiFi随身设备客服Agent...
    # 
    # # Tools
    # ## default
    # namespace default {
    # ...（22个工具的schema描述）...
    # }
    # <|im_end|>
    # <|im_start|>user
    # 客户消息: 我手机搜不到WiFi信号了，帮我看看
    # ...
    # <|im_end|>
    # <|im_start|>assistant    ← add_generation_prompt=True追加的这一行
    
    # 3. Tokenize并搬到NPU
    inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
    
    # 4. model.generate() 生成
    output_ids = self._model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    
    # 5. 解码输出（只保留新生成的token）
    generated = output_ids[0][inputs["input_ids"].shape[-1]:]
    text = self._tokenizer.decode(generated, skip_special_tokens=False)
    
    return ModelOutput(raw_text=text, ...)
```

### 3.3 Qwen3 Chat Template 如何渲染 Tools

当 `tokenizer.apply_chat_template(messages, tools=tools)` 被调用时：

1. 把22个工具的 `to_tool_schema()` 输出格式化为模型能理解的文本
2. 在system prompt后追加 `# Tools` 段落，列出每个工具的：
   - 工具名
   - 描述
   - 参数（类型、是否必填、描述）
3. 渲染成 Qwen3 的对话格式（`<|im_start|>role...<|im_end|>`）

模型因此"知道"有哪些工具可用、每个工具做什么、需要什么参数。

---

## 4. Step 2: 工具调用解析

### 4.1 从模型输出文本中提取 `<tool_call>`

**代码位置**: `agent/runtime.py`  行号: 48-87

```python
# 正则匹配 <tool_call>...</tool_call>，re.DOTALL允许跨行
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)

def parse_tool_calls(text: str) -> tuple[list[dict], str | None]:
    matches = list(TOOL_CALL_RE.finditer(text))
    if not matches:
        return [], None  # ← 没有tool_call，说明是最终回复
    
    tool_calls = []
    for match in matches:
        # 提取标签内的JSON
        payload = parse_json_object_fragment(match.group(1))
        name = payload.get("name")
        arguments = payload.get("arguments", {})
        tool_calls.append({"name": name, "arguments": arguments})
    return tool_calls, None
```

**三种输出情况**:

| 模型输出 | parse_tool_calls 返回值 | 含义 |
|---------|----------------------|------|
| 纯文本（无XML标签） | `([], None)` | 最终回复 → break终止 |
| `<tool_call>坏JSON</tool_call>` | `([], "invalid_tool_call_json: ...")` | 格式错误 → parse_error反馈 |
| `<tool_call>{"name":"...",...}</tool_call>` | `([{"name":"..."}], None)` | 有效工具调用 → 执行 |

### 4.2 双解析路径

**代码位置**: `agent/runtime.py`  行号: 294-300

```python
# 路径1: provider原生function-calling（如API provider）
tool_calls = output.tool_calls if output.tool_calls else []

# 路径2: 从raw_text解析XML（本地HF provider走这条）
if not tool_calls:
    tool_calls, parse_error = parse_tool_calls(output.raw_text)
```

本地模型（LocalHFProvider）输出的是纯文本，需要从文本中解析XML标签。

---

## 5. Step 3: 工具执行 —— 读工具 vs 写工具的本质区别

### 5.1 执行入口

**代码位置**: `envs/toolfactory.py`  行号: 154-210

```python
def execute(self, tool_name, arguments, env_snapshot, sandbox, context):
    tool = self.get(tool_name)           # 1) 从注册表取工具定义
    tool.validate_args(arguments)        # 2) 校验参数（必填+未知参数检查）
    
    injected = self._maybe_fault(...)    # 3) 故障注入（环境主动制造错误）
    if injected: return injected
    
    if tool.handler is None:
        raise ToolExecutionError("tool_not_implemented")
    
    result = tool.handler(arguments, env_snapshot, sandbox, context)  # 4) 执行
    return {"ok": True, "result": result, ...}
```

### 5.2 读工具 vs 写工具 —— 核心区别

**分流逻辑**:

**代码位置**: `envs/toollist/common.py`  行号: 912-926

```python
def execute_named_tool(tool_name, args, env_snapshot, sandbox, context):
    if "sandbox_write" in TOOL_SPECS[tool_name]["permissions"]:
        return _write_handler(tool_name, args, env_snapshot, sandbox, context)
    return _read_handler(tool_name, args, env_snapshot, sandbox)
```

| 维度 | 读工具（8个） | 写工具（13个） |
|------|------------|-------------|
| 权限标记 | `permissions = ("read",)` | `permissions = ("sandbox_write",)` 可能还有 `"irreversible_action"` |
| Handler | `_read_handler()` | `_write_handler()` |
| 数据来源 | `env_snapshot["readonly_tables"]` | 参数 + env_snapshot（只读校验） |
| 副作用 | **无** —— 只查不记 | **有** —— 写sandbox台账 + 审计日志 |
| 返回 | 查询结果dict | 操作结果dict + 台账记录 |
| 示例 | `wifi.get_info` → 查wifi_config表 | `wifi.open` → 写switch_log台账 |

### 5.3 读工具详解

**代码位置**: `envs/toollist/common.py`  行号: 173-273

```python
def _read_handler(tool_name, args, env, sandbox):
    if tool_name == "wifi.get_info":
        config = _table(env, "wifi_config")  # ← 从readonly_tables查
        if not config:
            raise ToolExecutionError("wifi_config_not_found")
        return _copy(config)  # ← 深拷贝返回，防止调用方修改
    
    if tool_name == "wifi.list_clients":
        clients = _table(env, "connected_clients")
        # 支持按mac_address/ip_address/band过滤
        result = {"clients": [...], "total_count": N}
        return result
    
    # ... 其他读分支
```

**读工具的本质**: 从 `env_snapshot["readonly_tables"][表名]` 查数据，不做任何修改。

### 5.4 写工具详解

**代码位置**: `envs/toollist/common.py`  行号: 281-651

```python
def _write_handler(tool_name, args, env, sandbox, context):
    # --- wifi.open 示例 ---
    if tool_name == "wifi.open":
        # 1. 前置校验：读取当前WiFi状态
        wifi_config = _table(env, "wifi_config")
        if wifi_config and wifi_config.get("enabled") is True:
            raise ToolExecutionError("wifi_already_open")  # ← 已是开启状态
        
        band = args.get("band", "all")
        
        # 2. 构造记录
        record = {
            "switch_id": _id("ON", context, band),  # 生成确定性ID
            "action": "open",
            "band": band,
            "status": "enabled",
        }
        
        # 3. 写台账（关键！）
        _write(sandbox, tool_name, record, context, 
               device_id=device_id, 
               audit_action="open_wifi", 
               audit_result="enabled")
        
        # 4. 返回操作结果
        return {"switch_id": ..., "action": "open", "band": band, "status": "enabled"}
```

### 5.5 写台账的内部流程

**代码位置**: `envs/sandbox_state.py`  行号: 144-217

```python
def write_record(self, tool_name, record, *, namespace_id, run_id, case_id, rollout_id, tool_call_id):
    # 1. 查映射表：这个写工具对应哪个台账
    mapping = WRITE_TOOL_FACTORS[tool_name]  # e.g. {"ledger": "switch_log", "fact": "wifi_opened"}
    ledger = mapping["ledger"]   # "switch_log"
    fact = mapping["fact"]       # "wifi_opened"
    
    # 2.  enriched：业务字段 + 溯源字段 + 事实标记
    enriched = {
        **record,                          # 业务字段（action/band/status等）
        "tool": tool_name,                 # 哪个工具写的
        "namespace_id": namespace_id,      # 隔离键
        "run_id": run_id, "case_id": case_id, "rollout_id": rollout_id,
        "tool_call_id": tool_call_id,
        fact: True,                        # wifi_opened=True（verifier读这个判断动作已发生）
        "verified_fact_key": fact,         # "wifi_opened"
    }
    
    # 3. 事件流语义：append到对应台账列表
    self.state.setdefault(ledger, []).append(enriched)
    # 例如：self.state["switch_log"].append({... enriched ...})
    
    # 4. 审计日志：所有写操作都记录到operation_log
    audit_entry = {
        "tool": tool_name, "action": "open_wifi", 
        "args": {...}, "result": "enabled", 
        "namespace_id": namespace_id, ...
    }
    self.state.setdefault("operation_log", []).append(audit_entry)
    
    return enriched
```

**写一次台账 = append到目标ledger + append到operation_log审计**。两个列表各多一条记录。

---

## 6. Step 4: Observation 回放 —— 工具结果怎么回到消息里

### 6.1 Observation 投影（模型可见 vs 审计字段）

**代码位置**: `agent/observations.py`  行号: 16-50

```python
# 模型允许看到的字段白名单
MODEL_VISIBLE_OBSERVATION_KEYS = {
    "ok",      # 是否成功
    "result",  # 成功时的业务结果
    "error",   # 失败时的错误码
    "message", # 错误说明
    "source",  # 错误来源（environment/llm/runtime）
}

def project_observation_for_model(observation):
    """把完整observation投影成模型下一步允许看到的内容"""
    projected = {
        key: observation[key] 
        for key in MODEL_VISIBLE_OBSERVATION_KEYS 
        if key in observation
    }
    projected["tool_name"] = observation.get("tool_name")
    projected["tool_call_id"] = observation.get("tool_call_id")
    return projected
    # 隐藏的字段：namespace_id, run_id, case_id, rollout_id等审计字段
```

### 6.2 包装成 role="tool" 消息

**代码位置**: `agent/observations.py`  行号: 53-73

```python
def observation_message(observation):
    """把observation包装成chat message格式"""
    return {
        "role": "tool",                           # ← tool role
        "tool_call_id": observation.get("tool_call_id"),
        "name": observation.get("tool_name"),     # "wifi.get_info"
        "content": project_observation_for_model(observation),  # 投影后的结果
    }
    # Qwen3 chat template会把tool role渲染成 <tool_response>{...}</tool_response>
```

### 6.3 追加到 Messages

**代码位置**: `agent/runtime.py`  行号: 411

```python
messages.append(observation_message(observation))
```

追加后的messages结构示例：

```python
[
    # 初始
    {"role": "system", "content": "你是WiFi客服Agent..."},
    {"role": "user",   "content": "客户: 我手机搜不到WiFi了..."},
    
    # 第1步：模型调用wifi.get_info
    {"role": "assistant", "content": '<tool_call>{"name":"wifi.get_info","arguments":{}}</tool_call>'},
    
    # 工具结果回放
    {"role": "tool", "name": "wifi.get_info", "tool_call_id": "tc_1",
     "content": {"ok": True, "result": {"ssid": "MyWiFi_5G", "enabled": False, ...}}},
    
    # 第2步：模型调用wifi.open
    {"role": "assistant", "content": '<tool_call>{"name":"wifi.open","arguments":{"band":"all"}}</tool_call>'},
    
    # 工具结果回放
    {"role": "tool", "name": "wifi.open", "tool_call_id": "tc_2",
     "content": {"ok": True, "result": {"action": "open", "status": "enabled"}}},
    
    # 第3步：模型给出最终回复
    {"role": "assistant", "content": "您的WiFi已恢复广播，请检查手机..."},
]
```

---

## 7. 完整测试脚本

以下脚本可直接运行，会**一步步打印台账变化和工具调用全过程**。

**保存为**: `test_agent_loop_trace.py`

```python
#!/usr/bin/env python3
"""
Agent Runtime 逐步追踪测试

功能：
1. 初始化环境（展示5个空台账 + 9张只读表）
2. 手动模拟模型调用工具（绕过模型推理，直接注入tool_call）
3. 每步打印：messages变化、台账变化、工具执行结果
4. 最终导出完整sandbox状态

运行方式:
    cd /home/z50061485/edge_model
    python test_agent_loop_trace.py
"""

import json
import sys
import os

# 添加到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from envs.toolfactory import ToolFactory
from envs.sandbox_state import SandboxState
from envs.namespace import build_namespace_id


# ============================================================================
# 测试用例：WiFi搜不到 → 查配置 → 开WiFi
# ============================================================================

TEST_CASE = {
    "case_id": "TRACE_TEST_001",
    "customer_message": "我手机搜不到WiFi信号了，帮我看看",
    "entities": {"device_id": "DEV_TRACE_001"},
    "primary_intent": "wifi_not_found",
}

TEST_ENV = {
    "case_id": "TRACE_TEST_001",
    "reference_now": "2026-07-15T10:00:00",
    "readonly_tables": {
        "device_info": {
            "DEV_TRACE_001": {
                "device_id": "DEV_TRACE_001",
                "model": "HW-5G-CPE-Pro",
                "firmware_version": "V3.2.1",
                "imei": "860000011112222",
                "uptime_seconds": 86400,
            }
        },
        "wifi_config": {
            "DEV_TRACE_001": {
                "device_id": "DEV_TRACE_001",
                "ssid": "MyWiFi_5G",
                "password": "********",
                "encryption": "WPA2-PSK",
                "channel": 36,
                "band": "5G",
                "bandwidth": "80MHz",
                "hidden": False,
                "enabled": False,  # WiFi被关闭了！
                "max_clients": 32,
            }
        },
        "network_status": {
            "DEV_TRACE_001": {
                "device_id": "DEV_TRACE_001",
                "connected": True,
                "signal_strength": -75,
                "rsrp": -85,
                "sinr": 15,
                "download_speed_kbps": 51200,
                "upload_speed_kbps": 10240,
                "latency_ms": 35,
                "packet_loss_percent": 0.5,
                "network_type": "5G",
            }
        },
        "connected_clients": {},
        "data_usage": {
            "DEV_TRACE_001": {
                "device_id": "DEV_TRACE_001",
                "total_upload_mb": 10240,
                "total_download_mb": 51200,
                "current_month_upload_mb": 2048,
                "current_month_download_mb": 10240,
                "remaining_quota_mb": 20480,
            }
        },
    },
    "policies": [
        {
            "policy_id": "P_WIFI_OPEN",
            "topic": "wifi_switch",
            "device_model": "HW-5G-CPE-Pro",
            "action_allowed": True,
        }
    ],
}


def print_header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def print_json(label, data):
    print(f"\n--- {label} ---")
    print(json.dumps(data, ensure_ascii=False, indent=2))


def print_sandbox(sandbox, label="当前台账状态"):
    """打印5个台账的当前内容"""
    print(f"\n>>> {label}")
    state = sandbox.state
    for key in ["wifi_config_log", "switch_log", "data_limit_log", "ip_config_log", "operation_log"]:
        records = state.get(key, [])
        print(f"  [{key}]: {len(records)} 条记录")
        for i, r in enumerate(records):
            tool = r.get('tool', 'N/A')
            fact = r.get('verified_fact_key', 'N/A')
            print(f"    [{i}] tool={tool}, fact={fact}")


def simulate_tool_call(tool_factory, tool_name, arguments, env, sandbox, context):
    """
    模拟一次工具调用，打印全过程。
    
    这相当于模型输出 <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    后，runtime调用tool_factory.execute()的执行过程。
    """
    print(f"\n  [模型调用] {tool_name}({json.dumps(arguments, ensure_ascii=False)})")
    
    # 执行工具
    observation = tool_factory.execute(
        tool_name,
        arguments,
        env_snapshot=env,
        sandbox=sandbox,
        context=context,
    )
    
    # 打印结果
    ok = observation.get("ok")
    if ok:
        print(f"  [执行结果] ✅ 成功")
        print(f"             result = {json.dumps(observation.get('result'), ensure_ascii=False, indent=2)[:200]}")
    else:
        print(f"  [执行结果] ❌ 失败")
        print(f"             error = {observation.get('error')}")
        print(f"             message = {observation.get('message')}")
        print(f"             source = {observation.get('source')}")
    
    return observation


def main():
    print_header("Agent Runtime 逐步追踪测试")
    print(f"\n测试Case: {TEST_CASE['case_id']}")
    print(f"客户诉求: {TEST_CASE['customer_message']}")
    
    # ========================================================================
    # Step 0: 环境初始化
    # ========================================================================
    print_header("Step 0: 环境初始化")
    
    # 0.1 创建ToolFactory（加载22个工具）
    print("\n[0.1] 创建 ToolFactory，加载22个WiFi工具...")
    tool_factory = ToolFactory()
    tool_names = sorted(tool_factory.tools.keys())
    print(f"      已加载 {len(tool_names)} 个工具:")
    for name in tool_names:
        tool = tool_factory.tools[name]
        perm = "写" if tool.is_write else "读"
        print(f"        - {name} ({perm}): {tool.description[:50]}...")
    
    # 0.2 创建SandboxState（5个空台账）
    print("\n[0.2] 创建 SandboxState（5个空台账）...")
    namespace_id = build_namespace_id("run_trace", "TRACE_TEST_001", "rollout_0001")
    sandbox = SandboxState.from_env_snapshot(TEST_ENV, namespace_id)
    print_sandbox(sandbox, "初始状态（全部为空）")
    
    # 0.3 上下文（runtime注入的字段）
    context = {
        "run_id": "run_trace",
        "case_id": "TRACE_TEST_001",
        "rollout_id": "rollout_0001",
        "namespace_id": namespace_id,
        "tool_call_id": "tc_1",
    }
    
    # 0.4 工具Schema（下发给模型的）
    print("\n[0.3] 生成 tool schemas（下发给模型的22个工具定义）...")
    schemas = tool_factory.tool_schemas()
    print(f"      共 {len(schemas)} 个schema")
    for s in schemas:
        name = s["function"]["name"]
        args = list(s["function"]["parameters"]["properties"].keys())
        print(f"        - {name}: args={args if args else '无'}")
    
    # ========================================================================
    # Step 1: 模型调用 wifi.get_info（读工具）
    # ========================================================================
    print_header("Step 1: 模型调用 wifi.get_info（读工具）")
    print("说明: 客户说搜不到WiFi，模型应该先查当前WiFi配置")
    
    context["tool_call_id"] = "tc_1"
    obs1 = simulate_tool_call(
        tool_factory, "wifi.get_info", {}, TEST_ENV, sandbox, context
    )
    print_json("wifi.get_info 返回的完整observation", obs1)
    print_sandbox(sandbox, "读工具执行后（台账不变，读工具无副作用）")
    
    # ========================================================================
    # Step 2: 模型调用 wifi.open（写工具）
    # ========================================================================
    print_header("Step 2: 模型调用 wifi.open（写工具）")
    print("说明: 查完发现WiFi是关闭的(enabled=False)，模型应该开启WiFi")
    
    context["tool_call_id"] = "tc_2"
    obs2 = simulate_tool_call(
        tool_factory, "wifi.open", {"band": "all"}, TEST_ENV, sandbox, context
    )
    print_json("wifi.open 返回的完整observation", obs2)
    print_sandbox(sandbox, "写工具执行后（switch_log + operation_log 各增加1条）")
    
    # ========================================================================
    # Step 3: 展示台账详情
    # ========================================================================
    print_header("Step 3: 台账详情分析")
    
    print("\n--- switch_log 台账内容 ---")
    for i, r in enumerate(sandbox.state.get("switch_log", [])):
        print(f"  记录[{i}]:")
        print(f"    tool: {r['tool']}")
        print(f"    action: {r.get('action')}")
        print(f"    band: {r.get('band')}")
        print(f"    status: {r.get('status')}")
        print(f"    wifi_opened: {r.get('wifi_opened')}")  # fact=True
        print(f"    verified_fact_key: {r.get('verified_fact_key')}")
        print(f"    namespace_id: {r.get('namespace_id')}")
    
    print("\n--- operation_log 审计日志内容 ---")
    for i, r in enumerate(sandbox.state.get("operation_log", [])):
        print(f"  记录[{i}]:")
        print(f"    tool: {r['tool']}")
        print(f"    action: {r.get('action')}")
        print(f"    result: {r.get('result')}")
    
    # ========================================================================
    # Step 4: 重复调用（测试幂等性和错误处理）
    # ========================================================================
    print_header("Step 4: 再次调用 wifi.open（测试幂等性）")
    print("说明: WiFi已经开启了，再次调用应该报错")
    
    context["tool_call_id"] = "tc_3"
    obs4 = simulate_tool_call(
        tool_factory, "wifi.open", {"band": "all"}, TEST_ENV, sandbox, context
    )
    print_json("重复调用的observation", obs4)
    print_sandbox(sandbox, "报错后台账不变（没有新记录产生）")
    
    # ========================================================================
    # Step 5: 调用带参数校验的写工具
    # ========================================================================
    print_header("Step 5: 调用 wifi.set_channel（测试参数校验）")
    
    # 5a: 缺少必填参数
    print("\n  [5a] 缺少必填参数 channel...")
    context["tool_call_id"] = "tc_4a"
    obs5a = simulate_tool_call(
        tool_factory, "wifi.set_channel", {}, TEST_ENV, sandbox, context
    )
    
    # 5b: 无效信道
    print("\n  [5b] 无效信道号 99...")
    context["tool_call_id"] = "tc_4b"
    obs5b = simulate_tool_call(
        tool_factory, "wifi.set_channel", {"channel": 99, "band": "2.4G"}, TEST_ENV, sandbox, context
    )
    
    # 5c: 正确调用
    print("\n  [5c] 正确调用 channel=6, band=2.4G...")
    context["tool_call_id"] = "tc_4c"
    obs5c = simulate_tool_call(
        tool_factory, "wifi.set_channel", {"channel": 6, "band": "2.4G"}, TEST_ENV, sandbox, context
    )
    print_sandbox(sandbox, "正确调用后（wifi_config_log + operation_log 各增加1条）")
    
    # ========================================================================
    # Step 6: 导出最终sandbox状态
    # ========================================================================
    print_header("Step 6: 最终 sandbox 导出")
    final_state = sandbox.export()
    print_json("sandbox_final_state", final_state)
    
    # 统计
    print("\n--- 统计 ---")
    for key in SANDBOX_KEYS:
        count = len(final_state.get(key, []))
        print(f"  {key}: {count} 条记录")
    
    # ========================================================================
    # Step 7: 验证工具查询
    # ========================================================================
    print_header("Step 7: 查询已执行的写工具")
    executed = sandbox.executed_write_tools(namespace_id)
    print(f"本rollout执行过的写工具: {sorted(executed)}")
    
    # 查询具体工具的台账记录
    print("\n--- wifi.open 的记录 ---")
    records = sandbox.records_for_tool("wifi.open", namespace_id)
    print(f"共 {len(records)} 条")
    for r in records:
        print(f"  band={r.get('band')}, status={r.get('status')}, wifi_opened={r.get('wifi_opened')}")
    
    print("\n--- wifi.set_channel 的记录 ---")
    records = sandbox.records_for_tool("wifi.set_channel", namespace_id)
    print(f"共 {len(records)} 条")
    for r in records:
        print(f"  band={r.get('band')}, channel={r.get('channel')}, status={r.get('status')}")
    
    print_header("测试完成")
    print("""
总结:
- 读工具（wifi.get_info）: 查只读表，无副作用，台账不变
- 写工具（wifi.open）: 校验 → 落switch_log台账 → 审计日志，台账+2条
- 参数校验: 缺必填参数/无效值会报错，台账不变
- 重复调用: 业务校验失败会报错（如wifi_already_open），台账不变
- 所有写操作都带namespace_id隔离，支持并发rollout
""")


if __name__ == "__main__":
    main()
```

### 运行方式

```bash
cd /home/z50061485/edge_model
python test_agent_loop_trace.py 2>&1 | tee /tmp/agent_loop_trace_$(date +%m%d_%H%M).log
```

### 预期输出（摘要）

```
======================================================================
  Agent Runtime 逐步追踪测试
======================================================================

测试Case: TRACE_TEST_001
客户诉求: 我手机搜不到WiFi信号了，帮我看看

======================================================================
  Step 0: 环境初始化
======================================================================

[0.1] 创建 ToolFactory，加载22个WiFi工具...
      已加载 22 个工具:
        - data.get_usage (读): 读取流量使用统计...
        - data.set_alert_threshold (写): 设置流量告警阈值...
        - data.set_limit (写): 设置流量上限...
        - device.get_info (读): 读取设备基本信息...
        - device.restart (写): 重启设备...
        - network.get_settings (读): 读取网络高级设置...
        - network.get_status (读): 读取网络实时状态...
        - network.set_ip_mode (写): 设置IP分配模式...
        - network.set_ip_pool (写): 设置DHCP地址池范围...
        - policy.search (读): 在客服策略库中检索适用政策...
        - system.get_logs (读): 读取系统日志...
        - user.change_password (写): 修改设备管理密码...
        - wifi.close (写): 关闭WiFi广播...
        - wifi.get_info (读): 读取WiFi当前配置...
        - wifi.hide_ssid (写): 设置是否隐藏WiFi SSID...
        - wifi.list_clients (读): 读取已连接客户端列表...
        - wifi.open (写): 开启WiFi广播...
        - wifi.set_bandwidth (写): 设置WiFi频带宽度...
        - wifi.set_channel (写): 设置WiFi信道号...
        - wifi.set_config (写): 设置WiFi的SSID、密码和加密方式...
        - wifi.switch_5g_mode (写): 切换蜂窝移动网络的5G注册模式...
        - wifi.switch_5g_priority (写): 开关WiFi的5GHz频段优选功能...

[0.2] 创建 SandboxState（5个空台账）...
>>> 初始状态（全部为空）
  [wifi_config_log]: 0 条记录
  [switch_log]: 0 条记录
  [data_limit_log]: 0 条记录
  [ip_config_log]: 0 条记录
  [operation_log]: 0 条记录

======================================================================
  Step 1: 模型调用 wifi.get_info（读工具）
======================================================================
说明: 客户说搜不到WiFi，模型应该先查当前WiFi配置

  [模型调用] wifi.get_info({})
  [执行结果] 成功
             result = {"ssid": "MyWiFi_5G", "enabled": False, ...}

>>> 读工具执行后（台账不变，读工具无副作用）
  [wifi_config_log]: 0 条记录
  [switch_log]: 0 条记录
  ...

======================================================================
  Step 2: 模型调用 wifi.open（写工具）
======================================================================
说明: 查完发现WiFi是关闭的，模型应该开启WiFi

  [模型调用] wifi.open({"band": "all"})
  [执行结果] 成功
             result = {"action": "open", "status": "enabled"}

>>> 写工具执行后（switch_log + operation_log 各增加1条）
  [wifi_config_log]: 0 条记录
  [switch_log]: 1 条记录
    [0] tool=wifi.open, fact=wifi_opened
  [data_limit_log]: 0 条记录
  [ip_config_log]: 0 条记录
  [operation_log]: 1 条记录
    [0] tool=wifi.open, fact=N/A

======================================================================
  Step 4: 再次调用 wifi.open（测试幂等性）
======================================================================
说明: WiFi已经开启了，再次调用应该报错

  [模型调用] wifi.open({"band": "all"})
  [执行结果] 失败
             error = wifi_already_open

>>> 报错后台账不变
  [switch_log]: 1 条记录  ← 没有增加！
```

---

## 8. 附录：核心代码速查表

| 功能 | 文件 | 行号 | 函数/类 |
|------|------|------|---------|
| Agent主循环 | `agent/runtime.py` | 180-421 | `run_agent_loop()` |
| tool_call解析 | `agent/runtime.py` | 48-87 | `parse_tool_calls()` |
| 工具执行 | `envs/toolfactory.py` | 154-210 | `ToolFactory.execute()` |
| 工具注册 | `envs/toolfactory.py` | 36-65 | `TOOL_MODULES` |
| 读工具handler | `envs/toollist/common.py` | 173-273 | `_read_handler()` |
| 写工具handler | `envs/toollist/common.py` | 281-651 | `_write_handler()` |
| 工具定义工厂 | `envs/toollist/common.py` | 929-951 | `make_tool()` |
| 工具元数据 | `envs/toollist/common.py` | 664-905 | `TOOL_SPECS` |
| 可写台账定义 | `schemas/env_schema.py` | 33-39 | `SANDBOX_KEYS` |
| 只读表定义 | `schemas/env_schema.py` | 55-65 | `READONLY_TABLES` |
| 空台账初始化 | `schemas/env_schema.py` | 68-83 | `default_sandbox()` |
| 写工具→台账映射 | `envs/sandbox_state.py` | 39-101 | `WRITE_TOOL_FACTORS` |
| 写台账 | `envs/sandbox_state.py` | 144-217 | `SandboxState.write_record()` |
| 查已执行工具 | `envs/sandbox_state.py` | 255-276 | `SandboxState.executed_write_tools()` |
| observation包装 | `agent/observations.py` | 53-73 | `observation_message()` |
| observation投影 | `agent/observations.py` | 30-50 | `project_observation_for_model()` |
| System Prompt | `agent/prompts/system.txt` | 1-50 | 完整prompt |
| Prompt渲染 | `agent/prompts/templates.py` | - | `render_prompt()` |
| 模型推理 | `agent/providers/local_hf_provider.py` | 169-233 | `LocalHFProvider.generate()` |
| Trajectory记录 | `agent/trajectory.py` | - | `Trajectory` 类 |
| 工具定义数据结构 | `envs/schemas.py` | 58-127 | `ToolDefinition` 类 |
| 错误定义 | `envs/schemas.py` | 16-40 | `ToolExecutionError` 类 |
