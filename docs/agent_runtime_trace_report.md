# Agent Runtime 逐步追踪测试报告

**测试时间**: 2026-07-15  
**测试脚本**: `test_agent_loop_trace.py`  
**测试场景**: WiFi搜不到 → 查配置 → 开WiFi → 调信道  
**核心目标**: 验证读工具/写工具的执行机制、台账变化规律、参数校验逻辑

---

## 一、测试环境初始化

### 1.1 工具加载 —— 22个WiFi工具全量注册

运行 `ToolFactory()` 时，系统从 `TOOL_MODULES` 列表动态import 22个工具模块，每个模块通过 `make_tool()` 工厂函数组装成完整的 `ToolDefinition`。

**加载结果**:

| 类型 | 数量 | 工具名 |
|------|------|--------|
| 读工具 | 8 | `wifi.get_info`, `wifi.list_clients`, `device.get_info`, `data.get_usage`, `network.get_status`, `network.get_settings`, `system.get_logs`, `policy.search` |
| 写工具 | 14 | `wifi.set_config`, `wifi.set_channel`, `wifi.set_bandwidth`, `wifi.hide_ssid`, `wifi.open`, `wifi.close`, `wifi.switch_5g_mode`, `wifi.switch_5g_priority`, `data.set_limit`, `data.set_alert_threshold`, `network.set_ip_mode`, `network.set_ip_pool`, `device.restart`, `user.change_password` |

每个工具的定义包含：中文描述（模型可见）、权限标签（`read` 或 `sandbox_write`）、参数schema（类型/必填/描述）。

### 1.2 沙盒初始化 —— 5个空台账

运行 `SandboxState.from_env_snapshot()` 时，系统以 `default_sandbox()` 的空骨架打底，创建5个空列表作为可写台账。

**初始状态**:

```
[wifi_config_log]: 0 条记录   ← WiFi配置变更台账
[switch_log]:      0 条记录   ← 开关/模式切换台账
[data_limit_log]:  0 条记录   ← 流量限制操作台账
[ip_config_log]:   0 条记录   ← IP配置变更台账
[operation_log]:   0 条记录   ← 运维操作审计台账
```

所有台账初始为空列表，采用**事件流语义**：每次写操作append一条记录，保留完整历史，不存在覆盖。

### 1.3 只读环境数据 —— 关键发现：WiFi被关闭了

`env_snapshot["readonly_tables"]` 中预设了设备数据。`wifi_config` 表的关键字段：

```json
{
  "ssid": "MyWiFi_5G",
  "enabled": false,      // ← WiFi当前是关闭状态！
  "hidden": false,
  "band": "5G",
  "channel": 36
}
```

这就是客户"搜不到WiFi"的根因：设备配置中 `enabled: false`，WiFi广播被关闭了。

---

## 二、第一步：读工具 wifi.get_info（诊断查询）

### 执行过程

```
[模型调用] wifi.get_info({})
```

模型没有传入任何参数（该工具定义为无参）。工具内部执行 `_read_handler` 的 `wifi.get_info` 分支，从 `readonly_tables["wifi_config"]` 读取设备配置。

### 返回结果

```json
{
  "ok": true,
  "result": {
    "DEV_TRACE_001": {
      "ssid": "MyWiFi_5G",
      "enabled": false,      // ← 关键发现
      "hidden": false,
      "band": "5G",
      "channel": 36,
      ...
    }
  }
}
```

### 台账变化 —— 零变化！

```
[wifi_config_log]: 0 条记录   ← 不变
[switch_log]:      0 条记录   ← 不变
[data_limit_log]:  0 条记录   ← 不变
[ip_config_log]:   0 条记录   ← 不变
[operation_log]:   0 条记录   ← 不变
```

**关键特性验证**：读工具是"纯查询"，只从只读表中取数据，**不写任何台账，无副作用**。这是读工具与写工具的本质区别。

---

## 三、第二步：写工具 wifi.open（修复操作）

### 执行过程

```
[模型调用] wifi.open({"band": "all"})
```

模型传入参数 `band: "all"`，表示要求开启全部频段（2.4G + 5G）。

工具内部走 `_write_handler` 的 `wifi.open` 分支，执行流程：

1. **前置校验**：读取 `wifi_config.enabled` → 当前为 `false`（已关闭），满足开启条件
2. **构造记录**：生成 `switch_id = "ON_all_tc_2"`（确定性ID）
3. **落台账**：通过 `_write()` → `sandbox.write_record()` 写入两个地方：
   - `switch_log` 台账 append 一条业务记录
   - `operation_log` 台账 append 一条审计记录
4. **返回结果**：操作成功的业务数据

### 返回结果

```json
{
  "ok": true,
  "result": {
    "switch_id": "ON_all_tc_2",
    "action": "open",
    "band": "all",
    "status": "enabled"
  }
}
```

### 台账变化 —— 写工具触发两条记录！

**变化前** → **变化后**：

```diff
  [wifi_config_log]: 0 条记录
- [switch_log]:      0 条记录
+ [switch_log]:      1 条记录   ← 新增！tool=wifi.open, fact=wifi_opened
  [data_limit_log]:  0 条记录
  [ip_config_log]:   0 条记录
- [operation_log]:   0 条记录
+ [operation_log]:   1 条记录   ← 新增！审计日志
```

**switch_log 新增记录详情**：

```json
{
  "switch_id": "ON_all_tc_2",
  "action": "open",
  "band": "all",
  "status": "enabled",
  "tool": "wifi.open",
  "namespace_id": "run_trace:TRACE_TEST_001:rollout_0001",
  "wifi_opened": true,              // ← fact标记，verifier用此判断"已开启"
  "verified_fact_key": "wifi_opened" // ← fact名称
}
```

**operation_log 新增审计记录**：

```json
{
  "tool": "wifi.open",
  "action": "open_wifi",
  "args": {"switch_id": "ON_all_tc_2", "action": "open", "band": "all", "status": "enabled"},
  "result": "enabled",
  "namespace_id": "run_trace:TRACE_TEST_001:rollout_0001"
}
```

**关键特性验证**：
- 一次写工具调用 = **目标台账 +1条** + **审计日志 +1条**
- `wifi_opened: true` 是 verifier 判断"WiFi是否已开启"的事实依据
- `namespace_id` 确保并发rollout之间互不污染

---

## 四、第三步：幂等性测试 —— 重复调用 wifi.open

### 执行过程

```
[模型调用] wifi.open({"band": "all"})   // ← 第二次调用
```

### 结果 —— 未报错，记录了第二条

```json
{
  "ok": true,
  "result": {
    "switch_id": "ON_all_tc_3",
    "action": "open",
    "band": "all",
    "status": "enabled"
  }
}
```

**注意**：当前实现中 `wifi.open` 的前置校验基于 `env`（只读快照）而非 `sandbox`（动态状态），而 `env` 中 `enabled: false` 未被修改，因此重复调用时前置校验仍然通过。

### 台账变化

```diff
- [switch_log]:      1 条记录
+ [switch_log]:      2 条记录   ← 又append了一条（事件流语义保留历史）
- [operation_log]:   1 条记录
+ [operation_log]:   2 条记录   ← 审计日志也+1
```

两条记录的区别仅在于 `tool_call_id`：
- 记录0: `tool_call_id: "tc_2"`, `switch_id: "ON_all_tc_2"`
- 记录1: `tool_call_id: "tc_3"`, `switch_id: "ON_all_tc_3"`

---

## 五、第四步：参数校验测试 —— wifi.set_channel

### 5a. 缺少必填参数

```
[模型调用] wifi.set_channel({})   // ← 没有传 channel
```

```json
{
  "ok": false,
  "error": "channel_required",
  "message": "channel_required",
  "source": "llm"
}
```

**校验机制**：`ToolDefinition.validate_args()` 检查必填参数，`channel` 标记为 `required=True`，缺失时抛出 `ToolExecutionError("channel_required")`。

**台账变化**：无（参数校验失败，不执行handler，不落台账）

### 5b. 无效参数值

```
[模型调用] wifi.set_channel({"channel": 99, "band": "2.4G"})
```

```json
{
  "ok": false,
  "error": "invalid_channel_for_2.4g",
  "message": "invalid_channel_for_2.4g",
  "source": "llm"
}
```

**校验机制**：`_write_handler` 中硬编码了2.4G有效信道列表 `[1,2,3,...,14]`，99不在列表中。

**台账变化**：无（业务校验失败，不落台账）

### 5c. 正确调用

```
[模型调用] wifi.set_channel({"channel": 6, "band": "2.4G"})
```

```json
{
  "ok": true,
  "result": {
    "channel_id": "CH_2.4G_6_tc_4c",
    "band": "2.4G",
    "channel": 6,
    "status": "applied"
  }
}
```

### 台账变化

```diff
- [wifi_config_log]: 0 条记录
+ [wifi_config_log]: 1 条记录   ← 新增！tool=wifi.set_channel, fact=wifi_channel_set
  [switch_log]:      2 条记录
  [data_limit_log]:  0 条记录
  [ip_config_log]:   0 条记录
- [operation_log]:   2 条记录
+ [operation_log]:   3 条记录   ← 审计日志+1
```

---

## 六、最终沙盒状态

### 6.1 各台账统计

| 台账 | 记录数 | 来源工具 |
|------|--------|---------|
| `wifi_config_log` | 1 | `wifi.set_channel` × 1 |
| `switch_log` | 2 | `wifi.open` × 2 |
| `data_limit_log` | 0 | — |
| `ip_config_log` | 0 | — |
| `operation_log` | 3 | `wifi.open` × 2 + `wifi.set_channel` × 1 |

### 6.2 验证查询结果

**已执行写工具集合**：`{wifi.open, wifi.set_channel}`

**wifi.open 查询**：2条记录，每条都有 `wifi_opened: True`

**wifi.set_channel 查询**：1条记录，`wifi_channel_set: True`, `channel: 6`, `band: "2.4G"`

---

## 七、核心机制验证结论

| 机制 | 验证结果 | 说明 |
|------|---------|------|
| 读工具无副作用 | ✅ | `wifi.get_info` 执行后5个台账全部0条 → 0条 |
| 写工具落双台账 | ✅ | 每次写 = 目标台账+1 + operation_log+1 |
| 参数必填校验 | ✅ | 缺`channel` → `channel_required` 错误 |
| 参数值域校验 | ✅ | 无效信道99 → `invalid_channel_for_2.4g` 错误 |
| 错误不落台账 | ✅ | 两次参数校验失败后台账均无新增 |
| fact标记机制 | ✅ | `wifi_opened: True`, `wifi_channel_set: True` |
| namespace隔离 | ✅ | 所有记录带相同 `namespace_id` |
| 事件流语义 | ✅ | 重复调用append多条，保留完整历史 |
| 审计日志 | ✅ | operation_log记录每次写操作的tool/action/args/result |
