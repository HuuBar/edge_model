# WiFi Agent 数据处理 Pipeline v2 — 系统性设计

> 从问题出发，分层分类处理，配置驱动，人机协作

**核心转变**: v1试图"一个脚本解决所有问题" → v2"先分类问题，再分别处理"

---

## 一、问题分类体系

### 1.1 问题四象限

```
                    确定性高
                       │
        A1 简单替换    │    B1 直接读取
        A2 参数路由    │    B2 ReAct解析
        ───────────────┼────────────────
        C1 格式标准化  │    D1 JSON Schema验证
        C2 多语言对齐  │    D3 逻辑验证
                       │
                    确定性低
           LLM辅助 ←─┼─→ 人工介入
                       │
        A3 LLM语义推断 │    A4 人工标记
        C3 轨迹重构    │    D4 人工审核
        B3 状态反推    │
        B4 知识提取    │    D2 LLM语义验证
```

### 1.2 四类问题定义

| 类别 | 名称 | 特征 | 处理方式 | 是否需要LLM |
|------|------|------|---------|------------|
| **A** | 工具名映射 | 旧工具名→新工具名 | 查找表/参数路由/语义推断 | A3需要 |
| **B** | 数据提取 | 从JSON字段提取值 | 直接读取/结构解析/状态反推 | B3,B4需要 |
| **C** | 结构转换 | 格式标准化 | 代码转换/多语言对齐 | 不需要 |
| **D** | 验证修复 | 确保数据质量 | Schema验证/语义验证/逻辑验证 | D2需要 |

---

## 二、Pipeline A: 工具名映射

### 2.1 问题分析

原始数据中的工具名与代码中的定义不一致，需要映射。但映射不是简单的一对一替换：

**示例**: `switch_wifi_enable` → 需要根据 `{"action": "ON"}` 或 `{"action": "OFF"}` 分别映射为 `wifi.open` 或 `wifi.close`

### 2.2 四级处理策略

```
旧工具名
  │
  ├─ 在A1查找表中？ → A1: 直接替换 (33个映射)
  │
  ├─ 在A2参数列表中？ → A2: 参数路由 (2个映射)
  │
  ├─ 有tool_result上下文？ → A3: LLM语义推断
  │   (prompt: "根据tool_name={old}和tool_result={result}，推断对应的新工具名")
  │
  └─ 以上都不是 → A4: 标记为待人工确认
```

### 2.3 映射表分级

**A1: 简单查找表** (`pipeline_a/a1_simple_map.py`)

确定性100%，无需LLM，直接字符串替换。

```python
SIMPLE_MAP = {
    "get_traffic_statistics": "data.get_usage",
    "get_wifi_info": "wifi.get_info",
    "list_wifi_clients": "wifi.list_clients",
    # ... 共33个
}
```

**A2: 参数路由表** (`pipeline_a/a2_param_route.py`)

需要根据参数值选择映射目标。

```python
PARAM_ROUTE = {
    "switch_wifi_enable": {
        "field": "action",
        "routes": {"ON": "wifi.open", "OFF": "wifi.close"}
    },
    # 未来可能有更多
}
```

**A3: LLM语义推断** (`pipeline_a/a3_llm_infer.py`)

当工具名不在A1/A2中，但有上下文（tool_result/thought）时，用LLM推断。

```python
prompt = f"""
原始数据中有一条工具调用记录：
- 工具名: {old_tool_name}
- 参数: {args}
- 执行结果: {tool_result}
- 思考过程: {thought}

请判断这个工具对应的新系统中的哪个工具。
新系统工具列表: {new_tool_list}

只输出工具名，不要解释。
"""
```

温度: 0.3（低随机性）

**A4: 人工标记** (`pipeline_a/a4_manual_mark.py`)

无法自动映射的，输出到 `review/manual_tool_mapping.json`，等待人工确认。

记录格式:
```json
{
  "case_id": "WIFI_001",
  "old_tool_name": "switch_game_turbo",
  "args": {"enabled": true},
  "tool_result": "",
  "suggested": null,
  "reason": "新系统中无对应工具",
  "status": "pending_manual"
}
```

---

## 三、Pipeline B: 数据提取

### 3.1 问题分析

原始JSON有多个字段需要提取，但提取方式不同：
- `query_en` → 直接读取为customer_message
- `react_label_en` → 需要解析ReAct结构
- `tool_result` → 需要从文本反推设备状态
- `rag.knowledge` → 需要结构化为env上下文

### 3.2 四级处理策略

**B1: 直接字段读取** (`pipeline_b/b1_direct_read.py`)

确定性100%，直接从JSON字段读取。

```python
FIELD_MAP = {
    "query_en": ("case", "customer_message"),
    "query_ar": ("case", "customer_message_ar"),
    "query_cn": ("case", "customer_message_cn"),
    "dataset": ("case", "scene_label"),
    "categories": ("case", "tool_categories"),
}
```

**B2: ReAct轨迹解析** (`pipeline_b/b2_react_parse.py`)

从 `react_label_en` 列表中提取结构化信息。

```python
def parse_react_step(step: dict) -> dict:
    """解析单步ReAct记录。"""
    content = step.get("content", {})
    return {
        "thought": content.get("thought", ""),
        "tool_name": content.get("tool_name", ""),
        "args": content.get("args", {}),
        "tool_result": content.get("tool_result", ""),
        "final_answer": content.get("final_answer", ""),
    }
```

**B3: 设备状态反推** (`pipeline_b/b3_state_infer.py`)

从 `tool_result` 文本中反推设备状态值。

```python
# 示例: tool_result = "总共17.3GB,当月3.2GB"
# 反推: data_usage.total_download_mb = 17700, data_usage.current_month_download_mb = 3200

STATE_PATTERNS = {
    "data_usage": {
        r"总共([\d.]+)\s*GB": "total_download_mb",  # ×1024
        r"当月([\d.]+)\s*GB": "current_month_download_mb",
        r"剩余([\d.]+)\s*GB": "remaining_quota_mb",
    },
    "wifi_config": {
        r"当前信道[:：]\s*(\d+)": "channel",
        r"SSID[:：]\s*(\S+)": "ssid",
    },
    "network_status": {
        r"信号强度[:：]\s*(-?\d+)": "signal_strength",
        r"延迟[:：]\s*(\d+)": "latency_ms",
    },
}
```

对于无法正则匹配的复杂情况，使用LLM辅助提取。

**B4: 知识提取** (`pipeline_b/b4_knowledge_extract.py`)

从 `rag.knowledge` 中提取有用的上下文信息，加入到env_snapshot。

---

## 四、Pipeline C: 结构转换

### 4.1 C1: 四件套格式转换

已有代码: `convert_react_to_quartet.py`

### 4.2 C2: 多语言对齐 (`pipeline_c/c2_multilang_align.py`)

**策略**: EN + AR + CN 同步保存，但训练时只读 EN/AR。

```python
def align_multilang(record: dict) -> dict:
    """多语言对齐：确保三种语言的数据都存在且对应。"""
    aligned = {
        "en": {
            "query": record.get("query_en", ""),
            "react": record.get("react_label_en", []),
            "rag": record.get("rag_en", {}),
        },
        "ar": {
            "query": record.get("query_ar", ""),
            "react": record.get("react_label_ar", []),
            "rag": record.get("rag_ar", {}),
        },
        "cn": {
            "query": record.get("query_cn", ""),
            # CN没有react_label和rag，只保存query用于调试
        },
    }
    return aligned
```

四件套中保存:
- `case.json`: `customer_message` (EN), `customer_message_ar` (AR), `customer_message_cn` (CN调试用)
- `gold.json`: 只用 EN 的 react_label
- SFT训练parquet: `messages` 中只用 EN/AR

### 4.3 C3: 轨迹重构 (`pipeline_c/c3_trajectory_rebuild.py`)

某些记录的trajectory有错误（如参数错误导致tool_result失败），需要重构为正确的执行路径。

```python
def rebuild_trajectory(react_label: list) -> list:
    """重构trajectory，修正错误步骤。

    策略:
    1. 如果tool_result表示参数错误 → 修正参数后重新生成observation
    2. 如果缺少必要的读步骤 → 插入（如wifi.set_channel前插入wifi.get_info）
    3. 如果步骤顺序不合理 → 重排
    """
```

### 4.4 C4: 意图标准化 (`pipeline_c/c4_intent_normalize.py`)

将 `categories` 和 `dataset` 标准化为 `primary_intent`。

---

## 五、Pipeline D: 验证修复

### 5.1 D1: 格式验证 (`validators/d1_format_validator.py`)

检查四件套的JSON结构完整性。

```python
REQUIRED_FIELDS = {
    "case.json": ["case_id", "customer_message", "entities", "primary_intent"],
    "env_snapshot.json": ["case_id", "readonly_tables", "policies"],
    "gold.json": ["case_id", "gold_trajectory", "parsed_actions", "tool_observations"],
    "verifier_spec.json": ["required_read_tools", "allowed_write_tools", "max_steps"],
}
```

### 5.2 D2: 语义验证 (`validators/d2_semantic_validator.py`)

用LLM验证工具名映射的正确性。

```python
prompt = f"""
检查以下工具调用是否合理：
- 用户请求: {customer_message}
- 调用的工具: {tool_name}
- 参数: {args}

这个工具调用是否能正确解决用户的请求？
回答: YES 或 NO，并简要说明原因。
"""
```

### 5.3 D3: 逻辑验证 (`validators/d3_logic_validator.py`)

检查trajectory的逻辑合理性：
- 写操作前是否有读操作（如wifi.open前是否有wifi.get_info）
- 参数值是否在有效范围内
- 步骤顺序是否合理

### 5.4 D4: 人工审核 (`validators/d4_manual_review.py`)

汇总所有需要人工确认的项到 `review/review_queue.json`。

---

## 六、LLM API 连接器

### 6.1 配置

```yaml
# config.yaml
llm:
  dsv4:
    base_url: "http://10.44.209.63:3000/v1"
    api_key: "sk-nxyLmHRnhpaxOj2ewYbuih68RUTYSQeQCKjc6woSf7DVGlX8"
    model_name: "dsv4"
    
  temperature:
    mapping: 0.3      # 工具映射：低随机性
    extraction: 0.3   # 数据提取：低随机性
    validation: 0.1   # 验证：极低随机性
    generation: 0.7   # 生成：正常随机性
    
  retry:
    max_retries: 3
    backoff: exponential  # 1s, 3s, 7s
    
  rate_limit:
    max_workers: 2
    delay_between_calls: 0.5
```

### 6.2 客户端代码 (`llm_client.py`)

```python
class LLMClient:
    """统一LLM调用客户端，内置限流、重试、退避。"""
    
    def __init__(self, config: dict):
        self.base_url = config["base_url"]
        self.api_key = config["api_key"]
        self.model = config["model_name"]
        
    def call(self, prompt: str, temperature: float = 0.3, 
             max_retries: int = 3) -> str:
        """带重试和退避的LLM调用。"""
        for attempt in range(max_retries):
            try:
                resp = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": temperature,
                    },
                    timeout=30,
                )
                if resp.status_code == 429:
                    wait = 2 ** attempt + 1
                    time.sleep(wait)
                    continue
                return resp.json()["choices"][0]["message"]["content"]
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt + 1)
                else:
                    raise
```

---

## 七、Pipeline Runner (Orchestrator)

### 7.1 执行流程

```python
def run_pipeline(config: dict, input_dir: Path, output_dir: Path):
    """执行完整pipeline。"""
    
    # 1. 读取所有JSON文件
    records = load_all_records(input_dir)
    
    # 2. 对每个记录执行四路pipeline
    for record in records:
        # Pipeline A: 工具名映射
        mapped_tools = run_pipeline_a(record, config["tool_mapping"])
        
        # Pipeline B: 数据提取
        extracted_data = run_pipeline_b(record, config["extraction"])
        
        # Pipeline C: 结构转换
        quartet = run_pipeline_c(mapped_tools, extracted_data, config)
        
        # Pipeline D: 验证
        validation_result = run_pipeline_d(quartet, config["validation"])
        
        # 收集结果
        if validation_result["passed"]:
            save_quartet(quartet, output_dir)
        else:
            save_review_item(validation_result, output_dir / "review")
    
    # 3. 输出统计
    print_stats(output_dir)
```

### 7.2 断点续跑

```python
# 使用checkpoint文件记录处理进度
checkpoint = load_checkpoint(output_dir / "checkpoint.json")
# 只处理未完成的记录
records = records[checkpoint["last_processed_index"]:]
```

---

## 八、完整文件结构

```
data_pipeline_v2/
├── config.yaml              # 配置驱动
├── runner.py                # Orchestrator
├── llm_client.py            # LLM API连接器
│
├── pipeline_a/              # 工具名映射
│   ├── __init__.py
│   ├── a1_simple_map.py     # 简单查找表 (33个映射)
│   ├── a2_param_route.py    # 参数路由 (2个映射)
│   ├── a3_llm_infer.py      # LLM语义推断
│   └── a4_manual_mark.py    # 人工标记
│
├── pipeline_b/              # 数据提取
│   ├── __init__.py
│   ├── b1_direct_read.py    # 直接字段读取
│   ├── b2_react_parse.py    # ReAct轨迹解析
│   ├── b3_state_infer.py    # 设备状态反推
│   └── b4_knowledge_extract.py  # 知识提取
│
├── pipeline_c/              # 结构转换
│   ├── __init__.py
│   ├── c1_quartet_convert.py   # 四件套转换
│   ├── c2_multilang_align.py   # 多语言对齐
│   ├── c3_trajectory_rebuild.py # 轨迹重构
│   └── c4_intent_normalize.py  # 意图标准化
│
├── pipeline_d/              # 验证修复
│   ├── __init__.py
│   ├── d1_format_validator.py   # 格式验证
│   ├── d2_semantic_validator.py # 语义验证
│   ├── d3_logic_validator.py    # 逻辑验证
│   └── d4_manual_review.py      # 人工审核
│
├── validators/              # 验证器集合
│   ├── __init__.py
│   ├── schema_validator.py
│   └── consistency_validator.py
│
├── outputs/                 # 输出目录
│   ├── quartets/            # 四件套
│   ├── splits/              # 划分结果
│   ├── review/              # 待人工审核项
│   └── logs/                # 处理日志
│
└── tests/                   # 测试
    ├── test_pipeline_a.py
    ├── test_pipeline_b.py
    └── test_integration.py
```

---

## 九、实施路线图

### Phase 1: 基础设施（1天）
- [ ] `llm_client.py` — LLM API连接器（限流+重试）
- [ ] `config.yaml` — 配置驱动
- [ ] `runner.py` — Orchestrator框架

### Phase 2: Pipeline A + B（2天）
- [ ] `a1_simple_map.py` — 33个简单映射
- [ ] `a2_param_route.py` — 参数路由
- [ ] `b1_direct_read.py` — 字段读取
- [ ] `b2_react_parse.py` — ReAct解析
- [ ] 集成测试：A+B跑通

### Phase 3: Pipeline C + D（2天）
- [ ] `c1_quartet_convert.py` — 四件套转换
- [ ] `c2_multilang_align.py` — 多语言对齐
- [ ] `d1_format_validator.py` — 格式验证
- [ ] 集成测试：A+B+C+D跑通

### Phase 4: 端到端验证（2天）
- [ ] 用A级数据（1063条）跑完整pipeline
- [ ] 检查review队列中的待确认项
- [ ] 人工确认A4/D4项
- [ ] 生成最终parquet

### Phase 5: SFT训练（3天）
- [ ] 用parquet训练4B模型
- [ ] 评测对比（训练前 vs 训练后）
- [ ] 调优

---

## 十、设计原则总结

| 原则 | 说明 |
|------|------|
| **问题驱动** | 不试图一个脚本解决所有问题，而是先分类，再分别处理 |
| **配置驱动** | 每类问题的处理策略在config.yaml中定义，可插拔可替换 |
| **人机协作** | LLM处理语义推断(A3/D2)，人工处理不确定项(A4/D4) |
| **验证闭环** | 每个pipeline输出都经过验证，确保数据质量 |
| **断点续跑** | 支持checkpoint，处理大量数据时不怕中断 |
| **多语言** | EN+AR+CN同步保存，训练时只用EN/AR |
