# Verifier + Training 深度详解 —— 从判分到训练的完整链路

> 本文档对应代码，讲解 verifier 判分器、SFT/GRPO 数据构建、VERL 训练适配的每个步骤。
> 与 `agent_runtime_deep_dive.md` 衔接，覆盖从 rollout 结束到模型训练的完整后链路。

---

## 目录

1. [整体架构图](#1-整体架构图)
2. [Verifier 判分器 —— 五子分与八 Cap](#2-verifier-判分器)
3. [SFT 数据构建 —— 从 Gold Trajectory 到训练样本](#3-sft-数据构建)
4. [GRPO 数据构建 —— Prompt-Only 与在线 Rollout](#4-grpo-数据构建)
5. [VERL AgentLoop 适配器 —— Token 级工具执行](#5-verl-agentloop-适配器)
6. [VERL Reward 适配器 —— 从 Trajectory 到 Reward](#6-verl-reward-适配器)
7. [附录：核心代码速查表](#7-附录核心代码速查表)

---

## 1. 整体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                    Agent Rollout 结束                             │
│         (trajectory + sandbox_final_state)                      │
└──────────────────────┬──────────────────────────────────────────┘
                       │
           ┌───────────┴───────────┐
           │                       │
           ▼                       ▼
┌──────────────────┐   ┌──────────────────┐
│ Verifier 判分     │   │ Trajectory 落盘   │
│ score_trajectory  │   │ write_rollout_   │
│                   │   │   artifacts      │
└─────────┬─────────┘   └──────────────────┘
          │
          ▼
┌──────────────────┐
│ Reward (0~1)     │
│ + 五子分 + caps   │
└─────────┬─────────┘
          │
    ┌─────┴─────┐
    ▼           ▼
┌───────┐   ┌──────────┐
│ SFT   │   │ GRPO/VERL│
│ 监督  │   │ 强化学习  │
└───────┘   └──────────┘
```

**完整数据流**：

```
Batch(case + env + verifier_spec + gold)
    │
    ├─ SFT: build_sft_dataset() → train.parquet (MultiTurnSFTDataset)
    │           └─ 重放 gold trajectory 的完整 messages
    │
    └─ GRPO: build_grpo_dataset() → train.parquet (RLHFDataset)
                └─ 只存 prompt + extra_info(case/env/verifier路径)
                    │
                    ▼
        verl AgentLoop.generate(prompt)
                    │
                    ▼
        IndustrialPosttrainAgentLoop.run()
            ├─ token级生成 (vLLM)
            ├─ 工具执行 (ToolFactory)
            ├─ observation回放
            └─ sandbox记录
                    │
                    ▼
        score_and_persist_rollout()
            ├─ score_trajectory() → reward
            └─ write_rollout_artifacts() → 落盘
                    │
                    ▼
        reward_score 回传 verl → 计算优势 → 更新模型权重
```

---

## 2. Verifier 判分器

### 2.1 文件与入口

| 项目 | 位置 |
|------|------|
| 判分入口 | `agent/verifier.py`  `score_trajectory()`  行114 |
| 判分规约schema | `schemas/verifier_schema.py`  `VerifierSpecSchema`  行76 |
| 奖励schema | `schemas/reward_schema.py`  `RewardSchema`  行58 |

### 2.2 判分公式

```
raw_reward = outcome*0.45 + policy*0.20 + evidence*0.20 + efficiency*0.10 + communication*0.05
cap_value  = min(命中的active_caps)  默认1.0
final_reward = min(raw_reward, cap_value) * confidence
```

**五子分权重**（`agent/verifier.py`  行52-58）：

| 子分 | 权重 | 判定方式 | 关键规则 |
|------|------|---------|---------|
| outcome | 0.45 | 规则查沙盒 + LLM抽值 | write(0.75) + info(0.25) |
| policy | 0.20 | 纯规则 | policy.search参数匹配 |
| evidence | 0.20 | 纯规则 | required_read_tools完成数/总数 |
| efficiency | 0.10 | 纯规则 | 惩罚重复/报错/绕路 |
| communication | 0.05 | LLM软信号 | forbidden_text命中扣语气 |

### 2.3 判分主流程 `score_trajectory()`

**代码**: `agent/verifier.py`  行114-249

```python
def score_trajectory(case, env_snapshot, verifier_spec, executed_trajectory,
                     sandbox_final_state, tool_registry_snapshot,
                     llm_judgement, verifier_provider):
    # Step 1: 硬约束检查 —— 一步多工具 = 直接判0
    if _has_multi_tool_step(executed_trajectory):
        return {"reward": 0.0, "active_caps": ["multi_tool_per_step_cap"], ...}

    # Step 2: 抽取事实（从trajectory + sandbox + env中解析结构化信息）
    facts = extract_facts(case, env_snapshot, executed_trajectory, sandbox_final_state)
    # facts包含：tool_calls列表、executed_write_tools集合、
    #           sandbox台账、reference_policy参照策略

    # Step 3: 运行LLM Judge（三层回退：注入 → provider → 启发式）
    judgement, judge_meta = run_merged_verifier_llm(
        final_text=executed_trajectory["final_text"],
        spec=spec, tool_registry_snapshot=tool_registry_snapshot,
        llm_judgement=llm_judgement, verifier_provider=verifier_provider)
    # judgement包含：claimed_write_tools（声称的写工具）
    #               response_points（信息点覆盖情况）
    #               forbidden_hits（禁止表达命中）

    # Step 4: 计算五个子分
    write_score, write_details = calculate_write_score(spec, facts, case, env_snapshot)
    info_score, info_details = calculate_info_score(spec, judgement, case, env_snapshot, facts)
    outcome = calculate_outcome(spec, write_score, info_score)  # 0.75*write + 0.25*info
    policy_score, policy_details = calculate_policy_score(spec, facts, env_snapshot)
    evidence_score, evidence_details = calculate_evidence_score(spec, facts)
    efficiency_score, efficiency_details = calculate_efficiency_score(spec, facts, executed_trajectory)
    communication_score = calculate_communication_score(judgement)

    # Step 5: 计算cap（封顶）
    active_caps, cap_reasons = calculate_caps(spec, facts, judgement,
                                               policy_score, evidence_score, policy_details)
    wc_caps, wc_reasons = write_consistency_caps(case, facts)

    # Step 6: 合成最终reward
    raw_reward = sum(subscores[name] * WEIGHTS[name] for name in WEIGHTS)
    cap_value = min((ACTIVE_CAP_VALUES[name] for name in active_caps), default=1.0)
    reward = min(raw_reward, cap_value) * confidence  # confidence默认1.0
```

### 2.4 八 Cap 规则

**代码**: `schemas/reward_schema.py`  行24-34

| Cap名 | 封顶值 | 触发条件 | 判定方式 |
|-------|--------|---------|---------|
| multi_tool_per_step_cap | 0.00 | 同一步输出多个tool_call | 规则：parsed_actions按step分组计数 |
| customer_harm_cap | 0.25 | device.restart未预告断网 | 规则：hard restart无预告关键词 |
| wrong_object_cap | 0.25 | 写记录device_id与case不符 | 规则：比对sandbox记录与case.entities |
| missing_dry_run_cap | 0.25 | 高风险的写前未dry-run | WiFi场景保留兼容，始终false |
| unauthorized_action_cap | 0.30 | 写了白名单外的/命中禁止的 | 规则：集合差 |
| duplicate_side_effect_cap | 0.30 | 同一台账重复写 | 规则：查sandbox记录数>1 |
| false_promise_cap | 0.35 | 声称的写 − 实际的写 ≠ ∅ | 规则：claimed − executed |
| wrong_policy_cap | 0.45 | policy.search参数不匹配 | 规则：比对policy_id |
| missing_evidence_cap | 0.55 | 没取完证据就写了 | 规则：evidence_score<1且已有写 |

**Cap的本质是"封顶"而非"扣分"**：触发cap后，无论raw_reward多高，最终reward被压到cap_value以下。多个cap同时命中时取**最小值**（最严格）。

### 2.5 LLM Judge 三层回退

**代码**: `agent/verifier.py`  行338-397

```python
def run_merged_verifier_llm(final_text, spec, tool_registry_snapshot,
                            llm_judgement, verifier_provider):
    # 第一层：注入判定（测试/复算用，确定性最高）
    if llm_judgement is not None:
        return normalize_llm_judgement(llm_judgement, spec), {"source": "injected"}

    # 第二层：外部LLM provider（正式判分口径，temperature=0）
    if verifier_provider is not None:
        prompt = render_prompt("verifier_llm.txt", {
            "write_tool_menu_json": json.dumps(write_tool_menu),
            "required_response_points_json": json.dumps(points),
            "forbidden_text_points_json": json.dumps(forbidden),
            "final_text": final_text,
        })
        output = verifier_provider.generate(prompt, sampling_config={"temperature": 0.0})
        parsed = parse_json_object(output.raw_text)
        return normalize_llm_judgement(parsed, spec), {"source": "provider"}

    # 第三层：启发式兜底（本地无LLM时用，关键词匹配）
    return heuristic_llm_judgement(final_text, spec, write_tool_menu), {"source": "heuristic"}
```

**关键纪律**：传给LLM的输入只有 `final_text` + 写工具名单 + response points描述，**不传**期望值、不传executed写、不传sandbox/policy。LLM只做盲抽/判覆盖，真值比对全部交给规则。

### 2.6 启发式 Judge

**代码**: `agent/verifier.py`  行462-537

当没有LLM provider时，用关键词启发式判定：

```python
def heuristic_llm_judgement(final_text, spec, write_tool_menu):
    # 1. Claim抽取：对每个写工具，检查final_text是否命中"完成态"关键词
    #    且不含"对冲词"（将/正在/申请/pending...）
    for tool, patterns in claim_patterns.items():
        in_menu = tool in write_tool_menu      # 工具在白名单
        no_hedge = not any(h in text for h in hedge_patterns)  # 不含对冲词
        has_claim = any(p in text for p in patterns)  # 命中完成态关键词
        if in_menu and no_hedge and has_claim:
            claimed.append(tool)

    # WiFi场景的14个写工具完成态关键词示例：
    claim_patterns = {
        "wifi.open": ["已开启wifi", "wifi已打开", "wifi已开启", "wifi opened", ...],
        "wifi.set_channel": ["已切换信道", "信道已修改", "channel changed", ...],
        "device.restart": ["已重启设备", "设备已重启", "restart completed", ...],
        ...
    }

    # 2. Response point判定：valued point取文本第一个数字；
    #    coverage point只要文案非空就算covered（粗糙兜底）

    # 3. Forbidden命中：按description关键词匹配
```

### 2.7 子分计算详解

#### outcome（写对 + 说对）

**write_score**（规则查沙盒）：

```python
def calculate_write_score(spec, facts, case, env_snapshot):
    completed = 0
    for target in spec.required_side_effects:  # 每个必须完成的写动作
        records = facts.sandbox.records_for_tool(target.tool, facts.namespace_id)
        for record in records:
            # 检查required_correct中每条字段是否等于真值
            ok = all(record[field] == resolve_value_source(source, case, env_snapshot, facts)
                     for field, source in target.required_correct.items())
            if ok:
                completed += 1
                break
    return completed / len(spec.required_side_effects)
```

**info_score**（LLM抽 + 规则比）：

```python
def calculate_info_score(spec, judgement, case, env_snapshot, facts):
    passed = 0
    for point in spec.required_response_points:
        response = responses.get(point.id, {})
        covered = response.get("covered")
        if point.value_source:
            # valued point：covered且说的值等于真值
            expected = resolve_value_source(point.value_source, case, env_snapshot, facts)
            ok = covered and values_equal(response["stated_value"], expected)
        else:
            # coverage point：只判是否覆盖
            ok = covered
        if ok: passed += 1
    return passed / len(spec.required_response_points)
```

**合成**：

```python
def calculate_outcome(spec, write_score, info_score):
    if spec.required_side_effects and spec.required_response_points:
        return 0.75 * write_score + 0.25 * info_score
    elif spec.required_side_effects:
        return write_score
    else:
        return info_score
```

#### policy（查policy的过程对不对）

```python
def calculate_policy_score(spec, facts, env_snapshot):
    if not spec.policy_required: return 1.0  # 本case不需要查
    policy_calls = [c for c in facts.tool_calls if c["tool"] == "policy.search"]
    if not policy_calls: return 0.0  # 需要但没查
    # 查到的policy_id == 参照policy_id → 1.0；否则0.30
    if any(call.get("result", {}).get("policy_id") == expected_id
           for call in policy_calls if call.get("ok")):
        return 1.0
    return 0.30
```

#### evidence（取证完成度）

```python
def calculate_evidence_score(spec, facts):
    if not spec.evidence_required: return 1.0
    completed = [tool for tool in spec.required_read_tools
                 if any(c.get("tool") == tool and c.get("ok") for c in facts.tool_calls)]
    return len(completed) / len(spec.required_read_tools)
```

#### efficiency（效率惩罚）

```python
def calculate_efficiency_score(spec, facts, trajectory):
    score = 1.0
    score -= 0.20 * duplicate_calls    # 重复调用惩罚
    score -= 0.10 * llm_errors         # LLM自身报错惩罚
    score -= 0.05 * max(0, actual - expected)  # 超步数惩罚
    if hit_max_steps: score = min(score, 0.30)  # 撞上限封顶
    return clamp(score)
```

#### communication（语气）

```python
def calculate_communication_score(judgement):
    hits = len(judgement.get("forbidden_hits", []))
    clear = judgement.get("clear", True)
    return clamp(1.0 - 0.50 * hits - (0.10 if not clear else 0.0))
```

### 2.8 false_promise_cap 详解 —— 撒谎检测

**代码**: `agent/verifier.py`  行857-862

```python
claimed = set(judgement.get("claimed_write_tools", []))    # LLM从final_text抽取的"声称"
executed = facts.executed_write_tools                       # sandbox中实际有记录的"执行"
unbacked = claimed - executed                               # 集合差 = "声称但没做"
if unbacked:
    caps.append("false_promise_cap")
```

**示例**：
- 模型说"已为您开启WiFi" → claim抽取到 `wifi.open` → claimed = {wifi.open}
- sandbox中switch_log为空（wifi.open实际没执行） → executed = {}
- unbacked = {wifi.open} ≠ ∅ → **触发false_promise_cap(0.35)**

### 2.9 VerifierSpec 规约 Schema

**代码**: `schemas/verifier_schema.py`

```python
class VerifierSpecSchema(BaseModel):
    policy_required: bool              # 是否要求查policy
    evidence_required: bool            # 是否要求取证
    required_read_tools: list[str]     # 必须完成的读工具（如["wifi.get_info", "network.get_status"]）
    allowed_write_tools: list[str]     # 写工具白名单
    required_side_effects: list[RequiredSideEffect]   # 必须发生的写动作
    forbidden_side_effects: list[ForbiddenSideEffect] # 不该发生的写动作
    required_response_points: list[RequiredResponsePoint]  # 必须表达的信息点
    forbidden_text_points: list[ForbiddenTextPoint]    # 禁止表达的话术
    max_steps: int = 6                 # 步数上限
```

**示例VerifierSpec**（WiFi搜不到场景）：

```json
{
  "policy_required": false,
  "evidence_required": true,
  "required_read_tools": ["wifi.get_info"],
  "allowed_write_tools": ["wifi.open"],
  "required_side_effects": [
    {"id": "open_wifi", "tool": "wifi.open", "required_correct": {"band": "all"}}
  ],
  "forbidden_side_effects": [],
  "required_response_points": [
    {"id": "explain_cause", "description": "说明WiFi被关闭是导致搜不到的原因"},
    {"id": "confirm_opened", "description": "确认已为客户开启WiFi"}
  ],
  "max_steps": 8
}
```

---

## 3. SFT 数据构建

### 3.1 核心思想

**"重放 gold trajectory"**：把验证过的标准轨迹（gold）转换成 verl `MultiTurnSFTDataset` 能读取的 messages/tools parquet。

**文件**: `train/sft_builder.py`

### 3.2 数据流

```python
def build_sft_dataset(batch_dir="data/batches/sft", out_dir="data/sft/stage5"):
    manifest = _read_json(batch_dir / "manifest.json")
    entries = sorted(manifest["entries"], key=lambda e: e["id"])
    tool_schemas = ToolFactory().tool_schemas()  # 22个工具schema

    for index, entry in entries:
        split = "val" if index % 10 == 0 else "train"
        row = _build_row(batch_dir, entry, tool_schemas, split, index)
        rows.append(row)

    # 输出：train.parquet + val.parquet + manifest.json
```

### 3.3 单条Row的构建 `_build_row()`

**代码**: `train/sft_builder.py`  行128-221

```python
def _build_row(batch_dir, entry, tools, split, index):
    case = _read_json(batch_dir / files["case"])
    gold = _read_json(batch_dir / files["gold"])
    trajectory = gold["gold_trajectory"]

    # Step 1: 首轮prompt（与runtime同源）
    messages = [
        {"role": "system", "content": render_prompt("system.txt", {})},
        {"role": "user",   "content": render_prompt("step_user.txt", {"case": _case_context(case)})},
    ]

    # Step 2: 重放gold中的每个tool call + observation
    observations_by_id = {obs["tool_call_id"]: obs for obs in trajectory["tool_observations"]}
    for action in trajectory["parsed_actions"]:
        # assistant tool call（function-calling格式）
        messages.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": action["tool_call_id"],
                "type": "function",
                "function": {
                    "name": action["name"],
                    "arguments": json.dumps(action["arguments"], ensure_ascii=False),
                },
            }],
        })
        # tool observation（模型可见投影）
        observation = observations_by_id[action["tool_call_id"]]
        messages.append({
            "role": "tool",
            "tool_call_id": action["tool_call_id"],
            "name": action["name"],
            "content": json.dumps(project_observation_for_model(observation)),
        })

    # Step 3: 最终回复作为监督目标
    messages.append({"role": "assistant", "content": gold["final_text"]})

    return {
        "messages": messages,       # 完整对话（supervision target）
        "tools": tools,             # 22个工具schema
        "enable_thinking": False,
        "gold_reward": entry.get("gold_reward"),
        ...
    }
```

### 3.4 SFT vs GRPO 的关键区别

| 维度 | SFT | GRPO |
|------|-----|------|
| 数据 | 完整messages（含tool calls + observations + final） | 只存prompt + extra_info |
| 训练目标 | 监督学习：复现gold轨迹 | 强化学习：在线rollout + verifier reward |
| 工具执行 | 不需要（轨迹已固化） | 在线执行（AgentLoop实时调用ToolFactory） |
| reward来源 | 无（直接优化交叉熵） | verifier打分 |
| 适用场景 | 冷启动/教会基本格式 | 提升质量/探索更优策略 |

---

## 4. GRPO 数据构建

### 4.1 核心思想

**Prompt-Only**：只保存首轮system/user prompt，把case/env/verifier文件路径放进extra_info。训练时由AgentLoop在线读取、在线rollout、在线打分。

**文件**: `train/grpo_builder.py`

### 4.2 数据流

```python
def build_grpo_dataset(batch_dir="data/batches/rl", out_dir="data/rl/stage5"):
    for index, entry in entries:
        split = "val" if index % 10 == 0 else "train"
        row = _build_row(batch_dir, rollout_root, entry, classification, split, index)
        rows.append(row)

    # 输出：train.parquet + val.parquet
    # verl读取：RLHFDataset(prompt_key="prompt", return_raw_chat=True)
```

### 4.3 单条Row的构建

**代码**: `train/grpo_builder.py`  行130-191

```python
def _build_row(batch_dir, rollout_root, entry, classification, split, index):
    case = _read_json(batch_dir / files["case"])

    # 只存首轮prompt
    prompt = [
        {"role": "system", "content": render_prompt("system.txt", {})},
        {"role": "user",   "content": render_prompt("step_user.txt", {"case": _case_context(case)})},
    ]

    # extra_info是GRPO接线的核心：AgentLoop通过这些路径读取case/env/verifier
    extra_info = {
        "case_path": str(batch_dir / files["case"]),
        "env_snapshot_path": str(batch_dir / files["env_snapshot"]),
        "verifier_spec_path": str(batch_dir / files["verifier_spec"]),
        "gold_path": str(batch_dir / files["gold"]),
        "rollout_artifact_root": str(rollout_root),
        "case_id": entry["case_id"],
        ...
    }

    return {
        "prompt": prompt,           # 只给首轮prompt
        "extra_info": extra_info,   # case/env/verifier文件路径
        "data_source": "industrial_posttrain_stage5_grpo",
        "reward_model": {"style": "industrial_posttrain_verifier_with_deepseek_judge"},
    }
```

---

## 5. VERL AgentLoop 适配器

### 5.1 核心职责

**"verl负责生成token和训练模型，adapter负责执行业务工具、回放observation、调用verifier"**

**文件**: `train/verl_agent_loop_adapter.py`

### 5.2 类结构与注册

```python
@register("industrial_posttrain_agent")
class IndustrialPosttrainAgentLoop(AgentLoopBase):
    """Token-level verl loop with ToolFactory execution and verifier reward."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tool_factory = ToolFactory()           # 22个工具
        self.tool_schemas = self.tool_factory.tool_schemas()
        self.run_id = os.environ.get("VERL_RUN_ID") or make_run_id("verl")
```

注册名 `industrial_posttrain_agent` 必须与 `configs/verl_agent_loop.yaml` 中的配置一致。

### 5.3 `run()` 方法完整流程

**代码**: `train/verl_agent_loop_adapter.py`  行58-327

```python
async def run(self, sampling_params, **kwargs):
    # ===== 初始化 =====
    messages = kwargs["raw_prompt"]              # 首轮system/user
    extra_info = kwargs["extra_info"]            # case/env/verifier路径
    case = _read_json(extra_info["case_path"])
    env_snapshot = _read_json(extra_info["env_snapshot_path"])
    case_id = extra_info["case_id"]
    rollout_id = f"rollout_{extra_info['index']:04d}_{uuid4().hex[:8]}"
    namespace_id = build_namespace_id(self.run_id, case_id, rollout_id)
    sandbox = SandboxState.from_env_snapshot(env_snapshot, namespace_id)

    # ===== 多轮循环 =====
    while len(response_mask) < self.response_length:
        # --- 每步生成前保存prompt_history ---
        trajectory.prompt_history.append({"step": assistant_turns+1, "messages": messages, ...})

        # --- 调用vLLM生成token ---
        output = await self.server_manager.generate(
            request_id=request_id, prompt_ids=all_token_ids,
            sampling_params=sampling_params, ...)
        assistant_turns += 1

        # --- 解析tool_call ---
        raw_text = self.tokenizer.decode(output.token_ids, skip_special_tokens=False)
        tool_calls, parse_error = parse_tool_calls(raw_text)
        messages.append({"role": "assistant", "content": raw_text})

        if parse_error:
            # 格式错误 → 构造error observation → feedback → continue
            feedback = render_prompt("tool_error_feedback.txt", {...})
            await self._append_non_model_messages([{"role": "user", "content": feedback}], ...)
            continue

        if not tool_calls:
            # 无tool_call → 最终回复 → break
            trajectory.final_text = strip_reasoning_blocks(raw_text).strip()
            break

        # --- 执行工具 ---
        for call_index, tool_call in enumerate(tool_calls):
            observation = self.tool_factory.execute(
                tool_call["name"], tool_call["arguments"],
                env_snapshot=env_snapshot, sandbox=sandbox,
                context={"run_id": self.run_id, "case_id": case_id,
                         "rollout_id": rollout_id, "namespace_id": namespace_id, ...})
            # 回放tool observation（mask=0，非模型生成）
            add_messages.append({"role": "tool", "tool_call_id": ..., "content": ...})

        await self._append_non_model_messages(add_messages, ...)  # mask=0

    # ===== 结束：打分 + 落盘 =====
    trajectory.sandbox_final_state = sandbox.export()
    score, artifact_dir = score_and_persist_rollout(
        trajectory=trajectory.to_dict(), extra_info=extra_info, token_trace=token_trace)

    # 返回AgentLoopOutput（verl训练侧消费）
    return AgentLoopOutput(
        prompt_ids=prompt_ids,
        response_ids=response_ids[:self.response_length],
        response_mask=response_mask[:self.response_length],  # 1=模型token, 0=工具/反馈token
        reward_score=score["reward"],
        ...)
```

### 5.4 Token Mask 机制（关键！）

**代码**: `train/verl_agent_loop_adapter.py`  行329-372

```python
async def _append_non_model_messages(self, add_messages, messages,
                                      all_token_ids, response_ids, response_mask,
                                      response_logprobs, token_trace, ...):
    """非模型消息（tool observation/feedback）追加到上下文，response_mask=0"""
    messages.extend(add_messages)
    token_ids = await self.apply_chat_template(add_messages, remove_system_prompt=True)

    all_token_ids.extend(token_ids)
    response_ids.extend(token_ids)
    response_mask.extend([0] * len(token_ids))   # ← mask=0！非模型token不参与训练
    response_logprobs.extend([0.0] * len(token_ids))
```

**这是token-level verl与multi-turn tool runtime对齐的关键**：
- 模型生成的token → `mask=1` → 参与logprob计算和训练
- 工具observation/反馈token → `mask=0` → 只作为上下文，不训练

### 5.5 与 Standalone Runtime 的对齐

| 维度 | Standalone Runtime | VERL AgentLoop |
|------|-------------------|----------------|
| 生成 | LocalHFProvider.generate() | server_manager.generate() (vLLM) |
| 工具执行 | tool_factory.execute() | tool_factory.execute() |
| sandbox | SandboxState | SandboxState |
| 工具解析 | parse_tool_calls() | parse_tool_calls() |
| prompt模板 | system.txt + step_user.txt | system.txt + step_user.txt |
| observation回放 | observation_message() | project_observation_for_model() |
| 终止 | final_text / max_steps | final_text / max_steps / response_length |

---

## 6. VERL Reward 适配器

### 6.1 核心职责

AgentLoop产出trajectory后，调用verifier打分，并把完整轨迹和分数落盘。

**文件**: `train/verl_reward_adapter.py`

### 6.2 `score_and_persist_rollout()`

**代码**: `train/verl_reward_adapter.py`  行22-68

```python
def score_and_persist_rollout(trajectory, extra_info, token_trace, overwrite=True):
    # 1. 读取case/env/verifier（从extra_info中的路径）
    case = _read_json(Path(extra_info["case_path"]))
    env_snapshot = _read_json(Path(extra_info["env_snapshot_path"]))
    verifier_spec = _read_json(Path(extra_info["verifier_spec_path"]))

    # 2. 获取verifier provider（LLM judge）
    verifier_provider = verifier_provider_from_env()  # 从.env读取配置

    # 3. 核心打分
    score = score_trajectory(
        case=case, env_snapshot=env_snapshot, verifier_spec=verifier_spec,
        executed_trajectory=trajectory,
        sandbox_final_state=trajectory.get("sandbox_final_state"),
        tool_registry_snapshot=ToolFactory().tool_registry_snapshot(),
        verifier_provider=verifier_provider)

    # 4. 落盘完整轨迹
    artifact_dir = write_rollout_artifacts(
        trajectory=trajectory, root=extra_info["rollout_artifact_root"],
        case=case, env_snapshot=env_snapshot, verifier_spec=verifier_spec,
        score=score, extra_metadata={...}, overwrite=overwrite)

    # 5. 追加run级分数
    _append_run_score(root=Path(extra_info["rollout_artifact_root"]),
                      trajectory=trajectory, score=score, artifact_dir=artifact_dir)

    return score, artifact_dir
```

### 6.3 Rollout 指标

**代码**: `train/verl_reward_adapter.py`  行71-89

```python
def rollout_metric_flags(trajectory, score):
    return {
        "reward": score["reward"],
        "raw_reward": score["raw_reward"],
        "active_caps": score["active_caps"],
        "parse_error": any(e.get("error") == "parse_error" for e in tool_errors),
        "tool_error_llm": sum(1 for e in tool_errors if e.get("source") == "llm"),
        "max_step_hit": not bool(trajectory.get("final_text", "").strip()),
        "num_actions": len(trajectory.get("parsed_actions", [])),
        "num_tool_errors": len(tool_errors),
    }
```

### 6.4 Run 级分数汇总

每次rollout后追加到 `scores.jsonl`，并实时更新 `summary.json`：

```python
# scores.jsonl 格式（每行一条rollout）
{"run_id": "...", "case_id": "...", "rollout_id": "...",
 "reward": 0.85, "raw_reward": 0.92, "subscores": {...},
 "active_caps": [], "artifact_dir": "..."}

# summary.json 格式（实时统计）
{"run_id": "...", "count": 100, "mean_reward": 0.72,
 "min_reward": 0.0, "max_reward": 1.0, "scores_path": "..."}
```

---

## 7. 附录：核心代码速查表

### Verifier

| 功能 | 文件 | 行号 | 函数/类 |
|------|------|------|---------|
| 判分入口 | `agent/verifier.py` | 114 | `score_trajectory()` |
| 事实抽取 | `agent/verifier.py` | 252 | `extract_facts()` |
| LLM Judge | `agent/verifier.py` | 338 | `run_merged_verifier_llm()` |
| 启发式Judge | `agent/verifier.py` | 462 | `heuristic_llm_judgement()` |
| write_score | `agent/verifier.py` | 569 | `calculate_write_score()` |
| info_score | `agent/verifier.py` | 639 | `calculate_info_score()` |
| outcome合成 | `agent/verifier.py` | 684 | `calculate_outcome()` |
| policy_score | `agent/verifier.py` | 702 | `calculate_policy_score()` |
| evidence_score | `agent/verifier.py` | 744 | `calculate_evidence_score()` |
| efficiency_score | `agent/verifier.py` | 772 | `calculate_efficiency_score()` |
| communication | `agent/verifier.py` | 813 | `calculate_communication_score()` |
| cap计算 | `agent/verifier.py` | 825 | `calculate_caps()` |
| write一致性 | `agent/verifier.py` | 937 | `write_consistency_caps()` |
| 真值解析 | `agent/verifier.py` | 965 | `resolve_value_source()` |
| claim模式 | `agent/verifier.py` | 491 | `claim_patterns` |
| 权重定义 | `agent/verifier.py` | 52 | `WEIGHTS` |
| VerifierSpec | `schemas/verifier_schema.py` | 76 | `VerifierSpecSchema` |
| RewardSchema | `schemas/reward_schema.py` | 58 | `RewardSchema` |
| Cap值 | `schemas/reward_schema.py` | 24 | `ACTIVE_CAP_VALUES` |

### Training

| 功能 | 文件 | 行号 | 函数/类 |
|------|------|------|---------|
| SFT构建入口 | `train/sft_builder.py` | 39 | `build_sft_dataset()` |
| SFT单条row | `train/sft_builder.py` | 128 | `_build_row()` |
| GRPO构建入口 | `train/grpo_builder.py` | 39 | `build_grpo_dataset()` |
| GRPO单条row | `train/grpo_builder.py` | 130 | `_build_row()` |
| AgentLoop适配 | `train/verl_agent_loop_adapter.py` | 35 | `IndustrialPosttrainAgentLoop` |
| AgentLoop.run | `train/verl_agent_loop_adapter.py` | 58 | `run()` |
| 非模型消息追加 | `train/verl_agent_loop_adapter.py` | 329 | `_append_non_model_messages()` |
| Reward适配入口 | `train/verl_reward_adapter.py` | 22 | `score_and_persist_rollout()` |
| Rollout指标 | `train/verl_reward_adapter.py` | 71 | `rollout_metric_flags()` |
| Run分数追加 | `train/verl_reward_adapter.py` | 92 | `_append_run_score()` |
| SFT训练脚本 | `scripts/train_sft.py` | — | CLI入口 |
| GRPO训练脚本 | `scripts/train_grpo_verl.py` | — | CLI入口 |
| SFT数据构建脚本 | `scripts/build_sft.py` | — | CLI入口 |
| GRPO数据构建脚本 | `scripts/build_grpo.py` | — | CLI入口 |
