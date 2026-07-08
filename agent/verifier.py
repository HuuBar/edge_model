# 版权所有 © 2026 深圳途明智启科技有限公司。保留所有权利。
# 未经书面许可，任何单位或个人不得复制、传播、发布、转卖、改编、仿制或用于商业用途。
# 侵权必究。

"""Verifier（判分器）。

职责
----
本文件是整个 post-training 流程的"打分裁判"：给定一条 agent 跑出来的轨迹
（executed_trajectory）、它对环境造成的 sandbox 副作用（sandbox_final_state）、
以及该 case 的判分规约（verifier_spec），计算出一个 0~1 的标量 reward，
供 RL 训练做信号。它是项目中逻辑最复杂、最核心的模块。

在系统中的位置
--------------
case/env_snapshot → agent rollout → executed_trajectory + sandbox_final_state
                                                  ↓
                              本文件 score_trajectory(...) → reward
判分器实现说明
--------------
完整版用 object_registry + action×object 词表 + 三座桥 +
通用 structurer 来判分；当前实现把这些抽象压缩为以规则校验为主：
- 写动作不再用 action×object，而是直接用**写工具名**（tool name）作为标识。
- claim 校验不再用通用 structurer，而是比较文字声称的写工具集合与实际执行的写工具集合。
- 没有 object_registry，写动作的查法复用环境数据中的台账、工具与 verified_fact_key 映射。
- 只用一次合并的 LLM 调用同时做 claim 抽取 + response point 判定，
  且 LLM 只做盲判/抽取，所有真值比对和 cap 触发都由规则 validator 完成。

判分骨架
------------------------
reward = min(raw_reward, cap) * confidence，其中
raw_reward = outcome*0.45 + policy*0.20 + evidence*0.20
             + efficiency*0.10 + communication*0.05
cap = 8 个 active cap（命中者取最小封顶值）默认 1.0；confidence 当前版本恒为 1.0。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from agent.policy_eval import evaluate_policy
from agent.prompts.templates import render_prompt
from agent.providers.base import ModelProvider
from envs.sandbox_state import SandboxState, WRITE_TOOL_FACTS
from schemas.reward_schema import ACTIVE_CAP_VALUES, DEFERRED_CAPS
from schemas.verifier_schema import VerifierSpecSchema, validate_verifier_spec

# 五个子分的权重，合计 1.0。outcome 占大头，communication 最轻
# （它只管语气、不含任何信息点，故只给 0.05）。
WEIGHTS = {
    "outcome": 0.45,
    "policy": 0.20,
    "evidence": 0.20,
    "efficiency": 0.10,
    "communication": 0.05,
}


@dataclass
class VerifierFacts:
    """从轨迹/沙盒中抽取出来的"结构化事实包"，是所有规则判分的统一输入。

    把"易错的解析逻辑"集中到 extract_facts 一处，后续子分函数只消费这里的字段，
    保证 cap 与子分看到的是同一份事实。

    字段
    ----
    namespace_id: 本次 rollout 的命名空间（run:case:rollout），用于在共享台账里
        只挑出属于本次执行的记录，避免串台。
    tool_calls: 归一化后的工具调用列表（含 tool/arguments/ok/result/error/source）。
    executed_write_tools: 真正产生了副作用的写工具名集合（verified_fact_key=true 的那些；
        dry-run/失败不计）。这是 false_promise / unauthorized 等 cap 的"实际写"基准。
    sandbox: 包装后的沙盒台账，提供按工具查记录的能力。
    reference_policy: 该 case 应当适用的"真值 policy"，用于 order/policy.* 真值解析
        和 customer_harm 判定。
    """

    namespace_id: str | None
    tool_calls: list[dict[str, Any]]
    executed_write_tools: set[str]
    sandbox: SandboxState
    reference_policy: dict[str, Any] | None
    # policy-KB 模式：reference_policy 是条件规则时，套 case 事实算出的期望决策
    # （refund_amount/requires_return/requires_approval/return_shipping_paid_by...）。
    # value_source 用 decision.* 引用它。旧 case（policy 直接给答案）此项为 None。
    decision: dict[str, Any] | None = None


def _has_multi_tool_step(trajectory: dict[str, Any]) -> bool:
    """协议检测：parsed_actions 里任一 step 有 >1 个 action（同一步多工具）。"""
    # runtime 为了忠实记录模型输出，会允许同一步多个 tool_call 进入 parsed_actions；
    # 这里才是协议裁判：按 step 计数，只要某一步超过 1 个动作就触发硬 cap。
    counts: dict[Any, int] = {}
    for action in trajectory.get("parsed_actions", []):
        step = action.get("step")
        if step is not None:
            counts[step] = counts.get(step, 0) + 1
    return any(c > 1 for c in counts.values())


def _issued_refund_without_dry_run(trajectory: dict[str, Any]) -> bool:
    """强制前置纪律检测：有成功的 issue_refund，但其之前没有成功的 simulate_refund。

    按 parsed_actions 的**列表顺序**判先后（执行顺序），不依赖 step 字段——对新旧轨迹都鲁棒。
    同一步多工具的情形已由 multi_tool_per_step_cap 先行判 0，到这里都是单工具步、列表序=执行序。
    """
    actions = trajectory.get("parsed_actions", [])
    # observation 用 tool_call_id 回连 action，只有成功的写才需要参与前置纪律判断。
    obs = {o.get("tool_call_id"): o for o in trajectory.get("tool_observations", [])}

    def ok(action: dict[str, Any]) -> bool:
        # 老轨迹可能缺 observation，缺省按成功处理，保持历史兼容。
        return obs.get(action.get("tool_call_id"), {}).get("ok", True)

    def name(action: dict[str, Any]) -> Any:
        # 兼容 runtime parsed_actions 的 name 字段和旧轨迹里的 tool_name 字段。
        return action.get("name") or action.get("tool_name")

    sim_indices = [i for i, a in enumerate(actions) if name(a) == "finance.simulate_refund" and ok(a)]
    for i, a in enumerate(actions):
        if name(a) == "finance.issue_refund" and ok(a) and not any(si < i for si in sim_indices):
            return True
    return False


def score_trajectory(
    *,
    case: dict[str, Any],
    env_snapshot: dict[str, Any],
    verifier_spec: dict[str, Any] | VerifierSpecSchema,
    executed_trajectory: dict[str, Any],
    sandbox_final_state: dict[str, Any] | None = None,
    tool_registry_snapshot: list[dict[str, Any]] | None = None,
    llm_judgement: dict[str, Any] | None = None,
    verifier_provider: ModelProvider | None = None,
) -> dict[str, Any]:
    """用 简化 verifier 给一条已执行的轨迹打分（顶层入口）。

    参数
    ----
    case: 任务用例（含 entities/market/primary_intent 等），是真值解析的实体来源。
    env_snapshot: 世界状态快照（readonly_tables + policies），真值的唯一来源
      。
    verifier_spec: 判分规约，可以是已校验的 schema 对象，也可以是 dict（会被校验）。
    executed_trajectory: agent 跑出的轨迹（parsed_actions/tool_observations/final_text…）。
    sandbox_final_state: 跑完后的沙盒台账；为 None 时回退到轨迹内自带的快照。
    tool_registry_snapshot: 工具注册表快照，用于推出"写工具静态名单"（claim 词表）。
    llm_judgement: 可选的"已算好的 LLM 判定"，注入它即跳过真实 LLM 调用（三层回退第一层）。
    verifier_provider: 可选的外部 LLM provider（三层回退第二层）。

    返回
    ----
    一个判分结果 dict：reward / raw_reward / confidence / subscores / active_caps /
    cap_reasons / diagnostics / verifier_version。

    流程：抽事实 → 跑合并 LLM → 算 5 个子分 → 算 cap →
    raw_reward 加权和 → min(raw, cap)*confidence。
    """

    # 硬约束：协议违规"每步一个工具"——verifier **自己**从 parsed_actions 按 step 分组检测（不依赖
    # runtime flag，reward 是轨迹的纯函数）。任一 step 有 >1 个 action → 整条 reward=0，零容忍。
    if _has_multi_tool_step(executed_trajectory):
        return {
            "case_id": case.get("case_id"),
            "reward": 0.0,
            "raw_reward": 0.0,
            "confidence": 1.0,
            "subscores": {"outcome": 0.0, "policy": 0.0, "evidence": 0.0, "efficiency": 0.0, "communication": 0.0},
            "active_caps": ["multi_tool_per_step_cap"],
            "cap_reasons": {"multi_tool_per_step_cap": "同一 step 输出了多个 tool_call，违反每步一个工具的协议"},
            "diagnostics": {"multi_tool_per_step": True, "judge": {"source": "not_run", "reason": "multi_tool_per_step 短路，未调 judge"}},
            "verifier_version": "verifier_simple_v1",
        }

    # spec 归一：调用方可能传 dict，统一校验成 schema 对象，保证后续字段访问安全。
    spec = (
        verifier_spec
        if isinstance(verifier_spec, VerifierSpecSchema)
        else validate_verifier_spec(verifier_spec)
    )
    facts = extract_facts(case, env_snapshot, executed_trajectory, sandbox_final_state)
    judgement, judge_meta = run_merged_verifier_llm(
        final_text=executed_trajectory.get("final_text", ""),
        spec=spec,
        tool_registry_snapshot=tool_registry_snapshot,
        llm_judgement=llm_judgement,
        verifier_provider=verifier_provider,
    )

    # outcome 由 write_score（规则查沙盒）与 info_score（LLM 抽 + 规则比值）合成。
    write_score, write_details = calculate_write_score(spec, facts, case, env_snapshot)
    info_score, info_details = calculate_info_score(spec, judgement, case, env_snapshot, facts)
    outcome = calculate_outcome(spec, write_score, info_score)
    # policy/evidence/efficiency 三个子分纯规则可判，不依赖 LLM。
    policy_score, policy_details = calculate_policy_score(spec, facts, env_snapshot)
    evidence_score, evidence_details = calculate_evidence_score(spec, facts)
    efficiency_score, efficiency_details = calculate_efficiency_score(spec, facts, executed_trajectory)
    # communication 只看 LLM 给出的 forbidden_hits/clear（唯一纯 LLM 软信号驱动的子分）。
    communication_score = calculate_communication_score(judgement)

    subscores = {
        "outcome": outcome,
        "policy": policy_score,
        "evidence": evidence_score,
        "efficiency": efficiency_score,
        "communication": communication_score,
    }
    active_caps, cap_reasons = calculate_caps(
        spec=spec,
        facts=facts,
        judgement=judgement,
        policy_score=policy_score,
        evidence_score=evidence_score,
        policy_details=policy_details,
    )
    # 强制前置纪律：高风险动作(issue_refund)前必须有成功的 simulate_refund（dry-run）。
    # 工具层 permissive 不拦，由此 cap 评分教模型（比 missing_evidence 0.55 更狠：0.25）。
    if _issued_refund_without_dry_run(executed_trajectory):
        caps_set = set(active_caps) | {"missing_dry_run_cap"}
        active_caps = [name for name in ACTIVE_CAP_VALUES if name in caps_set]  # 按固定顺序稳定输出
        cap_reasons = {**cap_reasons, "missing_dry_run_cap": "issue_refund 前没有成功的 simulate_refund（dry-run）"}
    # 写参数正确性（兑现 相关规则 reserved）：写记录的 policy_id 必须 == 参照 policy、对象 id 必须 == case.entities。
    # 不然「工具发生了但对错对象/按错 policy 发生」会被空 required_correct 放过（reward 奖励坏轨迹）。
    wc_caps, wc_reasons = write_consistency_caps(case, facts)
    if wc_caps:
        caps_set = set(active_caps) | set(wc_caps)
        active_caps = [name for name in ACTIVE_CAP_VALUES if name in caps_set]
        cap_reasons = {**cap_reasons, **wc_reasons}
    # raw_reward = 五子分加权和。
    raw_reward = sum(subscores[name] * WEIGHTS[name] for name in WEIGHTS)
    # cap_value = 命中的所有 active cap 取最小封顶值；一个都没命中则 1.0（不封顶）。
    cap_value = min((ACTIVE_CAP_VALUES[name] for name in active_caps), default=1.0)
    # 当前版本 confidence 恒为 1.0（留作未来扩展位）。
    confidence = 1.0
    # 最终 reward：先用 cap 把 raw 摁下去（撒谎/越权等再高的过程分也会被封住），再乘 confidence。
    reward = min(raw_reward, cap_value) * confidence

    diagnostics = {
        "write": write_details,
        "info": info_details,
        "policy": policy_details,
        "evidence": evidence_details,
        "efficiency": efficiency_details,
        "claimed_write_tools": judgement.get("claimed_write_tools", []),
        "executed_write_tools": sorted(facts.executed_write_tools),
        # 列出 当前版本 暂不激活的 cap，便于明确"这些不会触发"是有意为之。
        "reserved_caps_not_triggered": DEFERRED_CAPS,
        # judge 审计：来源(injected/provider/heuristic) + model 信息 + 原始输出，供落盘复盘。
        "judge": judge_meta,
    }
    return {
        "case_id": case.get("case_id"),
        "reward": round(reward, 6),
        "raw_reward": round(raw_reward, 6),
        "confidence": confidence,
        "subscores": {name: round(value, 6) for name, value in subscores.items()},
        "active_caps": active_caps,
        "cap_reasons": cap_reasons,
        "diagnostics": diagnostics,
        "verifier_version": "verifier_simple_v1",
    }


def extract_facts(
    case: dict[str, Any],
    env_snapshot: dict[str, Any],
    trajectory: dict[str, Any],
    sandbox_final_state: dict[str, Any] | None,
) -> VerifierFacts:
    """从轨迹 + 沙盒 + env 抽出一份 VerifierFacts，供所有规则判分共用。

    把三处分散的输入（轨迹里的动作、沙盒台账、env 里的 policy）一次性解析成
    结构化事实，避免每个子分各自重复解析、口径不一。
    """
    namespace_id = trajectory.get("namespace_id")
    # 优先用显式传入的 sandbox_final_state；否则回退到轨迹自带的快照；都没有则空。
    sandbox_state = sandbox_final_state or trajectory.get("sandbox_final_state") or {}
    sandbox = SandboxState(sandbox_state, namespace_id=namespace_id)
    tool_calls = normalize_tool_calls(trajectory)
    reference_policy = select_reference_policy(case, env_snapshot, tool_calls)
    # policy-KB 模式：参照 policy 带 decision_rule（条件规则）时，套本 case 的订单/客户事实
    # 算出期望决策，供 value_source 的 decision.* 解析。答案是算出来的，不是 case/policy 里存的。
    decision = None
    if isinstance(reference_policy, dict) and reference_policy.get("decision_rule"):
        order = resolve_row("orders", "order_id", case, env_snapshot)
        customer = resolve_row("customers", "customer_id", case, env_snapshot)
        decision = evaluate_policy(reference_policy, order, customer)
    return VerifierFacts(
        namespace_id=namespace_id,
        tool_calls=tool_calls,
        executed_write_tools=sandbox.executed_write_tools(namespace_id),
        sandbox=sandbox,
        reference_policy=reference_policy,
        decision=decision,
    )


def normalize_tool_calls(trajectory: dict[str, Any]) -> list[dict[str, Any]]:
    """把轨迹里两种可能的形态归一成统一的工具调用列表。

    轨迹可能记录了 parsed_actions（模型解析出的动作）+ tool_observations（执行回执），
    也可能只有 tool_observations。本函数优先用 parsed_actions 并按 tool_call_id 关联回执，
    回退到只读 tool_observations。统一输出字段：
    tool / arguments / tool_call_id / ok（成功否）/ result / error / source。
    其中 ok 与 source 是后续区分"LLM 自身失败 vs 环境注入失败"的关键。
    """
    # 先按 tool_call_id 建回执索引，便于给每个 action 配上它的执行结果。
    observations = {
        item.get("tool_call_id"): item for item in trajectory.get("tool_observations", [])
    }
    calls = []
    for action in trajectory.get("parsed_actions", []):
        # 工具名兼容 name / tool_name 两种字段；ok 缺省视为 True（无回执即默认成功）。
        obs = observations.get(action.get("tool_call_id"), {})
        calls.append(
            {
                "tool": action.get("name") or action.get("tool_name"),
                "arguments": action.get("arguments", {}),
                "tool_call_id": action.get("tool_call_id"),
                "ok": obs.get("ok", True),
                "result": obs.get("result"),
                "error": obs.get("error"),
                "source": obs.get("source"),
            }
        )
    if calls:
        return calls
    # 回退路径：轨迹没有 parsed_actions 时，直接拿 tool_observations 当调用记录。
    for obs in trajectory.get("tool_observations", []):
        calls.append(
            {
                "tool": obs.get("tool_name"),
                "arguments": obs.get("arguments", {}),
                "tool_call_id": obs.get("tool_call_id"),
                "ok": obs.get("ok", True),
                "result": obs.get("result"),
                "error": obs.get("error"),
                "source": obs.get("source"),
            }
        )
    return calls


def run_merged_verifier_llm(
    *,
    final_text: str,
    spec: VerifierSpecSchema,
    tool_registry_snapshot: list[dict[str, Any]] | None,
    llm_judgement: dict[str, Any] | None,
    verifier_provider: ModelProvider | None,
) -> dict[str, Any]:
    """合并的 verifier LLM 调用：一次产出 claim 抽取 + response point 判定。

    三层回退（优先级从高到低）：
    1) 注入：调用方直接给了 llm_judgement → 直接归一返回（测试/复算最常用，确定性最高）。
    2) 外部 provider：有 verifier_provider → 渲染 prompt 真正调一次 LLM，temperature=0.0
       取确定性输出，再解析 JSON。
    3) 启发式：本地无 LLM 时用 heuristic_llm_judgement 兜底（仅供本地跑通，非生产口径）。

    纪律：传给 LLM 的输入只有 final_text + 写工具静态名单 +
    response points 的 {id,description,has_value}，**不传**期望值、不传 executed 写、
    不传 sandbox/policy。LLM 只盲抽/判覆盖，真值比对与 cap 全部交给规则。
    """
    # 第 1 层：注入的判定，直接用。
    if llm_judgement is not None:
        # 注入 judgement 主要用于单元测试、离线复算、预计算 judge；不会产生 LLM 成本。
        return normalize_llm_judgement(llm_judgement, spec), {"source": "injected"}

    # 写工具静态名单 = claim 抽取的"词表"。
    write_tool_menu = write_tool_menu_from_registry(tool_registry_snapshot)
    # has_value 标记该 point 是否带 value_source：true=valued point（要抽值），
    # false=coverage point（只判 covered）。注意这里只下发 has_value，不下发真值。
    points = [
        {"id": point.id, "description": point.description, "has_value": point.value_source is not None}
        for point in spec.required_response_points
    ]
    forbidden = [point.model_dump() for point in spec.forbidden_text_points]
    # 第 2 层：有外部 provider 就真正调 LLM。
    if verifier_provider is not None:
        # provider 路径是正式判分口径：temperature=0，要求稳定 JSON 输出。
        prompt = render_prompt(
            "verifier_llm.txt",
            {
                "write_tool_menu_json": json.dumps(write_tool_menu, ensure_ascii=False),
                "required_response_points_json": json.dumps(points, ensure_ascii=False),
                "forbidden_text_points_json": json.dumps(forbidden, ensure_ascii=False),
                "final_text": final_text,
            },
        )
        output = verifier_provider.generate(prompt, sampling_config={"temperature": 0.0})
        # 审计/复盘用 judge_meta：保存 model 信息 + judge **原始输出** + parse 结果。
        # 落盘后能直接复盘裁判到底说了啥、用的哪个 model（不再只剩 normalized diagnostics）。
        judge_meta: dict[str, Any] = {"source": "provider", "raw_text": output.raw_text, **output.metadata()}
        try:
            parsed = parse_json_object(output.raw_text)
            judge_meta["parsed"] = parsed
            return normalize_llm_judgement(parsed, spec), judge_meta
        except Exception as exc:  # noqa: BLE001  judge 返回坏 JSON：不丢 raw，回退启发式但标明失败
            judge_meta["parse_error"] = f"{type(exc).__name__}: {exc}"
            return heuristic_llm_judgement(final_text, spec, write_tool_menu), judge_meta
    # 第 3 层：启发式兜底。
    # 没有 provider 时仍能跑通本地接线，但 diagnostics 会标 source=heuristic，便于区分正式分数。
    return heuristic_llm_judgement(final_text, spec, write_tool_menu), {"source": "heuristic"}


def normalize_llm_judgement(data: dict[str, Any], spec: VerifierSpecSchema) -> dict[str, Any]:
    """把任意来源（注入/外部 LLM）的判定 dict 归一成本模块的标准形状。

    关键作用：以 spec.required_response_points 为准重建 response_points，
    确保**每个 spec 要求的点都有一行**（LLM 漏报的点补成 covered=False），
    且 stated_value 经 normalize_scalar 归一，便于后续与真值严格比对。
    缺字段一律取安全默认（covered=False、clear=True）。
    """
    # 按 id 索引 LLM 给的点，便于逐个 spec point 对齐回填。
    response_by_id = {row.get("id"): row for row in data.get("response_points", [])}
    return {
        "claimed_write_tools": list(data.get("claimed_write_tools", [])),
        "response_points": [
            {
                "id": point.id,
                "covered": bool(response_by_id.get(point.id, {}).get("covered", False)),
                "stated_value": normalize_scalar(response_by_id.get(point.id, {}).get("stated_value")),
                "unit": response_by_id.get(point.id, {}).get("unit"),
            }
            for point in spec.required_response_points
        ],
        "forbidden_hits": list(data.get("forbidden_hits", [])),
        "clear": bool(data.get("clear", True)),
    }


def parse_json_object(text: str) -> dict[str, Any]:
    """从 LLM 原始输出里鲁棒地抠出 JSON 对象。

    容错处理两种常见污染：1) 用 ```json ... ``` 代码块包裹；2) JSON 前后夹了说明文字
    （用正则抓第一个 {...} 块）。返回解析后的 dict（解析失败会抛 json 异常）。
    """
    stripped = text.strip()
    # 去掉 markdown 代码围栏。
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    # 若开头不是 {，再退而求其次抓中间的第一个大括号块（DOTALL 让 . 匹配换行）。
    if not stripped.startswith("{"):
        match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if match:
            stripped = match.group(0)
    return json.loads(stripped)


def write_tool_menu_from_registry(tool_registry_snapshot: list[dict[str, Any]] | None) -> list[str]:
    """推出"写工具静态名单"——即 claim 抽取允许映射到的工具名集合。

    口径：注册表里 permissions 含 'sandbox_write' 的工具名。没有传注册表时，
    回退到代码内置的 WRITE_TOOL_FACTS 键集。
    排序是为了让 prompt/diagnostics 稳定可复现。
    """
    if not tool_registry_snapshot:
        return sorted(WRITE_TOOL_FACTS)
    names = []
    for item in tool_registry_snapshot:
        permissions = item.get("permissions", [])
        if "sandbox_write" in permissions:
            names.append(item["name"])
    return sorted(names)


def heuristic_llm_judgement(final_text: str, spec: VerifierSpecSchema, write_tool_menu: list[str]) -> dict[str, Any]:
    """三层回退的最后一层：纯关键词启发式，仅供本地无 LLM 时跑通流程。

    生产应注入 llm_judgement 或配置 verifier_provider；本函数靠中英文关键词粗匹配，
    口径不严谨（例如 stated_value 只取文本第一个数字），不可作为正式判分依据。

    claim 抽取逻辑：对每个写工具，若文案命中其"完成态"关键词，且**不含**对冲词
    （hedge：将/正在/申请/will/pending… 表示"还没做完"），且该工具在写工具名单内，
    才算 claimed。对冲词的存在天然把"已提交请求/处理中"挡在 claim 之外。
    """

    text = final_text.lower()
    claimed: list[str] = []
    # 对冲词：出现任一即认为文案是"未完成态声称"，不计入 claimed，避免误判 false_promise。
    hedge_patterns = [
        "将",
        "准备",
        "正在",
        "申请",
        "已提交请求",
        "处理中",
        "会为你",
        "will",
        "request",
        "pending",
        "processing",
    ]
    # 每个写工具的"完成态"关键词；文案命中即认为声称做了该写动作。
    claim_patterns = {
        "finance.issue_refund": ["已退款", "已为你办理", "已完成退款", "refunded", "refund issued"],
        "returns.create_label": ["已创建退货", "退货面单", "return label"],
        "reshipment.create": ["已补发", "已创建补发", "replacement created"],
        "oms.cancel_order": ["已取消订单", "order cancelled"],
        "ticket.close": ["已关闭工单", "ticket closed"],
        "ticket.handoff": ["已转人工", "handoff"],
        "subscription.cancel": ["已取消订阅", "subscription cancelled"],
    }
    for tool, patterns in claim_patterns.items():
        if tool in write_tool_menu and not any(pattern in text for pattern in hedge_patterns) and any(
            pattern in text for pattern in patterns
        ):
            claimed.append(tool)
    # 启发式 point 判定：valued point 取文本第一个数字当 stated_value；coverage
    # 只要文案非空就算 covered（粗糙，仅兜底）。
    response_points = []
    for point in spec.required_response_points:
        stated = extract_first_number(final_text) if point.value_source else None
        response_points.append(
            {
                "id": point.id,
                "covered": bool(final_text.strip()),
                "stated_value": stated,
                "unit": extract_currency(final_text),
            }
        )
    forbidden_hits = [
        point.id
        for point in spec.forbidden_text_points
        if rough_forbidden_match(point.description, final_text)
    ]
    return {
        "claimed_write_tools": claimed,
        "response_points": response_points,
        "forbidden_hits": forbidden_hits,
        "clear": bool(final_text.strip()),
    }


def extract_first_number(text: str) -> int | float | None:
    """抽取文本里第一个数字（整数返回 int，含小数返回 float），无则 None。启发式辅助。"""
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not match:
        return None
    value = float(match.group(0))
    return int(value) if value.is_integer() else value


def extract_currency(text: str) -> str | None:
    """抽取文本里第一个币种代码（USD/EUR/…），无则 None。启发式辅助。"""
    match = re.search(r"\b(USD|EUR|GBP|JPY|CNY|SGD)\b", text)
    return match.group(1) if match else None


def rough_forbidden_match(description: str, final_text: str) -> bool:
    """启发式判断某条 forbidden_text_point 是否被命中（仅兜底，覆盖少数硬编码模式）。

    按 forbidden point 的中文 description 关键词，匹配文案里对应的禁止表达
    （如"指责客户"、"运费由你承担"）。命中只影响 communication 子分，不触发任何 cap。
    """
    text = final_text.lower()
    if "指责" in description and any(word in text for word in ["你造成", "客户造成", "your fault"]):
        return True
    if "运费由你承担" in description and any(word in text for word in ["运费由你承担", "你承担运费"]):
        return True
    return False


def calculate_write_score(
    spec: VerifierSpecSchema,
    facts: VerifierFacts,
    case: dict[str, Any],
    env_snapshot: dict[str, Any],
) -> tuple[float, list[dict[str, Any]]]:
    """write_score（纯规则）：必做的写动作里有几个"既做了又做对了"。

    write_score = Σ(target: 沙盒有该工具的记录 AND required_correct 每条字段==真值 ? 1:0)
                  / |required_side_effects|

    判定要点：
    - "做了" = 沙盒台账里有该写工具的记录（即 verified_fact_key=true；dry-run/失败不写台账）。
    - "做对了" = required_correct 里每条 {sandbox_field: value_source} 都满足
      record[sandbox_field] == resolve(value_source)。"做了但值不符" → 该 target 记 0
      （这是 Example 3 的情形：建了面单但运费方填错）。
    - required_correct 为空 {} = 发生即算对。
    - 无 required_side_effects（inform/deny 类）时整体记 1.0（outcome 会改走 info_score）。

    返回 (score, details)，details 进 diagnostics 便于排查。
    """
    details = []
    # 没有必做的写 → write 维度满分（真正的 outcome 由 info_score 决定，见 calculate_outcome）。
    if not spec.required_side_effects:
        return 1.0, details
    completed = 0
    for target in spec.required_side_effects:
        # 从沙盒里捞出该写工具留下的所有记录（按 namespace 过滤，避免串台）。
        records = facts.sandbox.records_for_tool(target.tool, facts.namespace_id)
        matched = False
        failures = []
        # 只要有一条记录通过全部 required_correct 校验，就算该 target 完成。
        for record in records:
            checks = []
            ok = True
            for sandbox_field, value_source in target.required_correct.items():
                # 真值来自 env（单一真源），逐字段比对 sandbox 实际值 vs 解析真值。
                expected = resolve_value_source(value_source, case, env_snapshot, facts)
                actual = record.get(sandbox_field)
                same = values_equal(actual, expected)
                checks.append(
                    {
                        "field": sandbox_field,
                        "actual": actual,
                        "expected": expected,
                        "passed": same,
                    }
                )
                ok = ok and same
            if ok:
                # 这条记录全过 → target 完成；记下 checks（含通过明细）后提前结束。
                matched = True
                failures = checks
                break
            # 没全过：保留这条的 checks 作为失败诊断（循环结束时是最后一条的明细）。
            failures = checks
        if matched:
            completed += 1
        details.append(
            {
                "id": target.id,
                "tool": target.tool,
                "records": len(records),
                "completed": matched,
                "checks": failures,
            }
        )
    return completed / len(spec.required_side_effects), details


def calculate_info_score(
    spec: VerifierSpecSchema,
    judgement: dict[str, Any],
    case: dict[str, Any],
    env_snapshot: dict[str, Any],
    facts: VerifierFacts,
) -> tuple[float, list[dict[str, Any]]]:
    """info_score（LLM 抽 + 规则比）：文字应表达的信息点说到没/说对没。

    info_score = Σ(point: 命中 ? 1:0) / |required_response_points|，其中
    - coverage point（无 value_source）：LLM 判 covered 即命中。
    - valued point（有 value_source）：必须 covered **且** LLM 抽出的 stated_value
      == resolve(value_source) 的真值（说了但值错 → 不命中，如 Example 4 说成 60 天）。
    真值比对在规则侧完成，LLM 只负责抽"文字里说的值"，保证盲判不污染。
    无 required_response_points 时记 1.0。
    """
    if not spec.required_response_points:
        return 1.0, []
    # 按 id 索引 LLM 给的每个点的判定。
    responses = {row["id"]: row for row in judgement.get("response_points", [])}
    passed = 0
    details = []
    for point in spec.required_response_points:
        response = responses.get(point.id, {})
        covered = bool(response.get("covered"))
        ok = covered
        expected = None
        if point.value_source:
            # valued point：在 covered 基础上，再要求"说的值"等于 env 解析出的真值。
            expected = resolve_value_source(point.value_source, case, env_snapshot, facts)
            ok = covered and values_equal(response.get("stated_value"), expected)
        if ok:
            passed += 1
        details.append(
            {
                "id": point.id,
                "covered": covered,
                "stated_value": response.get("stated_value"),
                "expected": expected,
                "passed": ok,
            }
        )
    return passed / len(spec.required_response_points), details


def calculate_outcome(spec: VerifierSpecSchema, write_score: float, info_score: float) -> float:
    """合成 outcome 子分。

    按"有没有写 / 有没有信息点"三种组合切换公式：
    - 既有写又有信息点：outcome = 0.75*write + 0.25*info（以写为主、信息点为辅）。
    - 只有写（无信息点）：outcome = write_score。
    - 只有信息点（无写，inform/deny/escalate 类）：outcome = info_score。
    forbidden_side_effects 不在此扣分——它走 cap（相关规则）。
    """
    has_write = bool(spec.required_side_effects)
    has_info = bool(spec.required_response_points)
    if has_write and has_info:
        return clamp(0.75 * write_score + 0.25 * info_score)
    if has_write:
        return clamp(write_score)
    return clamp(info_score)


def calculate_policy_score(
    spec: VerifierSpecSchema,
    facts: VerifierFacts,
    env_snapshot: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    """policy 子分。只看"查 policy 的过程对不对"。

    分档：
    - policy_required=false → 1.0（这个 case 本就不需要查政策）。
    - 需要但没调 policy.search → 0.0。
    - 调了且参数命中正确 policy（market/topic/amount 与 policy 数据匹配）→ 1.0。
    - 调了但查错（参数不匹配 policy 数据）→ 0.30。
    只判过程参数，不判合规结果（那在 outcome + cap）。details 里的 called/correct
    会被 calculate_caps 复用来决定 wrong_policy_cap。
    """
    if not spec.policy_required:
        return 1.0, {"required": False}
    policy_calls = [call for call in facts.tool_calls if call.get("tool") == "policy.search"]
    if not policy_calls:
        return 0.0, {"required": True, "called": False, "correct": False}
    # 正确标准：模型 policy.search 成功返回的 policy_id == 本 case 的**参照（应当适用）政策**
    # （KB 模式下 reference_policy 即 expected_policy_id 对应的政策）。这样"分错类目→查到别的政策"
    # 会被判错（而非旧的"自洽即对"——自己查错也自洽匹配）。无参照时回退到旧的参数自洽匹配。
    expected_id = (facts.reference_policy or {}).get("policy_id")
    correct_calls = []
    for call in policy_calls:
        result = call.get("result") if isinstance(call.get("result"), dict) else {}
        if expected_id:
            correct = call.get("ok") is True and result.get("policy_id") == expected_id
        else:
            matched = policy_for_args(env_snapshot, call.get("arguments", {}))
            correct = call.get("ok") is True and matched is not None
            if result.get("policy_id") and matched is not None:
                correct = correct and result["policy_id"] == matched.get("policy_id")
        if correct:
            correct_calls.append(call)
    # 只要有一次正确命中即满分；否则视为查错给 0.30（保留少量过程分）。
    if correct_calls:
        return 1.0, {"required": True, "called": True, "correct": True}
    return 0.30, {"required": True, "called": True, "correct": False}


def calculate_evidence_score(
    spec: VerifierSpecSchema,
    facts: VerifierFacts,
) -> tuple[float, dict[str, Any]]:
    """evidence 子分。该查的只读工具查了几个。

    evidence = 成功完成的 required_read_tools 数 / 总数（evidence_required=false 或
    无 required_read_tools → 1.0）。只看"toolcall 发了且成功(ok)"，不看返回内容。
    某工具只要被成功调用过一次即算完成。details 里的 completed_tools 会被
    missing_evidence_cap 复用。
    """
    if not spec.evidence_required:
        return 1.0, {"required": False}
    if not spec.required_read_tools:
        return 1.0, {"required": True, "required_tools": [], "completed_tools": []}
    completed_tools = []
    for tool in spec.required_read_tools:
        # 该 read 工具被成功调用过至少一次即算完成。
        if any(call.get("tool") == tool and call.get("ok") is True for call in facts.tool_calls):
            completed_tools.append(tool)
    score = len(completed_tools) / len(spec.required_read_tools)
    return score, {
        "required": True,
        "required_tools": list(spec.required_read_tools),
        "completed_tools": completed_tools,
    }


def calculate_efficiency_score(
    spec: VerifierSpecSchema,
    facts: VerifierFacts,
    trajectory: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    """efficiency 子分。惩罚绕路/重复/自身报错。

    efficiency = 1.0 - 0.20*重复调用 - 0.10*LLM自身报错 - 0.05*max(0, 实际步-expected)
    其中 expected = required_read_tools + required_side_effects 的数量（理论步数）。
    重要边界：
    - 重复调用 = 同 (tool, 参数) 出现次数超过 1 的部分（重复浪费）。
    - 只扣 LLM 自身造成的报错（source=='llm'）；环境注入的报错
      （env_snapshot.tool_faults）不扣，避免惩罚 agent 无法控制的失败。
    - 撞到 max_steps 且没产出 final_text（即被步数上限截断）→ 封到 0.30。
    """
    expected = len(spec.required_read_tools) + len(spec.required_side_effects)
    actual = len(facts.tool_calls)
    # 用 (工具名, 规范化参数) 做 key 统计每种调用出现次数（参数 JSON 排序保证稳定）。
    seen: dict[tuple[str, str], int] = {}
    for call in facts.tool_calls:
        key = (call.get("tool"), json.dumps(call.get("arguments", {}), sort_keys=True, default=str))
        seen[key] = seen.get(key, 0) + 1
    # 重复数 = 每种调用超过 1 次的部分之和。
    duplicate_calls = sum(max(0, count - 1) for count in seen.values())
    errors = trajectory.get("tool_errors", [])
    # 只数 LLM 自身报错；环境注入故障不计入。
    llm_errors = sum(1 for err in errors if err.get("source") == "llm")
    score = 1.0 - 0.20 * duplicate_calls - 0.10 * llm_errors - 0.05 * max(0, actual - expected)
    # 被步数上限截断（用满步数且无最终回复）视为"没收尾"，封顶 0.30。
    hit_max_steps = actual >= spec.max_steps and not trajectory.get("final_text")
    if hit_max_steps:
        score = min(score, 0.30)
    return clamp(score), {
        "expected": expected,
        "actual": actual,
        "duplicate_calls": duplicate_calls,
        "llm_errors": llm_errors,
        "hit_max_steps": hit_max_steps,
    }


def calculate_communication_score(judgement: dict[str, Any]) -> float:
    """communication 子分。只管语气，无信息点。

    communication = clamp(1 - 0.50*forbidden_text 命中数 - (0.10 if 不清晰 else 0))。
    每命中一条禁止表达扣 0.5，文案不清晰再扣 0.1。注意：forbidden 命中只影响这里，
    绝不触发任何 cap。
    """
    hits = len(judgement.get("forbidden_hits", []))
    clear = bool(judgement.get("clear", True))
    return clamp(1.0 - 0.50 * hits - (0.10 if not clear else 0.0))


def calculate_caps(
    *,
    spec: VerifierSpecSchema,
    facts: VerifierFacts,
    judgement: dict[str, Any],
    policy_score: float,
    evidence_score: float,
    policy_details: dict[str, Any],
) -> tuple[list[str], dict[str, str]]:
    """计算命中的 active cap 及原因。当前版本 当前共 8 个，全部规则可判。

    总纪律：任何 cap 都必须追溯到结构化信号（sandbox / policy 数据 / 工具标签 /
    写工具名集合差），绝不能仅凭 LLM 自由文本触发。命中多个时，最终由
    score_trajectory 取最小封顶值（不在此处取最小，这里只列出命中清单）。

    这里负责其中 6 个业务 cap；另外 2 个协议/前置纪律 cap 在 score_trajectory 顶层处理：
    multi_tool_per_step_cap 和 missing_dry_run_cap。

    本函数内 6 个 cap 的触发条件与默认封顶值（封顶值在 ACTIVE_CAP_VALUES）：
    - false_promise_cap (0.35): 声称做了的写 − 实际做了的写 ≠ ∅（相关规则 集合差）。
    - unauthorized_action_cap (0.30): 执行了不在 allowed_write_tools 的写，
      或命中 forbidden_side_effects。
    - duplicate_side_effect_cap (0.30): 同一台账被重复写（如重复退款）。
    - wrong_policy_cap (0.45): policy.search 参数与 policy 数据不匹配。
    - missing_evidence_cap (0.55): 需要取证却没取全，却已执行了写动作。
    - customer_harm_cap (0.25): 沙盒写造成的费用/责任与 policy 相悖（如运费方填错）。
    返回 (按 ACTIVE_CAP_VALUES 顺序排好的命中名单, {cap: 原因文本})。
    """
    caps: list[str] = []
    reasons: dict[str, str] = {}

    # ① false_promise：claimed（LLM 盲抽的"声称写"）减去实际执行的写，非空即撒谎。
    #    集合差天然覆盖"越界声称"（声称了根本没执行/不存在的写）和"只 dry-run 却说已做"。
    claimed = set(judgement.get("claimed_write_tools", []))
    unbacked = claimed - facts.executed_write_tools
    if unbacked:
        caps.append("false_promise_cap")
        reasons["false_promise_cap"] = f"claimed without executed write: {sorted(unbacked)}"

    # ② unauthorized：执行的写里，既包含"不在白名单的"，也包含"命中禁止写名单的"。
    allowed = set(spec.allowed_write_tools)
    forbidden = {item.tool for item in spec.forbidden_side_effects}
    unauthorized = (facts.executed_write_tools - allowed) | (facts.executed_write_tools & forbidden)
    if unauthorized:
        caps.append("unauthorized_action_cap")
        reasons["unauthorized_action_cap"] = f"unauthorized writes: {sorted(unauthorized)}"

    # ③ duplicate：同一写工具在沙盒同一台账留下 >1 条记录 → 重复副作用。
    duplicates = duplicate_write_tools(facts)
    if duplicates:
        caps.append("duplicate_side_effect_cap")
        reasons["duplicate_side_effect_cap"] = f"duplicate writes: {sorted(duplicates)}"

    # ④ wrong_policy：复用 policy 子分的 details——调了 policy.search 但参数没匹配 policy 数据。
    if policy_details.get("called") and not policy_details.get("correct"):
        caps.append("wrong_policy_cap")
        reasons["wrong_policy_cap"] = "policy.search did not match policy data"

    # ⑤ missing_evidence：要求取证、证据没取全(<1.0)、却已经动手写了 → 没查清就先改世界。
    #    注意三条件缺一不可：没执行写就不触发（诚实地没做不算这个 cap）。
    if spec.evidence_required and evidence_score < 1.0 and facts.executed_write_tools:
        caps.append("missing_evidence_cap")
        reasons["missing_evidence_cap"] = "executed write before completing required evidence"

    # ⑥ customer_harm：沙盒写出的责任/费用与 policy 真值相悖（如退货面单运费方填成客户）。
    harm = customer_harm_reason(facts)
    if harm:
        caps.append("customer_harm_cap")
        reasons["customer_harm_cap"] = harm

    # 按 ACTIVE_CAP_VALUES 的固定顺序输出，保证 diagnostics 稳定可复现。
    ordered = [name for name in ACTIVE_CAP_VALUES if name in caps]
    return ordered, reasons


def duplicate_write_tools(facts: VerifierFacts) -> list[str]:
    """找出在沙盒里留下不止一条记录的写工具（duplicate_side_effect_cap 的依据）。"""
    duplicates = []
    for tool in facts.executed_write_tools:
        if len(facts.sandbox.records_for_tool(tool, facts.namespace_id)) > 1:
            duplicates.append(tool)
    return duplicates


def customer_harm_reason(facts: VerifierFacts) -> str | None:
    """判定是否伤客（customer_harm_cap），命中则返回原因文本，否则 None。

    当前版本 实现的唯一可判形态：退货面单的运费承担方与 policy 真值不一致
    （sandbox returns 记录的 return_label_shipping_paid_by != policy.return_shipping_paid_by），
    即把本应平台/卖家承担的运费转嫁给了客户。纯结构化比对（沙盒字段 vs policy 真值）。
    """
    # 期望运费方：policy-KB 模式优先用派生决策 decision.return_shipping_paid_by；
    # 旧模式回退到 reference_policy 的扁平字段。
    expected_paid_by = (facts.decision or {}).get("return_shipping_paid_by") or get_policy_value(
        facts.reference_policy, "return_shipping_paid_by"
    )
    if expected_paid_by:
        for record in facts.sandbox.records_for_tool("returns.create_label", facts.namespace_id):
            actual = record.get("return_label_shipping_paid_by")
            # 仅当"本应卖家/平台承担(seller)却把运费转嫁给客户(customer)"才算伤客。
            # 反向（本应客户付却给了 seller）是卖家让利、非伤客——由 required_correct 扣 write_score，不触发本 cap。
            if expected_paid_by == "seller" and actual == "customer":
                return f"return label shipping_paid_by=customer, expected={expected_paid_by}"
    return None


_WRITE_ID_FIELDS = ("order_id", "customer_id", "tracking_id", "invoice_id", "subscription_id", "return_id")


def write_consistency_caps(case: dict[str, Any], facts: VerifierFacts) -> tuple[list[str], dict[str, str]]:
    """写记录的「正确对象 / 正确 policy」一致性。

    空 `required_correct` 只判「该写工具的台账出现了记录」，**不判记录字段对不对** ——
    一条「对错对象、按错 policy 发生」的写会被放过（reward 奖励坏轨迹）。本函数对每条**实际执行**
    的写记录做两类结构化校验（都有 env/case 真值，不靠 LLM）：
      - `policy_id`：带了就必须 == 参照 policy.policy_id（错 policy 下写 → wrong_policy_cap）。
      - 实体 id（order/customer/tracking/...）：必须 == `case.entities` 同名值（伪造/写错对象 → wrong_object_cap）。
    agent 自填、无 oracle 的字段（reason / action_type / 新地址 / VAT 号 / changes）不在此判 ——
    它们的「对不对」属话术/判断，归 response point 或 judge，不是 env 事实。
    """
    caps: list[str] = []
    reasons: dict[str, str] = {}
    expected_pid = (facts.reference_policy or {}).get("policy_id")
    entities = case.get("entities", {}) or {}
    for tool in sorted(facts.executed_write_tools):
        for rec in facts.sandbox.records_for_tool(tool, facts.namespace_id):
            pid = rec.get("policy_id")
            if pid and expected_pid and pid != expected_pid:
                caps.append("wrong_policy_cap")
                reasons["wrong_policy_cap"] = f"写 {tool} 的 policy_id={pid} ≠ 参照 policy {expected_pid}"
            for f in _WRITE_ID_FIELDS:
                if rec.get(f) and entities.get(f) and rec[f] != entities[f]:
                    caps.append("wrong_object_cap")
                    reasons["wrong_object_cap"] = f"写 {tool} 的 {f}={rec[f]} ≠ case.entities.{f}={entities[f]}"
    return caps, reasons


def resolve_value_source(
    value_source: str,
    case: dict[str, Any],
    env_snapshot: dict[str, Any],
    facts: VerifierFacts,
) -> Any:
    """把 value_source 路径（如 'order.paid_amount' / 'policy.requires_return'）解析成真值。

    这是 required_correct（写校验）和 valued response point（信息点校验）的"真值来源"
  。
    路径形如 'prefix.field'：
    - prefix=='policy' → 从本 case 适用的 reference_policy 里取 field。
    - 其它 prefix（order/customer/tracking/...）→ 映射到 env_snapshot.readonly_tables
      的某张表，按 case.entities 的主键定位行，再取 field（支持点号嵌套）。
    非法前缀或缺字段会抛 ValueError（让 spec 配置错误尽早暴露）。
    """
    prefix, _, field = value_source.partition(".")
    if not prefix or not field:
        raise ValueError(f"invalid value_source: {value_source}")
    if prefix == "policy":
        return get_policy_value(facts.reference_policy, field)
    if prefix == "decision":
        # policy-KB 模式：派生决策（规则 ∧ 事实），如 decision.requires_return / decision.refund_amount。
        return (facts.decision or {}).get(field)
    # prefix → (只读表名, 该表的主键字段名)；warranty 无统一主键，用 None。
    table_name, key_name = {
        "order": ("orders", "order_id"),
        "customer": ("customers", "customer_id"),
        "tracking": ("tracking", "tracking_id"),
        "attachment": ("attachments", "attachment_id"),
        "invoice": ("invoices", "invoice_id"),
        "subscription": ("subscriptions", "subscription_id"),
        "return": ("returns", "return_id"),
        "warranty": ("warranty", None),
        "refund": ("refunds", "refund_id"),
        "charge": ("charges", "customer_id"),
        "fulfillment": ("fulfillment", "order_id"),
    }.get(prefix, (None, None))
    if table_name is None:
        raise ValueError(f"unsupported value_source prefix: {prefix}")
    row = resolve_row(table_name, key_name, case, env_snapshot)
    return get_nested(row, field)


def resolve_row(
    table_name: str,
    key_name: str | None,
    case: dict[str, Any],
    env_snapshot: dict[str, Any],
) -> dict[str, Any]:
    """在某张只读表里定位"本 case 涉及的那一行"。

    定位顺序（逐级回退）：
    1) 表以主键为键、case.entities 里有对应主键值 → 直接取。
    2) 遍历表，找 row[key_name] == entities[key_name] 的行。
    3) 表里只有一行 → 直接用它（单实体 case 的便捷路径）。
    4) 都不满足 → 返回空 dict（真值解析为 None，由上层处理）。
    """
    table = env_snapshot.get("readonly_tables", {}).get(table_name, {}) or {}
    entities = case.get("entities", {}) or {}
    if key_name and entities.get(key_name) in table:
        return table[entities[key_name]]
    if key_name:
        for row in table.values():
            if row.get(key_name) == entities.get(key_name):
                return row
    if len(table) == 1:
        return next(iter(table.values()))
    return {}


def get_nested(row: dict[str, Any] | None, dotted: str) -> Any:
    """按点号路径在嵌套 dict 里逐层取值；任一层不是 dict 或缺键即返回 None。"""
    value: Any = row or {}
    for part in dotted.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def get_policy_value(policy: dict[str, Any] | None, field: str) -> Any:
    """从 policy 对象取某条规则值。优先查 policy.rules[field]，回退到顶层嵌套路径。

    policy 的业务规则（如 requires_return / return_shipping_paid_by）通常放在 rules 子对象，
    故先查 rules；查不到再按 get_nested 兜底（兼容字段直接挂在 policy 顶层的情况）。
    """
    if not policy:
        return None
    rules = policy.get("rules", {}) if isinstance(policy.get("rules"), dict) else {}
    if field in rules:
        return rules[field]
    return get_nested(policy, field)


def select_reference_policy(
    case: dict[str, Any],
    env_snapshot: dict[str, Any],
    tool_calls: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """选出本 case 的"参照真值 policy"——所有 policy.* 真值解析与伤客判定的基准。

    多级回退确定唯一参照 policy：
    1) case.entities.policy_id 显式指定 → 直接按 id 找。
    2) 用 case + order 推出的 (market/topic/amount/order_status) 去 policy 数据匹配。
    3) 实在不行，看 agent 实际成功调用 policy.search 命中的 policy_id（信任其结果）。
    4) env 里只有一条 policy → 用它；否则 None。
    注意这是"verifier 认定的应当适用 policy"，与 agent 查了什么相互独立
    （即便 agent 查错，真值仍由前几级确定，保证判分客观）。
    """
    entities = case.get("entities", {}) or {}
    # policy-KB 模式：case 用 expected_policy_id 指定"答案钥匙"，直接在 KB（env.policies）里按 id 取规则。
    expected = case.get("expected_policy_id")
    if expected:
        for policy in env_snapshot.get("policies", []):
            if policy.get("policy_id") == expected:
                return policy
    policy_id = entities.get("policy_id")
    if policy_id:
        for policy in env_snapshot.get("policies", []):
            if policy.get("policy_id") == policy_id:
                return policy
    order = resolve_row("orders", "order_id", case, env_snapshot)
    metadata = case.get("metadata", {}) if isinstance(case.get("metadata"), dict) else {}
    topic = metadata.get("topic") or case.get("topic")
    args = {
        "market": case.get("market") or order.get("market"),
        "topic": topic,
        "amount": order.get("paid_amount"),
        "order_status": order.get("status"),
    }
    matched = policy_for_args(env_snapshot, args)
    if matched:
        return matched
    # 第 3 级回退：信任 agent 实际成功命中的 policy_id。
    for call in tool_calls:
        if call.get("tool") == "policy.search" and call.get("ok") is True:
            result = call.get("result") if isinstance(call.get("result"), dict) else {}
            if result.get("policy_id"):
                for policy in env_snapshot.get("policies", []):
                    if policy.get("policy_id") == result["policy_id"]:
                        return policy
    # 第 4 级回退：全 env 只有一条 policy 时直接用；否则放弃（返回 None）。
    policies = env_snapshot.get("policies", [])
    return policies[0] if len(policies) == 1 else None


def policy_for_args(env_snapshot: dict[str, Any], args: dict[str, Any]) -> dict[str, Any] | None:
    """按一组检索参数在 policy 数据里找匹配项（policy 子分与参照 policy 共用）。

    匹配规则：market/topic 给了就必须相等；amount 不得超过 policy.match.amount_max；
    order_status 给了且 policy 规定了就必须相等。返回第一条全部条件满足的 policy，无则 None。
    "给了才校验"的设计让缺省参数不会误杀候选。
    """
    for policy in env_snapshot.get("policies", []):
        if args.get("market") and policy.get("market") != args.get("market"):
            continue
        topic = args.get("topic")
        if topic and policy.get("topic") != topic:
            continue
        match = policy.get("match", {}) or {}
        if args.get("amount") is not None and match.get("amount_max") is not None:
            if args["amount"] > match["amount_max"]:
                continue
        if args.get("order_status") and match.get("order_status"):
            if args["order_status"] != match["order_status"]:
                continue
        return policy
    return None


def values_equal(actual: Any, expected: Any) -> bool:
    """容差相等比较：两边先 normalize_scalar 归一，数字按浮点比、字符串忽略大小写与空白。

    这样 "50"/"50.0"/50、"USD"/"usd"、"true"/True 等"同义不同形"能判等，
    避免因 LLM 抽值或 sandbox 存值的表面格式差异造成误判。
    """
    actual = normalize_scalar(actual)
    expected = normalize_scalar(expected)
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return float(actual) == float(expected)
    if isinstance(actual, str) and isinstance(expected, str):
        return actual.strip().lower() == expected.strip().lower()
    return actual == expected


def normalize_scalar(value: Any) -> Any:
    """把字符串形态的标量归一成原生类型，便于 values_equal 严格比较。

    处理：'true'/'false' → bool；纯数字串 → int/float；带币种后缀的金额串
    （如 '50 USD'）→ 提取数值部分。非字符串或无法识别的原样返回。
    """
    if isinstance(value, str):
        stripped = value.strip()
        lowered = stripped.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        number_match = re.fullmatch(r"[-+]?\d+(?:\.\d+)?", stripped)
        if number_match:
            numeric = float(stripped)
            return int(numeric) if numeric.is_integer() else numeric
        amount_match = re.fullmatch(
            r"[-+]?\d+(?:\.\d+)?\s*(?:USD|EUR|GBP|JPY|CNY|SGD)?",
            stripped,
            re.IGNORECASE,
        )
        if amount_match:
            numeric_text = re.match(r"[-+]?\d+(?:\.\d+)?", stripped)
            if numeric_text:
                numeric = float(numeric_text.group(0))
                return int(numeric) if numeric.is_integer() else numeric
    return value


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """把值夹到 [low, high]（默认 [0,1]）。所有子分对外都先 clamp，保证落在 0~1。"""
    return max(low, min(high, value))


class SimpleVerifier:
    """score_trajectory 的面向对象封装，方便需要持有实例的调用方。

    把 verifier_provider 存为实例状态，score() 时自动注入（调用方未显式传则用它），
    行为与直接调用 score_trajectory 完全一致。
    """

    def __init__(self, verifier_provider: ModelProvider | None = None):
        """记住要使用的外部 LLM provider（可为 None，走注入/启发式回退）。"""
        self.verifier_provider = verifier_provider

    def score(self, **kwargs: Any) -> dict[str, Any]:
        """转发到 score_trajectory；未显式给 verifier_provider 时补上实例持有的那个。"""
        kwargs.setdefault("verifier_provider", self.verifier_provider)
        return score_trajectory(**kwargs)
