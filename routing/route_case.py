"""路由阶段：把一个 case 的（K 条 rollout 聚合后）指标路由到训练/评估池。

决策树严格按 设计要求：先判可自动判定的工程/数据 bug（quarantine），再判 tool_gap，
再按 parse/基础步骤/能力/spread 分流到 sft/rl/eval/long_async/later/risk_memory，否则 more_probe。

route_case 不调用模型、不做业务直觉判断；只吃结构化指标 + case/spec/registry 的静态事实。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from schemas.route_schema import RouteDecisionSchema

# route 名 -> pool 名。more_probe 不落池。
# generated: 0a512 09M59
ROUTE_TO_POOL: dict[str, str | None] = {
    "quarantine": "pool_quarantine",
    "tool_gap": "pool_tool_gap",
    "sft_parser_format": "pool_sft_parser_format",
    "sft_curriculum": "pool_sft_curriculum",
    "later_recheck": "pool_later_recheck",
    "all_high_eval": "pool_all_high_eval",
    "rl_main": "pool_rl_main",
    "risk_memory": "pool_risk_memory",
    "long_async": "pool_long_async",
    "more_probe": None,
}

# 长程/弱模型类任务的判定（撞 max_steps/stale 高，或天然多步），用于 later_recheck 归因。
LONG_HORIZON_AXES = {"long_latency", "async_required", "multi_item"}


@dataclass(frozen=True)
class Thresholds:
    """routing 阈值（默认值与 configs/routing.yaml 一致，改一处同步另一处）。"""
    high_reward: float = 0.80
    low_reward: float = 0.30
    rl_spread_min: float = 0.30
    all_high_spread_max: float = 0.10
    parse_error_high: float = 0.50
    tool_error_high: float = 0.50
    max_step_hit_high: float = 0.50
    stale_high: float = 0.50


def route_case(
    *,
    case: dict[str, Any],
    verifier_spec: dict[str, Any],
    metrics: dict[str, Any],
    tool_registry: set[str],
    gold_reward: float | None = None,
    integrity: dict[str, Any] | None = None,
    thresholds: Thresholds | None = None,
) -> RouteDecisionSchema:
    """对单个 case 做路由决策，返回 RouteDecisionSchema。

    integrity：可选的工程一致性事实（probe 落盘时算好），缺省则现场从 case/spec/metrics 推。
        id_mismatch / version_or_hash_mismatch / sandbox_missing_for_success 等布尔。
    """
    t = thresholds or Thresholds()
    integrity = integrity or {}
    reasons: list[str] = []
    metadata = case.get("metadata", {}) if isinstance(case.get("metadata"), dict) else {}
    axes = set(metadata.get("control_axis") or case.get("control_axis", []) or [])

    def decide(route: str, why: str) -> RouteDecisionSchema:
        reasons.append(why)
        return RouteDecisionSchema(case_id=case.get("case_id", ""), route=route, reasons=reasons, metrics=metrics)

    # —— 1) quarantine：可自动判定的工程/数据 bug——
    if integrity.get("id_mismatch"):
        return decide("quarantine", "case/env/verifier id 不一致")
    missing_entity = _missing_entities(case, integrity)
    if missing_entity:
        return decide("quarantine", f"required entity 缺失: {missing_entity}")
    if gold_reward is not None and gold_reward < t.high_reward:
        return decide("quarantine", f"seed/gold 不能 replay 到高分（gold_reward={gold_reward}）")
    if integrity.get("sandbox_missing_for_success"):
        return decide("quarantine", "成功写动作缺 sandbox 副作用")
    if integrity.get("version_or_hash_mismatch"):
        return decide("quarantine", "group 内 version/hash 不一致")

    # —— 2) tool_gap：gold path 需要的写工具 registry 表达不了 ——
    spec_write_tools = {se.get("tool") for se in verifier_spec.get("required_side_effects", [])}
    missing_tools = sorted(t for t in spec_write_tools if t and t not in tool_registry)
    if missing_tools:
        return decide("tool_gap", f"gold 需要的写工具不在 registry: {missing_tools}")

    # —— 3) 失败归因：parse / 基础步骤 ——
    if metrics.get("parse_error_rate", 0.0) >= t.parse_error_high:
        return decide("sft_parser_format", f"parse_error_rate={metrics['parse_error_rate']} 偏高")
    if metrics.get("tool_error_rate", 0.0) >= t.tool_error_high:
        return decide("sft_curriculum", f"tool_error_rate={metrics['tool_error_rate']} 偏高（基础步骤反复失败）")

    max_r = metrics.get("max_reward", 0.0)
    spread = metrics.get("reward_spread", 0.0)
    min_r = metrics.get("min_reward", 0.0)

    # —— 4) 全低分 + 长程/弱模型 -> later_recheck（数据有效但当前模型不够）——
    long_horizon = bool(axes & LONG_HORIZON_AXES) or metrics.get("max_step_hit_rate", 0.0) >= t.max_step_hit_high
    if max_r < t.low_reward and long_horizon:
        return decide("later_recheck", f"max_reward={max_r} 低且长程/弱模型，留待训练后重 probe")

    # —— 5) 全高、区分度低 -> all_high_eval（降采样/eval）——
    if min_r >= t.high_reward and spread <= t.all_high_spread_max:
        return decide("all_high_eval", f"min_reward={min_r} 高且 spread={spread} 小（区分度低）")

    # —— 6) 有 spread 且有高分 -> rl_main（GRPO 主训练）——
    if spread >= t.rl_spread_min and max_r >= t.high_reward:
        return decide("rl_main", f"spread={spread}≥{t.rl_spread_min} 且 max_reward={max_r}≥{t.high_reward}")

    # —— 6.5) base 一致中低分（从不达高分）+ spread 小 + 无 parse/tool 错 -> sft_curriculum ——
    # "模型太弱但数据有效"的一类：执行干净（非格式/工具失败），但稳定够不到成功线，
    # 多为缺少某项判断/取证技能（如没查客户、没套政策阈值）。有 gold 路径可教 → SFT 课程。
    # 放在 rl_main 之后：只要有一条高分 rollout 就优先 RL，否则才判定"需教"。
    if (max_r < t.high_reward and spread <= t.all_high_spread_max
            and metrics.get("parse_error_rate", 0.0) < t.parse_error_high
            and metrics.get("tool_error_rate", 0.0) < t.tool_error_high):
        return decide("sft_curriculum", f"max_reward={max_r}<{t.high_reward} 且 spread={spread} 小、执行干净（无 parse/tool 错）→ 教正确路径")

    # —— 7) 高风险/记忆专项 ——
    if axes & {"high_risk", "fraud_or_abuse"} or "memory_required" in axes:
        return decide("risk_memory", f"高风险/记忆类轴: {sorted(axes & {'high_risk', 'fraud_or_abuse', 'memory_required'})}")

    # —— 8) 长任务/异步 ——
    if metrics.get("max_step_hit_rate", 0.0) >= t.max_step_hit_high or metrics.get("stale_rate", 0.0) >= t.stale_high:
        return decide("long_async", "max_step/stale 比例高")

    # —— 9) 兜底：继续探 ——
    return decide("more_probe", "未命中明确路由，继续 probe")


def _missing_entities(case: dict[str, Any], integrity: dict[str, Any]) -> list[str]:
    """优先用 probe 算好的 missing_entities；否则返回空（构造期已校验实体存在）。"""
    given = integrity.get("missing_entities")
    return list(given) if given else []
