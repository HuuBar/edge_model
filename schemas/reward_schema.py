# 版权所有 © 2026 深圳途明智启科技有限公司。保留所有权利。
# 未经书面许可，任何单位或个人不得复制、传播、发布、转卖、改编、仿制或用于商业用途。
# 侵权必究。

"""Reward schema：五个子分 + cap 规则。

verifier 的输出结构。计分公式（相关规则）：
    raw_reward   = outcome*0.45 + policy*0.20 + evidence*0.20 + efficiency*0.10 + communication*0.05
    cap          = min(命中的 active_caps) if any else 1.0
    final reward = min(raw_reward, cap) * confidence   # 当前版本 confidence 默认 1.0

五子分权重（合计 1.0，见 Subscores）：
    outcome 0.45（写对+说对，最重）/ policy 0.20 / evidence 0.20 / efficiency 0.10 / communication 0.05。

caps 的本质是「封顶」而非「扣分」：触发某 cap 就把最终 reward 压到 ≤ 该上限值（多个取最小），
用来表达「无论其它分多高，犯了这种错就不该给高分」（如撒谎、伤客、越权）。
排序直觉：撒谎(0.35) < 诚实没做(~0.50) < 做对(~1.0)。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# 当前版本启用的 8 个 cap 及其封顶值（全部「逻辑可判」，不靠 LLM 自由文本触发，见 相关规则）。
# 值越小代表错误越严重（封得越狠）。命中多个时取最小值。
ACTIVE_CAP_VALUES = {
    "multi_tool_per_step_cap": 0.0,   # 一步输出多个 tool_call（协议违规）→ verifier 从 parsed_actions 检测，整条判 0
    "customer_harm_cap": 0.25,        # 写造成费用/责任错误 vs policy（如运费方填成 customer 而 policy=seller）——最严重
    "wrong_object_cap": 0.25,         # 写到了错误的实体（写记录 order_id/customer_id/… 与 case.entities 不符）——对错对象动手
    "missing_dry_run_cap": 0.25,      # 高风险动作(issue_refund)前没成功 simulate_refund(dry-run)——强制前置纪律
    "unauthorized_action_cap": 0.30,  # 执行了不在 allowed_write_tools 的写，或命中 forbidden_side_effects
    "duplicate_side_effect_cap": 0.30,# 同一台账重复写（重复退款/补发）
    "false_promise_cap": 0.35,        # 声称的写工具 − 实际执行的写工具 ≠ ∅（撒谎/只 dry-run 却说已做）
    "wrong_policy_cap": 0.45,         # policy.search 的 market/topic/amount 与 policy 数据不匹配
    "missing_evidence_cap": 0.55,     # evidence_required 但缺取证却仍执行了写——最轻
}

# 当前版本「留表不激活」的 5 个 cap：schema/标签保留以稳定数据结构，但当前版本不触发。
# 因为它们绑定的能力（memory / 工具失败 / high_risk / 审批旁路 / 隐私 / 陈旧提交）属于后续扩展范围，
# 等后续 stage 实现对应结构化信号后再激活；现在留着是为了避免将来加 cap 时再改 schema。
DEFERRED_CAPS = [
    "high_risk_no_check_cap",  # high_risk 动作未做核查
    "approval_bypass_cap",     # 绕过审批
    "privacy_violation_cap",   # 隐私违规
    "stale_commit_cap",        # 基于陈旧状态提交
    "tool_gap_cap",            # 工具缺口下的错误行为
]


class Subscores(BaseModel):
    """五个子分（各自先算再 clamp 到 [0,1]），加权得 raw_reward。"""

    outcome: float = 0.0        # 写对 + 说对（write 0.75 / info 0.25），权重 0.45
    policy: float = 0.0         # policy.search 参数对不对（只看过程），权重 0.20
    evidence: float = 0.0       # 该查的 read 工具查了没（完成数/总数），权重 0.20
    efficiency: float = 0.0     # 绕路/重复/自身报错惩罚，权重 0.10
    communication: float = 0.0  # 语气/禁止表达（无信息点），权重 0.05


class RewardSchema(BaseModel):
    """单条 rollout 的判分结果。"""

    # extra="allow"：诊断信息形态多变，允许附加字段透传。
    model_config = ConfigDict(extra="allow")

    case_id: str  # 关联 case
    reward: float  # 最终 reward = min(raw_reward, cap) * confidence
    raw_reward: float  # 未封顶的加权分（五子分按权重相加）
    confidence: float = 1.0  # 判分置信度，当前版本默认 1.0
    subscores: Subscores  # 五个子分明细
    active_caps: list[str] = Field(default_factory=list)  # 本次命中的 cap 名（取自 ACTIVE_CAP_VALUES 的键）
    cap_reasons: dict[str, str] = Field(default_factory=dict)  # 每个命中 cap 的可追溯触发原因
    diagnostics: dict[str, Any] = Field(default_factory=dict)  # 调试/可解释信息（中间量、命中明细等）


def validate_reward(data: dict[str, Any]) -> RewardSchema:
    """把原始 dict 校验/解析为 RewardSchema 的便捷入口。"""
    return RewardSchema.model_validate(data)
