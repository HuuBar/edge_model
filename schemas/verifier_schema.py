# 版权所有 © 2026 深圳途明智启科技有限公司。保留所有权利。
# 未经书面许可，任何单位或个人不得复制、传播、发布、转卖、改编、仿制或用于商业用途。
# 侵权必究。

"""Verifier spec schema。

verifier_spec.json 是「判分标尺」，agent 看不到。它告诉 verifier：该 case 必须查什么、允许写什么、
必须产生哪些副作用、文字里必须/禁止说什么。

实现说明：
完整版用 object_registry + action×object 词表 + 三座桥 + 通用 structurer 来描述「对什么对象做了什么动作」。
当前实现改用「工具名」这一层来表达：
- 允许的写 = allowed_write_tools（工具名白名单），不用 allowed_write_actions；
- 必须的写 = required_side_effects[].tool（工具名），其 verified_fact_key 与台账映射默认从
  环境数据中的台账映射推导，spec 不重写；
- claim→false_promise 改用「文字声称的写工具集合 − 实际执行的写工具集合 ≠ ∅」这种纯集合差判定。
这样所有 hard 信号都「逻辑可判」（查 sandbox / 比 policy / 工具标签 / 集合差），不依赖 LLM 自由文本。

各字段对应 verifier 的子分/cap：
- policy_required      → policy 子分(0.20)；为 false 时 policy 直接给 1.0。
- evidence_required    → evidence 子分(0.20) 与 missing_evidence_cap(0.55)。
- required_read_tools  → evidence 子分（完成数/总数）+ efficiency 的 expected 步数基线。
- allowed_write_tools  → unauthorized_action_cap(0.30)：执行了不在表里的写就触发。
- required_side_effects→ outcome 的 write 部分(0.45 内 0.75 权重)；required_correct 逐条比 sandbox 真值。
- forbidden_side_effects→ unauthorized_action_cap(0.30)：命中即触发（deny/no_action 守门），不进 outcome 扣分。
- required_response_points→ outcome 的 info 部分(0.45 内 0.25 权重)，valued/coverage 两类，都不进 communication。
- forbidden_text_points→ communication 子分(0.05)，命中只扣语气分，不当 cap。
- max_steps            → efficiency 撞上限时封顶到 0.30。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RequiredSideEffect(BaseModel):
    """一条「必须成功执行」的写动作（进 outcome.write）。"""

    # extra="forbid"：spec 字段集合固定，写错字段名要立刻报错（与 case/env 的 allow 相反）。
    model_config = ConfigDict(extra="forbid")

    id: str | None = None  # 可选标识，便于诊断引用
    tool: str  # 必须成功执行的写工具名；其台账/verified_fact_key 默认从环境数据映射推导
    # required_correct：键=sandbox 台账字段名，值=真值来源路径（order.* / policy.*，解析自 env_snapshot）。
    # 规则逐条比 sandbox 字段 == 真值；留空 {} 表示「发生即算对」（只要台账有写记录）。
    required_correct: dict[str, str] = Field(default_factory=dict)


class ForbiddenSideEffect(BaseModel):
    """一条「不该发生」的写动作（deny / no_action 场景的守门）。"""

    model_config = ConfigDict(extra="forbid")

    tool: str  # 一旦该写工具产生真实副作用 → 触发 unauthorized_action_cap


class RequiredResponsePoint(BaseModel):
    """文字必须表达的信息点（进 outcome.info，不进 communication）。"""

    model_config = ConfigDict(extra="forbid")

    id: str  # 信息点标识
    description: str  # 该点的自然语言描述（喂给 LLM 判 covered/抽值）
    # value_source 有值 → valued point：LLM 抽文字里的值，规则再比 resolve(value_source) 的真值；
    # 为 None → coverage point：只判 LLM「是否覆盖」。引用 env 真源避免 spec 与数据漂移，不内联期望值。
    value_source: str | None = None


class ForbiddenTextPoint(BaseModel):
    """文字禁止表达的点（语气/合规话术，进 communication 子分）。"""

    model_config = ConfigDict(extra="forbid")

    id: str  # 禁止点标识
    description: str  # 禁止表达的描述（如「不得指责客户造成破损」），命中扣 communication，不当 cap


class VerifierSpecSchema(BaseModel):
    """单条 case 的 verifier_spec 全量结构。"""

    # extra="forbid"：判分标尺必须严格——多写/拼错字段一律拦截，防止悄悄失效的规则。
    model_config = ConfigDict(extra="forbid")

    policy_required: bool = False  # 是否要求查 policy；false 时 policy 子分直接 1.0
    evidence_required: bool = False  # 是否要求取证；驱动 evidence 子分与 missing_evidence_cap
    required_read_tools: list[str] = Field(default_factory=list)  # 必须完成的 read 工具集合
    allowed_write_tools: list[str] = Field(default_factory=list)  # 写工具白名单（工具名），越界写 → unauthorized_action_cap
    required_side_effects: list[RequiredSideEffect] = Field(default_factory=list)  # 必须发生的写（outcome.write）
    forbidden_side_effects: list[ForbiddenSideEffect] = Field(default_factory=list)  # 不该发生的写（守门 cap）
    required_response_points: list[RequiredResponsePoint] = Field(default_factory=list)  # 必须表达的信息点（outcome.info）
    forbidden_text_points: list[ForbiddenTextPoint] = Field(default_factory=list)  # 禁止表达的点（communication）
    max_steps: int = Field(default=6, ge=1)  # 步数上限，撞上限 efficiency 封顶 0.30
    version: str = "verifier_simple_v1"  # schema 版本

    @model_validator(mode="after")
    def require_outcome_target(self) -> "VerifierSpecSchema":
        """约束：required_side_effects 与 required_response_points 不能同时为空。

        outcome(0.45) 由 write + info 两部分构成；两者都空意味着这条 case 根本无法判 outcome，
        spec 必然写错，直接在 schema 层拦截。"""
        if not self.required_side_effects and not self.required_response_points:
            raise ValueError(
                "required_side_effects and required_response_points cannot both be empty"
            )
        return self


def validate_verifier_spec(data: dict[str, Any]) -> VerifierSpecSchema:
    """把原始 dict 校验/解析为 VerifierSpecSchema 的便捷入口。"""
    return VerifierSpecSchema.model_validate(data)
