"""售后 case schema。

case.json 同时承载离线 join 字段与 agent-facing 输入。runtime 只会投影线上 agent
真实可见的字段（ticket/customer/market/order/message）进 prompt；metadata、entities
和期望策略等字段只给 verifier/routing/builder 使用。

设计要点：
- extra="allow"：故意放宽。case 在不同业务主题下会带各自特有的业务字段（例如某些 case
  额外挂 risk/memory 片段、渠道信息等），又处于 当前版本 持续演进期，schema 不想每加一个
  业务字段就改一次。允许未知字段透传，避免 validator 频繁拦截合法数据；核心字段仍强校验。
- field_validator 只校验三个「分类学枚举」字段（primary_intent/axis/difficulty），把脏标签挡在数据
  入库前，保证下游按 taxonomy 做分桶/统计时不出现拼写漂移。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from schemas.taxonomy import CONTROL_AXES, DIFFICULTIES, PRIMARY_INTENTS


class CaseSchema(BaseModel):
    """单条售后 case 的结构。"""

    # 允许携带 schema 未声明的业务字段透传，原因见模块 docstring。
    model_config = ConfigDict(extra="allow")

    case_id: str  # 离线 case 唯一标识，贯穿 env/verifier/trajectory/reward 做 join；不进 prompt
    ticket_id: str | None = None  # agent-facing 会话 id；builder/runtime 会补 opaque id
    metadata: dict[str, Any] = Field(default_factory=dict)  # 离线分类/采样/期望 policy 信息；不进 prompt
    primary_intent: str | None = None  # 兼容旧字段；新数据优先放 metadata.primary_intent
    control_axis: list[str] = Field(default_factory=list)  # 兼容旧字段；新数据优先放 metadata.control_axis
    difficulty: str | None = None  # 兼容旧字段；新数据优先放 metadata.difficulty
    market: str  # 客户市场/区域，会作为会话上下文给 agent
    customer_id: str | None = None  # 可选顶层 agent-facing 字段；也可放 entities.customer_id
    order_id: str | None = None  # 仅当当前会话已绑定或客户明确提供时给 agent
    customer_message: str  # 客户原始诉求文本，agent 的主要输入
    entities: dict[str, Any] = Field(default_factory=dict)  # 全量实体，仅给 verifier/value_source 与 runtime 可见投影源
    max_steps: int = Field(default=20, ge=1)  # 步数上限；撞上限会触发 efficiency 封顶
    expected_resolution: str | None = None  # 期望的处置结果描述（人读/参考用，非判分依据）
    version: str = "case_v1"  # schema 版本，便于数据演进时区分
    expected_policy_id: str | None = None  # policy-KB 真值钥匙；给 verifier，不进 prompt

    @field_validator("primary_intent")
    @classmethod
    def known_intent(cls, value: str | None) -> str | None:
        """校验主意图必须是 taxonomy 已登记的 23 个意图之一，挡住拼写漂移。"""
        if value is not None and value not in PRIMARY_INTENTS:
            raise ValueError(f"unknown primary_intent: {value}")
        return value

    @field_validator("control_axis")
    @classmethod
    def known_axes(cls, values: list[str]) -> list[str]:
        """校验所有控制轴标签都在 taxonomy.CONTROL_AXES 内；列出全部未知项便于定位。"""
        unknown = sorted(set(values) - CONTROL_AXES)
        if unknown:
            raise ValueError(f"unknown control_axis: {unknown}")
        return values

    @field_validator("difficulty")
    @classmethod
    def known_difficulty(cls, value: str | None) -> str | None:
        """校验难度必须是 L1–L5 之一。"""
        if value is not None and value not in DIFFICULTIES:
            raise ValueError(f"unknown difficulty: {value}")
        return value


def validate_case(data: dict[str, Any]) -> CaseSchema:
    """把原始 dict 校验/解析为 CaseSchema 的便捷入口。"""
    return CaseSchema.model_validate(data)
