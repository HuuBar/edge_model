"""路由决策 schema。

预留给后续按 case 特征把样本分流到不同处理/训练路径的决策记录。
当前 schema 保持宽松，以便后续扩展字段。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RouteDecisionSchema(BaseModel):
    """单条路由决策。"""

    # extra="allow"：字段会随路由策略扩展，允许透传。
    model_config = ConfigDict(extra="allow")

    case_id: str  # 关联 case
    route: str  # 分流到的路径名
    reasons: list[str] = Field(default_factory=list)  # 选该路径的依据
    metrics: dict[str, Any] = Field(default_factory=dict)  # 路由相关指标
