"""数据批次清单 schema。

用于按数据批次组织产物：记录批次来源、内容哈希、条目列表，
支持数据版本追溯与完整性校验。schema 保持宽松，便于后续扩展字段。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ManifestSchema(BaseModel):
    """单个数据批次清单。"""

    # extra="allow"：字段会随数据流程扩展，允许透传。
    model_config = ConfigDict(extra="allow")

    manifest_id: str  # 批次标识
    version: str  # 清单版本
    source: str | None = None  # 数据来源
    hashes: dict[str, str] = Field(default_factory=dict)  # 文件名→内容哈希，用于完整性校验
    entries: list[dict[str, Any]] = Field(default_factory=list)  # 批次内条目列表
