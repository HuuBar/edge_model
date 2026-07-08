# 版权所有 © 2026 深圳途明智启科技有限公司。保留所有权利。
# 未经书面许可，任何单位或个人不得复制、传播、发布、转卖、改编、仿制或用于商业用途。
# 侵权必究。

"""已执行 rollout 轨迹的 schema。

轨迹是 agent 在某个 case 上跑一遍的完整记录，是 verifier 的核心输入之一：
- final_text 喂给合并的 verifier LLM（判 response point + 抽 claimed_write_tools）；
- sandbox_final_state 供规则查台账，判「写做没做/做对没」以及算 executed_write_tools 集合；
- parsed_actions / tool_observations / tool_errors 供 evidence、efficiency、policy 等规则子分使用。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TrajectorySchema(BaseModel):
    """单条已执行轨迹。"""

    # extra="allow"：rollout 记录器可能附加 token/timing 等元数据，允许透传。
    model_config = ConfigDict(extra="allow")

    case_id: str  # 关联 case
    run_id: str  # 一次评测/训练运行的标识
    rollout_id: str  # 该次 rollout 的标识（同 case 可多 rollout）
    namespace_id: str  # 沙箱 namespace 标识（隔离各 rollout 的副作用）
    prompt_history: list[dict[str, Any]] = Field(default_factory=list)  # 完整对话/提示历史
    raw_model_outputs: list[dict[str, Any]] = Field(default_factory=list)  # 模型逐步原始输出
    parsed_actions: list[dict[str, Any]] = Field(default_factory=list)  # 解析出的工具调用动作（含参数）
    tool_observations: list[dict[str, Any]] = Field(default_factory=list)  # 工具返回的观测
    tool_errors: list[dict[str, Any]] = Field(default_factory=list)  # 工具报错（需区分 LLM 自身错 vs 环境注入错）
    final_text: str = ""  # 最终对客回复文本（verifier LLM 的主输入）
    sandbox_final_state: dict[str, Any] = Field(default_factory=dict)  # rollout 结束时的沙箱终态（查台账判 outcome）
    model_metadata: dict[str, Any] = Field(default_factory=dict)  # 模型/采样元信息


def validate_trajectory(data: dict[str, Any]) -> TrajectorySchema:
    """把原始 dict 校验/解析为 TrajectorySchema 的便捷入口。"""
    return TrajectorySchema.model_validate(data)
