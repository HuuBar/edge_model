"""Trajectory 数据结构。

Trajectory 是 agent runtime 一次 rollout 的完整产物。
它把「生成时发生的一切」结构化保存下来，供后续 sandbox 落盘、verifier 评分、
routing 分流、以及训练侧 consistency audit 使用。一条 trajectory 既是评分对象，
也是训练样本来源，因此字段必须足够完整、且全部可 JSON 序列化。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Trajectory:
    """一次 rollout 的完整轨迹记录。

    前七个字段是「身份 + 版本指纹」，必须在创建时给定（无默认值）：
    - ``case_id`` / ``run_id`` / ``rollout_id``：定位这条轨迹属于哪个 case、哪次 run、
      哪条 rollout；三者也共同决定 sandbox 隔离用的 namespace_id。
    - ``namespace_id``：sandbox 写隔离键（run+case+rollout），保证并发 rollout 之间
      互不污染可变状态。
    - ``prompt_template_version`` / ``prompt_hash`` / ``tool_schema_hash``：版本与哈希
      指纹，是 GRPO group 同质性校验和 rollout/training 一致性审计的关键。

    其余字段带默认值（空容器），在 agent loop 推进过程中逐步填充：
    - ``prompt_history``：★必存。每一步真正喂给模型的 messages + tool_schemas + 哈希
      快照。没有它，训练侧无法还原模型当时看到的输入，也就无法做 rollout/training
      consistency audit。
    - ``raw_model_outputs``：每步模型原始输出（raw_text / tool_calls / 元数据），
      训练样本保留模型原生格式，不被 runtime 的统一解析覆盖。
    - ``parsed_actions``：runtime 解析出的结构化工具调用（带 step / tool_call_id）。
    - ``tool_observations``：每次工具执行返回的 observation（成功与失败都记）。
    - ``tool_errors``：失败子集（解析错误 + 工具执行 ok=False），方便统计
      parse_error_rate / tool_error_rate 等 route 指标。
    - ``final_text``：面向客户的最终回复文本（无 tool_call 即终止时的输出；
      若撞到 max_steps 未自然终止，则置空，表示这条 rollout 没有给出最终回复）。
    - ``model_metadata``：模型/采样元信息（取首步），用于版本对齐。
    - ``sandbox_final_state``：rollout 结束时 sandbox 的最终状态，verifier 据此判定
      写动作（required_side_effects）是否真实发生且正确。
    """

    # 身份字段：三者共同定位“哪次运行、哪个 case、哪条 rollout”。
    case_id: str
    run_id: str
    rollout_id: str
    # sandbox 隔离键：写工具落台账时会带上它，防止并发 rollout 互相污染。
    namespace_id: str
    # 版本/一致性指纹：训练侧用它们确认 rollout 时看到的 prompt 和工具 schema 没变。
    prompt_template_version: str
    prompt_hash: str
    tool_schema_hash: str
    # 每一步喂给模型的完整输入快照。注意这是 list[dict]，不是最终 prompt 字符串，
    # 因为 chat_template 渲染可能由 tokenizer/model 负责，训练侧需要保留结构化 messages。
    prompt_history: list[dict[str, Any]] = field(default_factory=list)
    # 每一步模型原始输出，保留 raw_text / provider metadata / 原生 tool_calls。
    raw_model_outputs: list[dict[str, Any]] = field(default_factory=list)
    # runtime 从模型输出中解析出的工具动作；这是“模型想做什么”的结构化记录。
    parsed_actions: list[dict[str, Any]] = field(default_factory=list)
    # toolfactory 执行工具后返回的完整 observation；这是“环境实际返回什么”。
    tool_observations: list[dict[str, Any]] = field(default_factory=list)
    # tool_observations 中的失败子集，再加上 parse_error 这类未执行工具的模型格式错误。
    tool_errors: list[dict[str, Any]] = field(default_factory=list)
    # 自然终止时的最终客户回复；撞 max_steps 时保持空字符串，表示没有完成回复。
    final_text: str = ""
    # 模型/采样/adapter 等元信息，通常取首步输出，避免每步重复。
    model_metadata: dict[str, Any] = field(default_factory=dict)
    # rollout 结束时的 sandbox 台账快照。verifier 判写动作 outcome 主要看这里。
    sandbox_final_state: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """导出为普通 dict，便于 JSON 落盘（trajectory.json）。

        显式逐字段列出而非用 ``dataclasses.asdict``，一是固定输出键集合与顺序，
        让落盘文件稳定可比；二是明确「对外契约」就是这些字段，新增内部字段不会
        意外泄漏到序列化结果里。"""
        return {
            # 身份 + 版本指纹区。
            "case_id": self.case_id,
            "run_id": self.run_id,
            "rollout_id": self.rollout_id,
            "namespace_id": self.namespace_id,
            "prompt_template_version": self.prompt_template_version,
            "prompt_hash": self.prompt_hash,
            "tool_schema_hash": self.tool_schema_hash,
            # runtime 过程记录区。
            "prompt_history": self.prompt_history,
            "raw_model_outputs": self.raw_model_outputs,
            "parsed_actions": self.parsed_actions,
            "tool_observations": self.tool_observations,
            "tool_errors": self.tool_errors,
            # 终止结果与审计区。
            "final_text": self.final_text,
            "model_metadata": self.model_metadata,
            "sandbox_final_state": self.sandbox_final_state,
        }
