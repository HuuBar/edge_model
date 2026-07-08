# 版权所有 © 2026 深圳途明智启科技有限公司。保留所有权利。
# 未经书面许可，任何单位或个人不得复制、传播、发布、转卖、改编、仿制或用于商业用途。
# 侵权必究。

"""工具 observation 规范化。

本模块把 toolfactory 执行工具后返回的 observation（一个普通 dict）转换成
chat 消息格式，方便 runtime 把它回放进 messages 列表；同时提供一个判定函数，
用来区分一次工具失败到底是「环境注入的错误」还是「模型自身的错误」。

为什么要区分错误来源：训练/评分侧需要知道某次失败是 env 主动注入的故障
（tool_faults，不应惩罚模型）还是模型用错了工具/参数（应计入 efficiency 等子分）。
"""

from __future__ import annotations

from typing import Any


MODEL_VISIBLE_OBSERVATION_KEYS = {
    # 工具是否成功执行。模型需要知道成功/失败，才能决定下一步是继续查、重试还是回复用户。
    "ok",
    # 成功时的业务结果，例如订单详情、物流状态、退款模拟金额。
    "result",
    # 失败时的错误码，例如 missing_required_arg、transient_error。
    "error",
    # 给模型看的简短错误说明；通常和 error 相同或更易读。
    "message",
    # 错误来源：environment 表示环境注入故障，llm 表示模型自己用错工具/参数。
    "source",
}


def project_observation_for_model(observation: dict[str, Any]) -> dict[str, Any]:
    """把完整 observation 投影成模型下一步允许看到的内容。

    ToolFactory 返回的 observation 同时服务两类读者：
    - 模型：只需要知道工具结果/错误，从而决定下一步动作。
    - 审计与 verifier：还需要 namespace_id、arguments、tool_call_id 等溯源字段。

    这里做的是“模型可见投影”：保留 agent-facing 字段，隐藏 runtime/audit 字段。
    完整 observation 仍会原样保存在 trajectory.tool_observations 里。
    """

    # 只挑白名单字段，避免 namespace_id、内部审计字段或未来新增的敏感字段泄漏进 prompt。
    projected = {
        key: observation[key]
        for key in MODEL_VISIBLE_OBSERVATION_KEYS
        if key in observation
    }
    # tool_name/tool_call_id 是 chat tool role 协议需要的最小关联字段，模型也可用它理解哪个工具返回了结果。
    projected["tool_name"] = observation.get("tool_name")
    projected["tool_call_id"] = observation.get("tool_call_id")
    return projected


def observation_message(observation: dict[str, Any]) -> dict[str, Any]:
    """把一条工具 observation 包装成 ``role="tool"`` 的 chat 消息。

    runtime 在工具执行后会把这条消息追加到 messages，下一步生成时模型即可看到
    工具返回。注意这里采用 OpenAI/Qwen3 兼容的 tool role 结构：
    Qwen3 chat_template 会把 tool role 渲染成 ``<tool_response>...</tool_response>``，
    因此 runtime 不需要手写自定义的历史拼接文本。

    - ``tool_call_id``/``name`` 用 ``.get`` 取，缺失时为 None，保证不会因 observation
      字段不全而抛异常（容错优先，宁可消息字段为空也不要中断 agent loop）。
    - ``content`` 只放 agent-facing observation 投影；完整 observation 仍在 trajectory 的
      ``tool_observations`` 中保存，供 verifier/audit 使用。
    """
    # 返回结构保持 chat message 形状；content 是 dict 而不是字符串，由 tokenizer/chat_template
    # 决定最终如何渲染成 <tool_response>。
    return {
        "role": "tool",
        "tool_call_id": observation.get("tool_call_id"),
        "name": observation.get("tool_name"),
        "content": project_observation_for_model(observation),
    }


def is_environment_error(observation: dict[str, Any]) -> bool:
    """判断一条失败 observation 是否为「环境注入」错误。

    判定条件同时满足两点：``ok`` 明确为 False（失败），且 ``source`` 标记为
    ``"environment"``（由 env 的 tool_faults 故障注入产生，而不是 ``source="llm"``
    的模型参数/格式错误）。这两类失败在 verifier 里待遇不同：环境错误通常不应
    惩罚模型，模型错误才计入过程子分。
    """
    # 必须同时看 ok 和 source。只看 source 可能把成功 observation 误判为错误；
    # 只看 ok 又无法区分环境故障和模型错误。
    return observation.get("ok") is False and observation.get("source") == "environment"
