# 版权所有 © 2026 深圳途明智启科技有限公司。保留所有权利。
# 未经书面许可，任何单位或个人不得复制、传播、发布、转卖、改编、仿制或用于商业用途。
# 侵权必究。

"""vLLM HTTP provider：通过 OpenAI 兼容端点访问自托管 vLLM 服务。

继承自 APIProvider 复用全部 HTTP/解析逻辑，仅额外提供"工具调用如何下发"的两条路径：
- ``native_tool_calling=True``：走 vLLM 原生 tools 字段（要求模型/部署支持函数调用）。
- ``native_tool_calling=False``（默认）：把工具清单渲染成文本菜单注入提示，让模型按约定
  的 ``<tool_call>`` 文本格式输出。这是为兼容那些未开启/不支持原生工具调用的本地模型
  （如训练中的小模型）而设的退路。
"""

from __future__ import annotations

from typing import Any

from agent.providers.api_provider import APIProvider
from agent.providers.base import ModelOutput
from agent.providers.tool_calling import render_text_tool_menu


class VLLMProvider(APIProvider):
    """vLLM provider；在基类之上增加 native vs 文本工具菜单的分流开关。"""

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "",
        native_tool_calling: bool = False,
    ):
        super().__init__(
            base_url=base_url,
            # 自托管 vLLM 通常无需鉴权
            api_key=None,
            model=model,
            provider_name="vllm",
            sanitize_tool_names=True,
        )
        # 是否使用原生工具调用；False 时改走文本工具菜单注入
        self.native_tool_calling = native_tool_calling

    def generate(
        self,
        messages_or_prompt: Any,
        sampling_config: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelOutput:
        """按 native_tool_calling 与是否有工具，分流到原生或文本菜单两条路径。"""
        if self.native_tool_calling or not tools:
            # 开启原生工具调用、或本就没有工具：直接交给基类按 OpenAI tools 字段处理
            return super().generate(messages_or_prompt, sampling_config=sampling_config, tools=tools)
        # 否则把工具渲染成文本菜单注入提示，并以 tools=None 调用（不走原生工具字段），
        # 让模型以约定文本格式自行"声明"工具调用
        return super().generate(
            inject_text_tool_menu(messages_or_prompt, tools),
            sampling_config=sampling_config,
            tools=None,
        )


def inject_text_tool_menu(messages_or_prompt: Any, tools: list[dict[str, Any]]) -> Any:
    """把工具清单渲染成文本菜单并注入到提示中（非原生工具调用路径）。

    注入策略：
    - 裸字符串提示：菜单置于提示最前。
    - 消息列表：优先把菜单追加到已有的 system 消息后（保留原 system 内容）；
      若无 system 消息，则在列表最前插入一条专门承载菜单的 system 消息。
    把工具说明放进 system 而非 user，是为了让"可用工具"成为稳定的系统级约束。
    """
    menu = render_text_tool_menu(tools)
    if isinstance(messages_or_prompt, str):
        return f"{menu}\n\n{messages_or_prompt}"
    # 浅拷贝每条消息，避免就地修改调用方对话历史
    messages = [dict(message) for message in messages_or_prompt]
    for message in messages:
        if message.get("role") == "system":
            # 追加到现有 system 之后，保留原系统提示
            message["content"] = f"{message.get('content', '')}\n\n{menu}"
            return messages
    # 没有 system 消息：在最前新增一条承载工具菜单
    return [{"role": "system", "content": menu}, *messages]
