# 版权所有 © 2026 深圳途明智启科技有限公司。保留所有权利。
# 未经书面许可，任何单位或个人不得复制、传播、发布、转卖、改编、仿制或用于商业用途。
# 侵权必究。

"""极简的 OpenAI 兼容 API provider（仅用标准库 urllib，无第三方 SDK 依赖）。

任何暴露 OpenAI ``/chat/completions`` 接口的服务（含 vLLM 等自托管端点）都可通过本类
访问。刻意不依赖 openai SDK，以降低运行环境依赖、便于在受限/离线训练机上使用。
"""

from __future__ import annotations

import json
import urllib.request
import urllib.error
from typing import Any

from agent.providers.base import ModelOutput
from agent.providers.tool_calling import normalize_openai_tool_calls, prepare_openai_tools


# generated: 0260c 01Ocd
class APIProvider:
    """基于 urllib 的 OpenAI 兼容客户端，是 vLLM 等 provider 的基类。"""

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        model: str = "",
        provider_name: str = "api",
        sanitize_tool_names: bool = True,
    ):
        # 去掉末尾斜杠，统一拼接路径时的形态
        self.base_url = base_url.rstrip("/")
        # 自托管端点可能无需鉴权，故 api_key 允许为 None
        self.api_key = api_key
        self.model = model
        # provider 标识，写入 ModelOutput 以追溯来源（子类如 vllm 会覆盖）
        self.provider_name = provider_name
        # 是否对工具名做"安全化"转义（工具名含点号等字符时需要，详见 tool_calling.py）
        self.sanitize_tool_names = sanitize_tool_names

    def generate(
        self,
        messages_or_prompt: Any,
        sampling_config: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelOutput:
        """发起一次 chat/completions 调用并归一化为 ModelOutput。

        流程：工具名安全化 -> 组装 payload（合并采样参数）-> 发 HTTP -> 解析响应 ->
        把工具名还原回原始名并归一化工具调用。
        """
        # 按需安全化工具名，并拿到 安全名->原始名 的反查表，供回程还原工具名
        converted_tools, reverse_map = (
            prepare_openai_tools(tools) if self.sanitize_tool_names else (tools, {})
        )
        payload = {"model": self.model, "messages": normalize_openai_messages(messages_or_prompt)}
        # 采样参数直接并入 payload（OpenAI 兼容字段如 temperature/top_p/max_tokens）
        payload.update(sampling_config or {})
        if converted_tools:
            payload["tools"] = converted_tools
            # 让模型自行决定是否调用工具
            payload.setdefault("tool_choice", "auto")
            # 默认关闭并行工具调用：agent 更偏向「观察后再决策」的串行轨迹。
            # 若调用方显式打开或后端仍返回多个，runtime 会按返回顺序逐个执行并回放多条 observation。
            payload.setdefault("parallel_tool_calls", False)
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                # 有 api_key 才带 Bearer 头；自托管端点可省略
                **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}),
            },
            method="POST",
        )
        try:
            # 同步阻塞请求，给足 120s 超时（生成可能较慢）
            with urllib.request.urlopen(request, timeout=120) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            # 把服务端返回的错误体读出来一并抛出，便于定位（如鉴权失败、参数非法）
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{self.provider_name} HTTP {exc.code}: {body}") from exc
        # 仅取第一个候选（n=1 场景）
        message = data["choices"][0]["message"]
        # content 可能为 None（纯工具调用时），统一兜底为空串
        text = message.get("content") or ""
        # 用反查表把安全工具名还原成原始名，并归一化为统一结构
        tool_calls = normalize_openai_tool_calls(message, reverse_map)
        return ModelOutput(
            raw_text=text,
            tool_calls=tool_calls,
            assistant_message=message,
            token_usage=data.get("usage", {}),
            model_name=self.model,
            provider=self.provider_name,
            sampling_config=sampling_config or {},
            # 记录服务端实际返回的模型名（可能与配置名不同）
            served_model_name=data.get("model"),
        )


def normalize_openai_messages(messages_or_prompt: Any) -> list[dict[str, Any]]:
    """把输入归一化为 OpenAI chat 接口可接受的 messages 列表。

    - 裸字符串：包成单条 user 消息。
    - 消息列表：逐条浅拷贝后规整 content 字段，确保最终都是字符串（或 None），
      因为 OpenAI 兼容接口的 content 必须是字符串而非任意对象。
    """
    if isinstance(messages_or_prompt, str):
        return [{"role": "user", "content": messages_or_prompt}]
    normalized = []
    for message in messages_or_prompt:
        # 浅拷贝，避免就地修改调用方传入的对话历史
        item = dict(message)
        if item.get("role") == "tool":
            content = item.get("content")
            # 工具结果若是结构化对象，序列化成 JSON 字符串（ensure_ascii=False 保留中文）
            item["content"] = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
            # OpenAI 的 tool 消息不接受 name 字段，移除以免报错
            item.pop("name", None)
        elif not isinstance(item.get("content"), (str, type(None))):
            # 非 tool 消息但 content 是对象时，同样序列化为字符串
            item["content"] = json.dumps(item.get("content"), ensure_ascii=False)
        normalized.append(item)
    return normalized
