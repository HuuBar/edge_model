# 版权所有 © 2026 深圳途明智启科技有限公司。保留所有权利。
# 未经书面许可，任何单位或个人不得复制、传播、发布、转卖、改编、仿制或用于商业用途。
# 侵权必究。

"""模型生成的 Provider（供应方）统一接口模块。

本模块定义了整个 agent 与底层大模型交互的"契约层"：

- ``ModelOutput``：所有 provider（OpenAI / Anthropic / vLLM / 本地 HF 等）一次生成
  调用的统一返回结构。无论底层走的是哪家 API、协议如何不同，最终都被收敛成同一个
  数据类，这样上层（rollout / 训练 / verifier）就只需面对一种数据形态。
- ``ModelProvider``：用 ``Protocol`` 描述的鸭子类型接口（structural typing）。任何
  实现了同签名 ``generate`` 方法的类都自动满足该协议，无需显式继承——这让新增 provider
  非常灵活，同时仍可被类型检查器校验。
- ``StaticProvider``：返回预设输出的确定性 provider，专供本地调试和按 seed 复现使用。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


# cache: 05K08 04728
@dataclass
class ModelOutput:
    """单次模型生成调用的统一输出结构。

    所有字段除 ``raw_text`` 外都带默认值，便于不同 provider 按各自能力填充其能提供的
    元数据子集；上层据此实现"对 provider 无感"。

    字段含义与"为什么需要"：

    - ``raw_text``：模型生成的原始文本（未做工具调用解析的纯文本部分）。是最基本、
      所有 provider 都必须给出的内容。
    - ``tool_calls``：归一化后的工具调用列表（见 tool_calling.py）。各家原始格式不同，
      这里统一成 ``{id, name, arguments, native}`` 的形态，供 agent 执行工具。
    - ``assistant_message``：本轮 assistant 的"原样"消息体。回写进对话历史时需要它，
      以保证下一轮请求的消息格式与该 provider 的 API 期望一致（尤其是带 tool_calls 的
      助手消息，必须原样回填才能与后续 tool 结果配对）。
    - ``token_usage``：本次调用的 token 计量（prompt/completion 等）。训练与成本核算、
      奖励/惩罚长度都依赖它，因此是关键审计字段。
    - ``model_name``：调用方配置的逻辑模型名。
    - ``provider``：产出该结果的 provider 标识（openai / anthropic / vllm / local_hf /
      static）。用于追溯结果来源、区分不同后端的行为差异。
    - ``sampling_config``：本次实际使用的采样参数（temperature/top_p 等）。复现实验、
      对齐训练与推理设置时必须留痕。
    - ``tokenizer_version``：分词器版本/类名。训练（尤其本地 HF）对分词一致性极其敏感，
      记录它可在数据与训练阶段对齐分词，避免 token 错位。
    - ``served_model_name``：服务端实际返回的模型名（API 真正服务的权重/部署名）。它可能
      与配置的 ``model_name`` 不同（如别名、灰度、vLLM 实际加载路径），需要单独记录以便审计
      "我以为调的是 A，实际服务的是 B"这类问题。
    """

    raw_text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    assistant_message: dict[str, Any] | None = None
    token_usage: dict[str, Any] = field(default_factory=dict)
    model_name: str = ""
    provider: str = ""
    sampling_config: dict[str, Any] = field(default_factory=dict)
    tokenizer_version: str | None = None
    served_model_name: str | None = None

    def metadata(self) -> dict[str, Any]:
        """汇总可用于审计/复现的元数据（不含 raw_text、tool_calls 等内容主体）。

        把"调用上下文"与"生成内容"分离，方便上层只把这一份轻量元数据落盘或入库，
        用于追踪是哪一个模型、用什么参数、在哪个服务上产出的结果。
        """
        return {
            "model_name": self.model_name,
            "provider": self.provider,
            "sampling_config": self.sampling_config,
            "tokenizer_version": self.tokenizer_version,
            "served_model_name": self.served_model_name,
            "token_usage": self.token_usage,
        }


class ModelProvider(Protocol):
    """所有模型 provider 必须满足的结构化协议（structural typing）。

    使用 ``Protocol`` 而非抽象基类：任何对象只要拥有同签名的 ``generate``，即被视为
    一个合法 provider，无需显式继承。这降低了新增后端的耦合，也方便测试替身。

    ``generate`` 的参数约定：

    - ``messages_or_prompt``：既可是一段裸字符串 prompt，也可是 OpenAI 风格的 messages
      列表。统一接收两种形态，由各 provider 内部归一，调用方无需关心格式细节。
    - ``sampling_config``：采样参数字典（temperature/top_p/max_tokens 等）。
    - ``tools``：OpenAI 风格的工具 schema 列表（可为空）。
    """

    def generate(
        self,
        messages_or_prompt: Any,
        sampling_config: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelOutput:
        ...


class StaticProvider:
    """确定性 provider：按预设列表逐条返回输出，便于本地调试和按 seed 复现。

    不访问任何外部模型，每次调用返回 ``outputs`` 中的下一条；用完后返回空串。常用于
    单测、流水线烟雾测试，以及在不消耗真实推理资源的情况下复现一段固定轨迹。
    """

    def __init__(self, outputs: list[str], model_name: str = "static"):
        # 拷贝一份预设输出，避免外部后续修改影响内部状态
        self.outputs = list(outputs)
        self.model_name = model_name
        # 游标：记录下一次该返回第几条输出
        self.index = 0

    def generate(
        self,
        messages_or_prompt: Any,
        sampling_config: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelOutput:
        # 预设输出耗尽后退化为空串，保证调用永不抛错（确定性、可重复）
        if self.index >= len(self.outputs):
            text = ""
        else:
            text = self.outputs[self.index]
        # 前移游标，使下一次调用返回后续输出
        self.index += 1
        return ModelOutput(
            raw_text=text,
            # 同步给出一个标准 assistant 消息体，便于直接回填对话历史
            assistant_message={"role": "assistant", "content": text},
            model_name=self.model_name,
            provider="static",
            sampling_config=sampling_config or {},
        )
