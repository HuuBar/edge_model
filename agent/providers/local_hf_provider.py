from __future__ import annotations

# 版权所有 © 2026 深圳途明智启科技有限公司。保留所有权利。
# 未经书面许可，任何单位或个人不得复制、传播、发布、转卖、改编、仿制或用于商业用途。
# 侵权必究。

"""本地 Hugging Face provider：用 transformers 直接在本机加载权重做推理（延迟导入）。

特意把 transformers 的 import 推迟到真正用到时（``_load``），使得在未选用该 provider
的环境（如只跑 API provider 的轻量进程）无需安装重型依赖即可导入本包。
"""

from __future__ import annotations

import json
from typing import Any

from agent.providers.base import ModelOutput


class LocalHFProvider:
    """本地 transformers 推理 provider；模型/分词器按需懒加载并缓存。"""

    def __init__(self, model_path: str, model_name: str | None = None, device_map: str = "auto"):
        self.model_path = model_path
        # 逻辑模型名缺省回退为权重路径，保证 ModelOutput.model_name 非空
        self.model_name = model_name or model_path
        # 设备映射策略，"auto" 交由 accelerate 自动分配（多卡/CPU 兜底）
        self.device_map = device_map
        # 延迟加载：首次 generate 时才真正实例化，避免构造即吃显存
        self._tokenizer = None
        self._model = None

    def _load(self) -> None:
        """惰性加载分词器与模型；已加载则直接返回（幂等）。"""
        if self._model is not None:
            return
        # 延迟导入 transformers：仅在确实使用本地推理时才付出依赖/加载成本
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # trust_remote_code：允许加载仓库自带的自定义建模代码（如 Qwen 等自定义架构）
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map=self.device_map,
            trust_remote_code=True,
        )

    def generate(
        self,
        messages_or_prompt: Any,
        sampling_config: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelOutput:
        """本地生成：构建 prompt -> 编码 -> generate -> 仅解码新增 token。"""
        # 确保模型/分词器已加载（首次调用触发实际加载）
        self._load()
        assert self._tokenizer is not None
        assert self._model is not None
        config = sampling_config or {}
        if isinstance(messages_or_prompt, list) and hasattr(self._tokenizer, "apply_chat_template"):
            # 优先用分词器自带的 chat template，把 messages（含 tools）渲染成模型期望的
            # 对话格式；add_generation_prompt 追加助手起始标记以引导续写
            prompt = self._tokenizer.apply_chat_template(
                normalize_chat_template_messages(messages_or_prompt),
                tools=tools,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            # 裸字符串或无 chat template：直接当作纯文本 prompt
            prompt = str(messages_or_prompt)
        # 编码并搬到模型所在设备
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        # 生成前清理缓存，避免NPU显存碎片导致OOM（尤其对4B/8B大模型）
        if hasattr(self._model.device, 'type') and self._model.device.type == 'npu':
            import torch
            torch.npu.empty_cache()
            torch.npu.synchronize()
        # 构建generate参数，确保与模型config不冲突
        gen_kwargs = {
            "max_new_tokens": int(config.get("max_new_tokens", 512)),
        }
        # 只有当do_sample=True时才传temperature/top_p，避免与模型默认config冲突
        do_sample = bool(config.get("do_sample", True))
        gen_kwargs["do_sample"] = do_sample
        if do_sample:
            gen_kwargs["temperature"] = float(config.get("temperature", 0.7))
            gen_kwargs["top_p"] = float(config.get("top_p", 0.9))
            # 如果模型config有默认top_k，可能会被使用；我们不覆盖
        output_ids = self._model.generate(
            **inputs,
            **gen_kwargs,
        )
        # 切掉输入部分，仅保留新生成的 token（generate 返回的是 prompt+续写拼接）
        generated = output_ids[0][inputs["input_ids"].shape[-1] :]
        # skip_special_tokens=False：保留特殊标记，便于上层解析工具调用等结构标记
        text = self._tokenizer.decode(generated, skip_special_tokens=False)
        return ModelOutput(
            raw_text=text,
            token_usage={
                # 本地推理无服务端 usage，自行用张量长度统计 token 数
                "prompt_tokens": int(inputs["input_ids"].shape[-1]),
                "completion_tokens": int(generated.shape[-1]),
            },
            model_name=self.model_name,
            provider="local_hf",
            sampling_config=config,
            # 记录分词器类名作为版本标识：训练/推理需对齐分词以避免 token 错位
            tokenizer_version=self._tokenizer.__class__.__name__,
            # 本地以权重路径作为"实际服务模型"标识
            served_model_name=self.model_path,
        )


def normalize_chat_template_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把 messages 规整成 Hugging Face chat_template 能稳定渲染的形态。

    Qwen3 chat_template 只会渲染字符串 content；tool observation 在 runtime 里保留为
    dict 以便审计，但直接喂给 tokenizer 会变成空 ``<tool_response>``。这里在 provider
    边界做 JSON 序列化，既不污染 trajectory，也保证本地 Qwen 能看到工具结果。
    """

    normalized = []
    for message in messages:
        item = dict(message)
        content = item.get("content")
        if not isinstance(content, (str, type(None))):
            item["content"] = json.dumps(content, ensure_ascii=False)
        normalized.append(item)
    return normalized
