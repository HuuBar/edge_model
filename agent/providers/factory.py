from __future__ import annotations

# 版权所有 © 2026 深圳途明智启科技有限公司。保留所有权利。
# 未经书面许可，任何单位或个人不得复制、传播、发布、转卖、改编、仿制或用于商业用途。
# 侵权必究。

"""Provider 工厂：由环境变量（及可选 .env 文件）驱动选择并构造模型 provider。

设计意图：把"用哪家模型后端"这一决策完全外置到环境变量，使同一份代码无需改动即可在
本地（vLLM / 本地 HF）、云端（OpenAI / Anthropic）等不同部署间切换，便于训练、评测、
线上推理共用同一套 agent 逻辑。
"""

from __future__ import annotations

import os
from pathlib import Path

from agent.providers.api_provider import APIProvider
from agent.providers.local_hf_provider import LocalHFProvider
from agent.providers.vllm_provider import VLLMProvider


def load_dotenv(path: str | Path = ".env") -> None:
    """极简版 .env 加载器：把 .env 中的键值注入到 os.environ（不覆盖已有值）。

    避免引入第三方依赖，自己解析。规则：
    - 文件不存在则直接返回，静默不报错（.env 是可选的）。
    - 跳过空行、注释行（# 开头）、以及不含 ``=`` 的行。
    - 用 ``setdefault`` 而非赋值：已存在的环境变量优先，确保真实环境变量可覆盖 .env，
      符合"显式环境变量 > .env 默认值"的惯例。
    """
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        # 仅按第一个 = 切分，使值中可包含 =（如 base64、URL 参数）
        key, value = line.split("=", 1)
        # 去掉值两端的引号，避免把字面引号当成配置内容
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def provider_from_env(provider: str | None = None):
    """根据 ``MODEL_PROVIDER`` 环境变量（或显式入参）构造对应的 provider 实例。

    选择优先级：显式入参 ``provider`` > 环境变量 ``MODEL_PROVIDER`` > 默认 ``"vllm"``。
    默认走 vllm，因为这是本项目本地自托管训练/推理的主路径。

    每个分支只读取该后端所需的环境变量，并对常用项给出默认值；其中 API key 类用 ``[]``
    下标读取（缺失即报 KeyError），因为没有密钥根本无法调用，应尽早失败。
    """
    # 先加载 .env，确保后续读取环境变量时默认值已就位
    load_dotenv()
    # 统一转小写，容忍大小写书写差异（如 "OpenAI"、"VLLM"）
    selected = (provider or os.environ.get("MODEL_PROVIDER") or "vllm").lower()
    if selected == "local_hf":
        # 本地 transformers 直跑：model_path 指向本地权重目录
        return LocalHFProvider(
            model_path=os.environ.get("LOCAL_HF_MODEL_PATH", "models/original_model/Qwen3-8B"),
            model_name=os.environ.get("LOCAL_HF_MODEL_NAME"),
        )
    if selected == "deepseek":
        # DeepSeek OpenAI 兼容端点（base_url 末尾不带 /v1，APIProvider 自动拼 /chat/completions）。
        # deepseek-v4-flash 既可当采样模型（支持 function calling）也可当 probe judge。
        return APIProvider(
            base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            api_key=os.environ["DEEPSEEK_API_KEY"],
            model=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"),
            provider_name="deepseek",
            sanitize_tool_names=True,  # 采样调工具时工具名含点号需安全化；judge 无工具时无副作用
        )
    if selected == "qwen":
        # 通义千问（DashScope）OpenAI 兼容端点。probe 期 verifier judge 默认走这个。
        return APIProvider(
            base_url=os.environ.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            api_key=os.environ["QWEN_API_KEY"],
            model=os.environ.get("QWEN_MODEL", "qwen3.7-max"),
            provider_name="qwen",
            sanitize_tool_names=False,  # judge 不调工具，无需安全化工具名
        )
    if selected == "vllm":
        # 本地/自托管 vLLM 的 OpenAI 兼容端点（默认本项目主路径）
        return VLLMProvider(
            base_url=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
            model=os.environ.get("VLLM_MODEL", "models/original_model/Qwen3-8B"),
            # 把字符串环境变量解析成布尔：是否使用 vLLM 原生工具调用，
            # 关闭时改用文本工具菜单注入（见 vllm_provider.py）
            native_tool_calling=os.environ.get("VLLM_NATIVE_TOOL_CALLING", "false").lower()
            in {"1", "true", "yes"},
        )
    # 未知取值直接报错，避免静默回退到非预期后端
    raise ValueError(
        f"unsupported MODEL_PROVIDER: {selected}; "
        "training release supports vllm/local_hf/deepseek/qwen"
    )


def verifier_provider_from_env():
    """构造 probe 期 verifier 的 LLM judge provider（与采样模型解耦）。

    由 ``VERIFIER_PROVIDER`` 选择（默认 ``qwen`` = qwen3.7-max）；可设为 ``none`` 表示
    用 verifier 内置启发式回退（不调 LLM）。其它取值复用 provider_from_env 的分支。
    """
    load_dotenv()
    selected = (os.environ.get("VERIFIER_PROVIDER") or "qwen").lower()
    if selected in {"none", "heuristic", ""}:
        return None
    return provider_from_env(selected)
