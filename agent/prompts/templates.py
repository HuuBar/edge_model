# 版权所有 © 2026 深圳途明智启科技有限公司。保留所有权利。
# 未经书面许可，任何单位或个人不得复制、传播、发布、转卖、改编、仿制或用于商业用途。
# 侵权必究。

"""Prompt 模板加载与哈希。

本模块负责两件事：
1. 从 ``agent/prompts/`` 目录加载 ``.txt`` 模板并用 Jinja 渲染。
2. 为 prompt 和 tool schema 计算稳定哈希（prompt_hash / tool_schema_hash）。

为什么需要哈希：每条 trajectory 都要记录 prompt_hash / tool_schema_hash /
prompt_template_version。GRPO group 要求同一组内 prompt 版本、
tool_schema_hash、env/verifier/model 版本全部一致，consistency audit 靠这些哈希
对齐 rollout 侧与 training 侧，确保训练时还原的 prompt 与生成时完全相同。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from jinja2 import Template

# prompt 模板版本号：随模板正文/结构变更而 bump，进入每条 trajectory 与 manifest，
# 是 GRPO group 同质性校验的关键字段之一（同组必须同 prompt_template_version）。
PROMPT_TEMPLATE_VERSION = "wifi_prompt_v1"
# prompt 目录固定为本文件所在目录（agent/prompts/），所有 .txt 模板都在这里。
PROMPT_DIR = Path(__file__).resolve().parent


def load_prompt(name: str) -> str:
    """按文件名读取 prompt 模板原文（UTF-8）。

    显式指定 ``encoding="utf-8"``，因为模板含中文，不能依赖平台默认编码，
    否则在不同操作系统/locale 下读出的字节不同会导致 prompt_hash 漂移。
    """
    return (PROMPT_DIR / name).read_text(encoding="utf-8")


def render_prompt(name: str, context: Mapping[str, Any]) -> str:
    """加载并用给定上下文渲染一个 Jinja 模板，返回最终 prompt 文本。

    Jinja 只做变量替换/循环（如把 case.entities 展开），不承载业务逻辑；
    runtime 渲染好的纯文本会直接放进 messages。
    """
    return Template(load_prompt(name)).render(**context)


def stable_hash(value: Any) -> str:
    """对任意值计算稳定的 SHA-256 哈希（返回十六进制字符串）。

    “稳定”是关键：
    - 字符串直接哈希其内容。
    - 非字符串（如 tool_schemas 列表/字典）先用 ``json.dumps`` 序列化，并强制
      ``sort_keys=True`` 让键顺序无关、``ensure_ascii=False`` 让中文按原文参与哈希、
      ``default=str`` 兜底处理不可直接序列化的对象。这样同一份语义内容无论字段
      书写顺序如何，都得到相同哈希——这正是 consistency audit 能跨进程比对的前提。
    """
    if isinstance(value, str):
        payload = value
    else:
        payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def prompt_hash(*texts: str) -> str:
    """把多段 prompt 文本（如 system + step_user）拼接后求稳定哈希。

    用固定分隔符 ``"\\n\\n"`` 连接各段，保证「同样的几段文本」总产生同一个
    prompt_hash；runtime 即以此作为整条 trajectory 的 prompt 指纹。
    """
    return stable_hash("\n\n".join(texts))
