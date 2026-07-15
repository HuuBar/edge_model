"""工具 schema 转换与原生工具调用归一化的公共工具集。

本模块解决三件事：

1. 工具名安全化（safe/restore）：本项目工具名常带点号（如 ``fs.read``、``shell.exec``），
   但多数模型 API 对函数名有字符限制（一般只允许字母、数字、下划线、连字符）。因此下发前
   把非法字符替换成 ``__``，并把"原始名"写进描述里；同时维护 安全名->原始名 的反查表，
   等模型回传工具调用时再还原成原始名，使上层始终用统一的真实工具名执行。

2. 原生工具调用归一化（normalize_*）：OpenAI 用 ``message.tool_calls``（arguments 为
   JSON 字符串），Anthropic 用 content 里的 ``tool_use`` block（input 为对象）。两者结构
   不同，这里统一收敛成 ``{id, name, arguments, native}`` 形态，屏蔽差异。

3. 文本工具菜单（render_text_tool_menu）：面向不支持/未开启原生工具调用的模型（如本地
   小模型），把工具清单渲染成中文文本说明，约定用一个或多个 ``<tool_call>`` JSON 块来表达调用。
"""

from __future__ import annotations

import copy
import json
import re
from typing import Any

# 安全化时用于替换非法字符的分隔串
SAFE_SEP = "__"


def safe_tool_name(name: str) -> str:
    """把工具名中不被 API 接受的字符替换为 ``__``。

    允许保留的字符集为字母/数字/下划线/连字符；其余（如点号 ``.``）全部替换，
    以满足各家函数调用接口对名称的字符约束。
    """
    return re.sub(r"[^a-zA-Z0-9_-]", SAFE_SEP, name)


def restore_tool_name(name: str, reverse_map: dict[str, str]) -> str:
    """用反查表把模型回传的安全名还原为原始工具名；表中无则原样返回。"""
    return reverse_map.get(name, name)


def prepare_openai_tools(tools: list[dict[str, Any]] | None) -> tuple[list[dict[str, Any]] | None, dict[str, str]]:
    """把 OpenAI 风格工具 schema 做安全化处理，并返回反查表。

    保持 OpenAI 的 ``{type, function:{name, description, parameters}}`` 结构不变，仅替换
    其中的 ``name`` 为安全名，同时把原始名追加进 description（便于模型理解真实工具语义，
    也方便人工排查）。返回 (转换后工具列表, 安全名->原始名 反查表)。
    """
    if not tools:
        return None, {}
    converted = []
    reverse_map = {}
    for tool in tools:
        # 深拷贝，避免污染调用方原始 schema
        item = copy.deepcopy(tool)
        fn = item.get("function", {})
        original = fn.get("name")
        if original:
            safe = safe_tool_name(original)
            # 记录映射，供回程还原
            reverse_map[safe] = original
            fn["name"] = safe
            description = fn.get("description", "")
            # 把原始名写进描述，弥补安全化后名称失真带来的语义损失
            fn["description"] = f"{description} Internal tool name: {original}".strip()
        converted.append(item)
    return converted, reverse_map


def prepare_anthropic_tools(tools: list[dict[str, Any]] | None) -> tuple[list[dict[str, Any]] | None, dict[str, str]]:
    """把 OpenAI 风格工具 schema 转换为 Anthropic 的工具结构并安全化。

    Anthropic 的工具 schema 是扁平的 ``{name, description, input_schema}``（注意参数键名为
    ``input_schema`` 而非 ``parameters``），故这里重建结构而非原样改名。同样把原始名写入
    描述，并返回反查表用于回程还原。
    """
    if not tools:
        return None, {}
    converted = []
    reverse_map = {}
    for tool in tools:
        fn = tool.get("function", {})
        original = fn.get("name")
        if not original:
            # 没有名字的工具无法调用，跳过
            continue
        safe = safe_tool_name(original)
        reverse_map[safe] = original
        description = fn.get("description", "")
        converted.append(
            {
                "name": safe,
                "description": f"{description} Internal tool name: {original}".strip(),
                # OpenAI 的 parameters 对应 Anthropic 的 input_schema；缺省给一个空对象 schema
                "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
            }
        )
    return converted, reverse_map


def normalize_openai_tool_calls(message: dict[str, Any], reverse_map: dict[str, str]) -> list[dict[str, Any]]:
    """把 OpenAI 的 ``message.tool_calls`` 归一化为统一结构。

    统一形态：``{id, name(已还原原始名), arguments(dict), native(原始调用对象)}``。
    保留 ``native`` 是为了回填对话历史时能原样回传该 provider 期望的格式。
    """
    normalized = []
    for call in message.get("tool_calls") or []:
        function = call.get("function", {})
        # OpenAI 的 arguments 是 JSON 字符串；缺省给空对象字符串
        raw_args = function.get("arguments") or "{}"
        if isinstance(raw_args, str):
            try:
                arguments = json.loads(raw_args)
            except json.JSONDecodeError:
                # 模型可能产出非法 JSON，容错为空参数而非抛错中断
                arguments = {}
        else:
            # 已是对象则直接用；其它类型兜底为空
            arguments = raw_args if isinstance(raw_args, dict) else {}
        normalized.append(
            {
                "id": call.get("id"),
                # 还原成上层使用的真实工具名（如 fs.read）
                "name": restore_tool_name(function.get("name", ""), reverse_map),
                "arguments": arguments,
                "native": call,
            }
        )
    return normalized


def normalize_anthropic_tool_calls(content: list[dict[str, Any]], reverse_map: dict[str, str]) -> list[dict[str, Any]]:
    """从 Anthropic 的 content block 列表中抽取 ``tool_use`` 并归一化。

    与 OpenAI 版产出同一结构；差别在于 Anthropic 的参数在 ``input`` 字段且本身就是对象，
    无需 JSON 解析。
    """
    normalized = []
    for block in content:
        # 只关心工具调用块，跳过 text 等其它类型
        if block.get("type") != "tool_use":
            continue
        normalized.append(
            {
                "id": block.get("id"),
                "name": restore_tool_name(block.get("name", ""), reverse_map),
                # input 已是对象；非对象时兜底为空
                "arguments": block.get("input") if isinstance(block.get("input"), dict) else {},
                "native": block,
            }
        )
    return normalized


def render_text_tool_menu(tools: list[dict[str, Any]] | None) -> str:
    """把工具清单渲染成中文文本菜单（用于非原生工具调用路径）。

    适用于不支持/未开启原生函数调用的模型：用提示词约定输出格式，要求模型输出一个或多个
    ``<tool_call>{"name":...,"arguments":{}}</tool_call>`` JSON 块，并显式禁止把工具调用
    写成解释文字或输出隐藏推理（``<think>``）。每个工具列出名称、描述，以及参数的类型与
    必填/可选标记，帮助模型正确构造调用。
    """
    if not tools:
        return ""
    lines = [
        "可用工具如下。需要调用工具时，输出一个或多个 <tool_call> JSON 块；不要把工具调用写成解释文字。",
        "不要输出 <think>...</think> 或任何隐藏推理。",
        "每个工具调用的格式：",
        '<tool_call>{"name":"tool.name","arguments":{}}</tool_call>',
        "工具清单：",
    ]
    for tool in tools:
        fn = tool.get("function", {})
        parameters = fn.get("parameters", {})
        # required 列表用于区分参数是必填还是可选
        required = parameters.get("required", [])
        properties = parameters.get("properties", {})
        # 此处直接用原始工具名（文本路径无需安全化，模型按字面名调用）
        lines.append(f"- {fn.get('name')}: {fn.get('description', '')}")
        if properties:
            args = []
            for name, spec in properties.items():
                marker = "required" if name in required else "optional"
                # 渲染为 名称(类型, required/optional)，类型缺省为 any
                args.append(f"{name}({spec.get('type', 'any')}, {marker})")
            lines.append(f"  args: {', '.join(args)}")
    return "\n".join(lines)
