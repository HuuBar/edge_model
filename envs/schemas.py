# 版权所有 © 2026 深圳途明智启科技有限公司。保留所有权利。
# 未经书面许可，任何单位或个人不得复制、传播、发布、转卖、改编、仿制或用于商业用途。
# 侵权必究。

"""工具元数据与执行结果的数据结构（schema）定义。

本模块定义三类基础结构：
- ToolExecutionError：结构化的工具错误（带 source 区分 LLM 错 vs 环境注入错）。
- ToolArg：单个工具参数的元信息（类型/是否必填/描述）。
- ToolDefinition：一个工具的完整定义（名字/描述/权限/参数/handler 等）。
这些结构既被 ToolFactory 用于注册和执行工具，也被用于向 LLM 暴露 OpenAI 风格的 tool schema。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


class ToolExecutionError(Exception):
    """结构化的工具执行错误。

    相比裸 Exception，它携带三个关键字段，便于 verifier 判分时区分错误来源：
    - code：机器可读的错误码（如 "tool_not_allowed" / "order_id_required"）。
    - message：人类可读消息（缺省与 code 相同）。
    - source：错误归因，决定这次报错算谁的责任——
        "llm"（默认）= 模型自己用错了工具（缺参数、调了不该调的工具），会计入 efficiency 扣分；
        "runtime" = 框架/handler 运行时异常（防御性兜底）；
        "environment" = 环境主动注入的故障（见 toolfactory._maybe_fault），不应归咎于模型。
    """

    def __init__(self, code: str, message: str | None = None, *, source: str = "llm"):
        super().__init__(message or code)
        self.code = code
        self.message = message or code
        self.source = source

    def to_observation(self) -> dict[str, Any]:
        """转成回传给 LLM/写入 trajectory 的 observation 字典。

        ok=False 表示这次工具调用失败；error/message/source 保留错误码、消息与归因，
        供下游（尤其 verifier 的 efficiency 子分）判断该错误是否要算到模型头上。
        """
        return {"ok": False, "error": self.code, "message": self.message, "source": self.source}


@dataclass(frozen=True)
class ToolArg:
    """单个工具参数的元信息（不可变）。

    type：JSON schema 类型字符串（"string" / "number" / "boolean" / "object" ...）。
    required：是否必填——validate_args 会据此在缺参时抛 "<参数名>_required" 错误。
    description：参数说明，会原样写进给 LLM 的 tool schema。
    用 frozen=True 冻结，保证工具定义在注册后不被意外修改。
    """

    type: str
    required: bool = False
    description: str = ""


@dataclass(frozen=True)
class ToolDefinition:
    """一个工具的完整定义（不可变）。

    name：工具名（含点号命名空间，如 "finance.issue_refund"），全局唯一键。
    description：给 LLM 看的工具用途说明。
    permissions：权限标签元组；其中 "sandbox_write" 标记该工具会写 sandbox 台账（见 is_write）。
    args：参数名 -> ToolArg 的映射。
    output_fields：输出字段的说明（文档用途）。
    failure_modes：该工具可能的失败模式（文档/测试用途）。
    verifier_facts：该工具能为 verifier 提供哪些事实键（如 refund_issued），供判分对齐。
    handler：实际执行逻辑的回调；为 None 表示尚未实现（execute 时会报 tool_not_implemented）。
    """

    name: str
    description: str
    permissions: tuple[str, ...]
    args: dict[str, ToolArg]
    output_fields: dict[str, str] = field(default_factory=dict)
    failure_modes: tuple[str, ...] = ()
    verifier_facts: tuple[str, ...] = ()
    handler: Callable[..., dict[str, Any]] | None = None

    @property
    def is_write(self) -> bool:
        """是否为写工具：权限里含 "sandbox_write" 即会改 sandbox 台账（如发起退款、关单）。

        读工具（查订单、查物流）不带此权限，没有副作用。
        write_tool_menu / verifier 据此区分两类工具。
        """
        return "sandbox_write" in self.permissions

    def validate_args(self, provided: dict[str, Any]) -> None:
        """校验参数：① 必填齐全；② **拒绝 schema 外的未知参数**（schema 即真契约）。

        缺必填 → "<name>_required"；传了未声明参数 → "<name>_unknown_arg"。两类 source 默认 "llm"
        （模型漏传/乱传，计入 efficiency）。严格拒未知参数可防"假边"——例如给只吃 attachment_id 的
        attachment.inspect 偷传 order_id 假装按订单范围核验。
        """
        missing = [key for key, spec in self.args.items() if spec.required and key not in provided]
        if missing:
            raise ToolExecutionError(f"{missing[0]}_required")
        unknown = sorted(key for key in provided if key not in self.args)
        if unknown:
            raise ToolExecutionError(f"{unknown[0]}_unknown_arg")

    def to_tool_schema(self) -> dict[str, Any]:
        """转换成 OpenAI function-calling 风格的 tool schema，供下发给 LLM。

        把 args 拆成 JSON schema 的 properties（带类型与描述），并把 required=True 的参数名
        收集进 required 列表。最终结构即各家 LLM API 通用的 {"type":"function","function":{...}}。
        """
        properties = {}
        required = []
        for name, spec in self.args.items():
            properties[name] = {"type": spec.type, "description": spec.description}
            if spec.required:
                required.append(name)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }
