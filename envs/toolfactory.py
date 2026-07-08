# 版权所有 © 2026 深圳途明智启科技有限公司。保留所有权利。
# 未经书面许可，任何单位或个人不得复制、传播、发布、转卖、改编、仿制或用于商业用途。
# 侵权必究。

"""工具工厂：工具注册表、schema 暴露与统一执行包装。

本模块是工具层的入口。职责有三：
1. 动态加载（import）TOOL_MODULES 列出的全部工具实现模块，组成 name -> ToolDefinition 注册表。
2. 向 LLM 暴露生产 tool schema；调试/authoring 可显式传 allowed_tools 做临时过滤。
3. 统一执行工具：权限校验 -> 参数校验 -> 故障注入 -> 调 handler -> 包装成标准 observation。

设计要点：故障注入(_maybe_fault)把"环境主动制造的报错"标记为 source="environment"，
与模型自己用错工具的 source="llm" 报错严格区分，避免环境注入的故障被算到模型的 efficiency 扣分里
。
"""

from __future__ import annotations

import importlib
from collections import defaultdict
from typing import Any

from envs.sandbox_state import SandboxState
from envs.schemas import ToolDefinition, ToolExecutionError

# 生产工具实现模块的导入路径列表。
# 每个模块内必须暴露一个名为 TOOL 的 ToolDefinition（见 _load_default_tools）。
# 这里登记客服域线上可用的工具：读工具（crm/oms/tms/policy/warranty 等查询）与写工具
# （finance.issue_refund、returns.create_label、ticket.close 等会落 sandbox 台账的动作）。
# reserved 或缺少支撑数据/发现入口的工具不进入默认生产注册表。
# 用"模块路径清单 + 动态 import"而非硬编码 import，是为了新增工具只需加一行路径、
# 注册逻辑零改动；模块与工具一一对应。
TOOL_MODULES = [
    "envs.toollist.crm_get_customer",
    "envs.toollist.oms_list_orders",
    "envs.toollist.oms_get_order",
    "envs.toollist.tms_get_tracking",
    "envs.toollist.attachment_list",
    "envs.toollist.attachment_inspect",
    "envs.toollist.policy_search",
    "envs.toollist.finance_simulate_refund",
    "envs.toollist.finance_get_refund_status",
    "envs.toollist.payment_get_charge",
    "envs.toollist.wms_get_fulfillment",
    "envs.toollist.invoice_get_invoice",
    "envs.toollist.returns_get_status",
    "envs.toollist.warranty_check",
    "envs.toollist.diagnostics_troubleshoot",
    "envs.toollist.subscription_get_status",
    "envs.toollist.oms_cancel_order",
    "envs.toollist.oms_modify_order",
    "envs.toollist.tms_intercept_shipment",
    "envs.toollist.tms_reroute_shipment",
    "envs.toollist.carrier_open_investigation",
    "envs.toollist.finance_issue_refund",
    "envs.toollist.payment_open_dispute_case",
    "envs.toollist.approval_create_case",  # escalate_approval 路径的写工具（taxonomy 已声明该决策形态，2026-06-20 un-defer）
    "envs.toollist.reshipment_create",
    "envs.toollist.invoice_update_vat",
    "envs.toollist.returns_create_label",
    "envs.toollist.subscription_cancel",
    "envs.toollist.message_reply",
    "envs.toollist.ticket_close",
    "envs.toollist.ticket_handoff",
]


class ToolFactory:
    """注册并执行课程环境工具。

    持有工具注册表（name -> ToolDefinition）和按命名空间统计的故障计数器。
    一个 ToolFactory 实例可服务多条并发 rollout：故障计数按 (namespace_id, tool_name)
    分桶，保证不同 rollout 的 transient_error "前 N 次失败"互不干扰。
    """

    def __init__(self, tools: dict[str, ToolDefinition] | None = None):
        # 默认从 TOOL_MODULES 动态加载全部工具；也可注入自定义注册表（便于测试）。
        self.tools = tools or self._load_default_tools()
        # 故障计数器：键为 (namespace_id, tool_name)，值为该 namespace 下该工具被触发故障的次数。
        # 用 per-namespace 计数而非全局计数，是因为多条 rollout 并发共用同一 factory，
        # transient_error 的"前 N 次失败"必须各 rollout 独立计，否则会互相吃掉失败配额。
        self._fault_counts: dict[tuple[str, str], int] = defaultdict(int)

    @staticmethod
    def _load_default_tools() -> dict[str, ToolDefinition]:
        """动态 import TOOL_MODULES 中的每个模块，收集其 TOOL 对象构成注册表。

        约定每个工具模块在模块级暴露一个名为 TOOL 的 ToolDefinition；
        以 tool.name 为键（而非模块名）入表，使工具名与调用方一致（如 "finance.issue_refund"）。
        动态加载让"增删工具 = 改 TOOL_MODULES 列表"，注册逻辑无需变更。
        """
        # registry 是运行时唯一的工具名索引。后续 get/tool_schemas/execute 都只查这张表，
        # 所以新增工具时必须保证 tool.name 与 prompt/schema 里的名字一致。
        registry: dict[str, ToolDefinition] = {}
        for module_name in TOOL_MODULES:
            # import_module 会执行工具模块顶层代码，从而拿到模块级 TOOL 定义。
            module = importlib.import_module(module_name)  # 按路径动态导入工具模块
            tool = getattr(module, "TOOL")  # 取模块约定暴露的 TOOL 定义
            # 若两个模块声明了同名 tool，后者会覆盖前者；当前项目通过 code review 保证不重复。
            registry[tool.name] = tool  # 以工具名为键登记
        return registry

    def get(self, name: str) -> ToolDefinition:
        """按名取工具定义；不存在则抛 unknown_tool（source 默认 llm）。"""
        if name not in self.tools:
            # 模型调用了注册表不存在的工具。这个错误属于模型动作错误，ToolExecutionError 默认 source=llm。
            raise ToolExecutionError("unknown_tool")
        return self.tools[name]

    def tool_registry_snapshot(self) -> list[dict[str, Any]]:
        """导出全部工具的元信息快照（按工具名排序）。

        用于落盘/调试/文档：每个工具给出 name、description、permissions（权限标签）
        以及 args 的 type/required/description。不含 handler 等不可序列化字段。
        """
        # 按工具名排序是为了让落盘快照稳定；否则 dict 插入顺序变化会影响 diff 和 hash 审计。
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "permissions": list(tool.permissions),
                "args": {
                    name: {
                        "type": spec.type,
                        "required": spec.required,
                        "description": spec.description,
                    }
                    for name, spec in tool.args.items()
                },
            }
            for tool in sorted(self.tools.values(), key=lambda item: item.name)
        ]

    def write_tool_menu(self) -> list[str]:
        """返回全部写工具（is_write=True）的名字列表（已排序）。

        即所有会落 sandbox 台账、产生真实业务副作用的工具，供 verifier/case 作者参考"哪些动作算 commit"。
        """
        # 只看 ToolDefinition.is_write，不根据权限字符串猜测，避免写工具分类口径漂移。
        return sorted(name for name, tool in self.tools.items() if tool.is_write)

    def tool_schemas(self, allowed_tools: list[str] | None = None) -> list[dict[str, Any]]:
        """生成下发给 LLM 的 tool schema 列表。

        线上 rollout 应传 None，暴露生产注册表全集。``allowed_tools`` 只保留给调试、
        单测或 authoring 辅助，不再表示 case 级权限；评分侧的 allowed_write_tools
        由 verifier 独立处理。
        """
        # allowed_tools 存在时只作为显式调试白名单；生产 rollout 传 None，暴露完整生产工具面。
        names = set(allowed_tools) if allowed_tools else set(self.tools)
        # 返回的是 OpenAI/Qwen 兼容的 function schema，不包含 handler、权限实现等 Python 对象。
        return [tool.to_tool_schema() for name, tool in self.tools.items() if name in names]

    def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        env_snapshot: dict[str, Any],
        sandbox: SandboxState,
        context: dict[str, Any],
        allowed_tools: list[str] | None = None,
    ) -> dict[str, Any]:
        """统一执行一次工具调用，返回标准化 observation。

        执行管线（任一步失败都会被下方 except 捕获并转成 ok=False 的 observation）：
          1. 取工具定义（不存在 -> unknown_tool）。
          2. 校验必填参数（缺参 -> <name>_required，source=llm）。
          3. 故障注入：若 env 为该工具配了 transient_error/hard_error 且命中，直接返回环境注入错误
             （source=environment，不算模型的错），不再调 handler。
          4. handler 未实现 -> tool_not_implemented（source=runtime）。
          5. 正常执行 handler，返回 ok=True + result。

        base 字段（tool_name/arguments/tool_call_id/namespace_id）无论成败都回带，
        便于把 observation 关联回具体调用与所属 rollout 命名空间。

        参数 context 携带运行期上下文（至少含 tool_call_id、namespace_id）；
        env_snapshot 为只读世界状态；sandbox 为本 rollout 的可写台账。
        """
        tool_call_id = context.get("tool_call_id")
        # base：成功/失败都会附带的回执元信息，用于把 observation 关联回调用与 rollout 命名空间。
        base = {
            "tool_name": tool_name,
            "arguments": arguments,
            "tool_call_id": tool_call_id,
            "namespace_id": context.get("namespace_id"),
        }
        try:
            # allowed_tools 是旧接口残留。runtime 不在这里做 case 级授权，避免“执行层截断轨迹”；
            # 是否允许该写动作由 verifier 看完整轨迹后评分。
            _ = allowed_tools  # compatibility only; verifier handles authorization, runtime no longer gates tools.
            tool = self.get(tool_name)  # 1) 取定义（未知工具会抛 unknown_tool）
            tool.validate_args(arguments)  # 2) 必填参数校验（缺参算模型的错）
            # 3) 故障注入：环境主动制造的报错优先于真正执行；命中则直接返回，不碰 handler。
            injected = self._maybe_fault(tool_name, env_snapshot, context)
            if injected is not None:
                return {**base, **injected}
            # 4) handler 缺失视为运行时问题（非模型责任）。
            if tool.handler is None:
                raise ToolExecutionError("tool_not_implemented", source="runtime")
            # 5) 正式执行：handler 拿到参数 + 只读 env + 可写 sandbox + 上下文，返回业务结果。
            result = tool.handler(arguments, env_snapshot, sandbox, context)
            return {**base, "ok": True, "result": result}
        except ToolExecutionError as exc:
            # 结构化错误：原样转成 observation（保留 code/message/source 供 verifier 归因）。
            return {**base, **exc.to_observation()}
        except Exception as exc:  # pragma: no cover - defensive wrapper for runtime stability
            # 兜底：任何未预期异常包成 runtime 错误，避免单次工具崩溃拖垮整条 rollout。
            wrapped = ToolExecutionError("tool_runtime_error", str(exc), source="runtime")
            return {**base, **wrapped.to_observation()}

    def _maybe_fault(
        self,
        tool_name: str,
        env_snapshot: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any] | None:
        """按 env_snapshot.tool_faults 决定是否给本次调用注入环境故障。

        返回 None = 不注入（放行去执行 handler）；返回 dict = 注入一个 source="environment" 的失败 observation。

        三种关心的故障模式：
          - none / latency：不注入错误。latency 在 当前版本 只是"慢但成功"，本函数不模拟延迟，故同 none 放行。
          - transient_error：前 fail_times 次失败、之后成功。用 per-namespace 计数实现"前 N 次"。
          - hard_error：每次都确定性失败。

        关键：注入的失败一律标 source="environment"。这是为了把"环境制造的报错"与
        "模型自己用错工具（source=llm）"区分开——verifier 计算 efficiency 时只惩罚后者，
        不能让模型为环境注入的瞬时/硬故障背锅，否则会错误地压低分数、训出错误信号。
        """
        # 取该工具的故障配置；缺省/为假值时视为无故障。
        fault = env_snapshot.get("tool_faults", {}).get(tool_name, {}) or {}
        mode = fault.get("mode", "none")
        # none 与 latency 都不产生错误（latency 仅"慢但成功"，当前版本 不真正 sleep），直接放行。
        if mode in {"none", "latency"}:
            return None
        # 计数键带 namespace_id：每条 rollout 独立计 transient 失败次数，互不串扰。
        key = (context.get("namespace_id", ""), tool_name)
        # 每次经过这里都先计数；transient_error 用这个计数决定是否仍处于失败窗口。
        self._fault_counts[key] += 1
        if mode == "transient_error":
            # 前 fail_times 次失败，之后（计数超过阈值）放行成功，模拟可重试的瞬时故障。
            fail_times = int(fault.get("fail_times", 1))
            if self._fault_counts[key] <= fail_times:
                return {
                    "ok": False,
                    "error": fault.get("error", "transient_error"),
                    "message": fault.get("error", "transient_error"),
                    "source": "environment",  # 环境注入，不计入模型 efficiency
                }
            return None
        if mode == "hard_error":
            # 确定性硬故障：每次调用都失败，无法靠重试绕过。
            return {
                "ok": False,
                "error": fault.get("error", "hard_error"),
                "message": fault.get("error", "hard_error"),
                "source": "environment",  # 同样是环境注入
            }
        # 未知/未处理的 mode：保守放行，不误伤。
        return None
