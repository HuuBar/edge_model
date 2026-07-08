# 版权所有 © 2026 深圳途明智启科技有限公司。保留所有权利。
# 未经书面许可，任何单位或个人不得复制、传播、发布、转卖、改编、仿制或用于商业用途。
# 侵权必究。

"""课程环境工具的共享实现。

三层调用结构（toolfactory → common → sandbox）：
- 上层 toolfactory：每个工具有一个薄包装文件（如 crm_get_customer.py、
  finance_issue_refund/tool.py），里面只有一句 ``TOOL = make_tool("xxx")``。
- 本层 common：``make_tool`` 工厂从 ``TOOL_SPECS`` 取出该工具的描述/权限/参数 schema，
  组装成一个 ``ToolDefinition``，并把 handler 统一指向 ``execute_named_tool``。
  ``execute_named_tool`` 按权限里有没有 ``sandbox_write`` 分流到 ``_read_handler``
  （读只读数据表）或 ``_write_handler``（落 sandbox 台账）。
- 底层 sandbox：真正的写入（append 台账 + 写审计日志）由
  ``envs.sandbox_state.SandboxState.write_record`` 完成。

读写工具因此都由同一个 ``make_tool`` 生成、共享同一套上下文与 schema 约定，
区别只在 handler 走读分支还是写分支。课程环境工具读取随包 JSON 数据，
写入 namespace 隔离的 sandbox JSON/JSONL，结果可 replay、可被 verifier 校验。
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from envs.sandbox_state import SandboxState
from envs.schemas import ToolArg, ToolDefinition, ToolExecutionError


def arg(type_: str, required: bool = False, description: str = "") -> ToolArg:
    """构造一个参数 schema 项（ToolArg），供 TOOL_SPECS 里声明每个 arg 的类型/是否必填。"""
    return ToolArg(type=type_, required=required, description=description)


def _tables(env_snapshot: dict[str, Any]) -> dict[str, Any]:
    """取整个只读 数据表集合（readonly_tables），所有读工具的数据来源。"""
    return env_snapshot.get("readonly_tables", {})


def _table(env_snapshot: dict[str, Any], name: str) -> dict[str, Any]:
    """按名字取单张只读表（如 orders、customers）；缺失或类型不对时回退成空 dict 而非报错。"""
    value = _tables(env_snapshot).get(name, {})
    return value if isinstance(value, dict) else {}


def _first_by(items: dict[str, Any], key: str, value: Any) -> dict[str, Any] | None:
    """在以 id 为键的表里按某字段值线性查第一条匹配行；用于只有 order_id 没有主键时回查。"""
    for row in items.values():
        if isinstance(row, dict) and row.get(key) == value:
            return row
    return None


def _copy(row: Any) -> Any:
    """深拷贝返回值，避免把只读课程数据 的内部对象直接暴露给调用方被改写。"""
    return deepcopy(row)


def _policy_rules(policy: dict[str, Any] | None) -> dict[str, Any]:
    """取政策的结构化规则块；写工具据此判断如退款是否允许、退货由谁付运费等。

    KB（policy_kb.py）以 `decision_rule` 为单一真源，故优先读它；`rules` 仅作旧字段回退。
    工具只 .get 自己需要的扁平键（return_shipping_paid_by / refund_allowed / duty_paid_by …），
    decision_rule 里的条件键（阈值/谓词）不影响这些读取。
    """
    if not policy:
        return {}
    rules = policy.get("decision_rule") or policy.get("rules", {})
    return rules if isinstance(rules, dict) else {}


def _find_policy(env_snapshot: dict[str, Any], policy_id: str | None) -> dict[str, Any] | None:
    """按 policy_id 精确取政策；写工具携带的 policy_id 在这里反查，用于读 rules 校验动作。"""
    if not policy_id:
        return None
    for policy in env_snapshot.get("policies", []):
        if policy.get("policy_id") == policy_id:
            return policy
    return None


def _policy_matches(policy: dict[str, Any], args: dict[str, Any]) -> bool:
    """policy.search 的匹配逻辑：按市场/topic/金额上限/订单状态逐项过滤，任一不符即排除。"""
    if args.get("market") and policy.get("market") != args.get("market"):
        return False
    topic = args.get("topic")
    if topic is not None and topic != policy.get("topic"):
        return False
    match = policy.get("match", {}) or {}
    amount = args.get("amount")
    if amount is not None and match.get("amount_max") is not None and amount > match["amount_max"]:
        return False
    if args.get("order_status") and match.get("order_status"):
        if args["order_status"] != match["order_status"]:
            return False
    return True


def _search_policy(env_snapshot: dict[str, Any], args: dict[str, Any]) -> dict[str, Any] | None:
    """按 args 条件返回第一条命中政策；命中后其 policy_id 会被写工具引用以证明动作合规。"""
    for policy in env_snapshot.get("policies", []):
        if _policy_matches(policy, args):
            return policy
    return None


def _record_context(context: dict[str, Any]) -> dict[str, str]:
    """校验并提取 runtime 注入的 namespace 隔离字段。

    这五个字段由 runtime 注入、不作为模型 tool arguments；所有 sandbox 写都必须带齐，
    用来把台账记录隔离到 namespace_id=run:case:rollout，缺任一个直接报错。
    """
    required = ["namespace_id", "run_id", "case_id", "rollout_id", "tool_call_id"]
    missing = [name for name in required if not context.get(name)]
    if missing:
        raise ToolExecutionError(f"{missing[0]}_required", source="runtime")
    return {name: str(context[name]) for name in required}


def _write(
    sandbox: SandboxState,
    tool_name: str,
    record: dict[str, Any],
    context: dict[str, Any],
    *,
    object_key: str | None = None,
    audit_action: str | None = None,
    audit_result: str | None = None,
) -> dict[str, Any]:
    """统一的写台账入口：所有写工具都经此把 record append 进对应 sandbox 台账并写审计日志。

    tool_name 决定写哪张台账；object_key 用于按对象去重/定位（如订单/订阅/ticket id）；
    audit_action/audit_result 落进 sandbox_audit_log 供留痕。底层落盘交给 SandboxState。
    """
    ctx = _record_context(context)
    return sandbox.write_record(
        tool_name,
        record,
        namespace_id=ctx["namespace_id"],
        run_id=ctx["run_id"],
        case_id=ctx["case_id"],
        rollout_id=ctx["rollout_id"],
        tool_call_id=ctx["tool_call_id"],
        object_key=object_key,
        audit_action=audit_action,
        audit_result=audit_result,
    )


def _id(prefix: str, context: dict[str, Any], *parts: Any) -> str:
    """生成确定性的对象 id（如 RF_<order>_<tool_call_id>）。

    带上 tool_call_id 后缀保证同一次调用幂等、可 replay，且不同调用不会撞 id。
    """
    clean = "_".join(str(p) for p in parts if p is not None)
    suffix = context.get("tool_call_id", "tc")
    return f"{prefix}_{clean}_{suffix}" if clean else f"{prefix}_{suffix}"


def _get_order(env_snapshot: dict[str, Any], order_id: str) -> dict[str, Any]:
    """读 orders 表取订单，查不到抛 order_not_found；读写工具都用它做订单前置校验。"""
    row = _table(env_snapshot, "orders").get(order_id)
    if not row:
        raise ToolExecutionError("order_not_found")
    return row


def _get_tracking(env_snapshot: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    """读 tracking 表取物流：优先用 tracking_id 主键，否则按 order_id 回查；查不到抛错。"""
    rows = _table(env_snapshot, "tracking")
    if args.get("tracking_id"):
        row = rows.get(args["tracking_id"])
    else:
        row = _first_by(rows, "order_id", args.get("order_id"))
    if not row:
        raise ToolExecutionError("tracking_not_found")
    return row


def _read_handler(tool_name: str, args: dict[str, Any], env: dict[str, Any], sandbox: SandboxState) -> dict[str, Any]:
    """读工具总分发：按 tool_name 命中分支，读对应只读表，不写任何台账、无副作用。

    finance.simulate_refund 是 read/dry-run：只返回 allowed 等结果供 agent 自己判断，不写 ledger、
    不拦后续——是否据此停手由 agent 决定、对错由 verifier 评分（训练期工具 permissive）。
    """
    if tool_name == "crm.get_customer":
        # 读 customers 表：客户画像（市场/等级/账户状态）。
        row = _table(env, "customers").get(args["customer_id"])
        if not row:
            raise ToolExecutionError("customer_not_found")
        return _copy(row)

    if tool_name == "memory.search":
        # 读 customer_memory 表：客户历史行为（退款/补发次数、风险记录）。
        row = _table(env, "customer_memory").get(args["customer_id"])
        if not row:
            raise ToolExecutionError("memory_snapshot_not_found")
        return {"customer_id": args["customer_id"], **_copy(row)}

    if tool_name == "oms.get_order":
        # 读 orders 表：订单详情，是大多数写动作的前置真相来源。
        return _copy(_get_order(env, args["order_id"]))

    if tool_name == "oms.list_orders":
        # 订单发现入口：只知道客户时，列出该客户订单候选，供模型定位 order_id。
        orders = []
        for order_id, row in _table(env, "orders").items():
            if isinstance(row, dict) and row.get("customer_id") == args["customer_id"]:
                item = _copy(row)
                item.setdefault("order_id", order_id)
                orders.append(item)
        orders.sort(key=lambda item: str(item.get("created_at") or item.get("order_date") or item.get("order_id")))
        return {"customer_id": args["customer_id"], "orders": orders}

    if tool_name == "tms.get_tracking":
        # 读 tracking 表：order_id 与 tracking_id 至少给一个，否则报错。
        if not args.get("order_id") and not args.get("tracking_id"):
            raise ToolExecutionError("order_id_or_tracking_id_required")
        return _copy(_get_tracking(env, args))

    if tool_name == "attachment.inspect":
        # 读 attachments 表：客户证据的检查结论（破损/错发/是否匹配订单）。
        row = _table(env, "attachments").get(args["attachment_id"])
        if not row:
            raise ToolExecutionError("attachment_not_found")
        return _copy(row)

    if tool_name == "attachment.list":
        # 证据发现入口：按订单列出附件 id，再由 attachment.inspect 做内容核验。
        attachments = []
        for attachment_id, row in _table(env, "attachments").items():
            if isinstance(row, dict) and row.get("order_id") == args["order_id"]:
                item = _copy(row)
                item.setdefault("attachment_id", attachment_id)
                attachments.append(item)
        attachments.sort(key=lambda item: str(item.get("uploaded_at") or item.get("attachment_id")))
        return {"order_id": args["order_id"], "attachments": attachments}

    if tool_name == "policy.search":
        # 读 policies 表：按条件检索政策，命中的 policy_id 后续写工具要带上以证明合规。
        if not args.get("topic"):
            raise ToolExecutionError("topic_required")
        policy = _search_policy(env, args)
        if not policy:
            raise ToolExecutionError("policy_not_found")
        return _copy(policy)

    if tool_name == "risk.check":
        # 读 risk_profiles 表：客户风险画像（high_risk cap 判定 当前版本 暂不接）。
        row = _table(env, "risk_profiles").get(args["customer_id"])
        if not row:
            raise ToolExecutionError("risk_profile_not_found")
        return _copy(row)

    if tool_name == "finance.simulate_refund":
        # 退款 dry-run（read，无副作用）：返回模拟结果供 agent 自己判断。
        # 工具 permissive：只拦物理现实（订单存在 / 不超付）；policy 缺失/无效不报错，
        # 只反映在结果里（allowed=false / policy_not_found），是否据此停手交给 agent，对错由 verifier 评分。
        order = _get_order(env, args["order_id"])  # 物理：订单必须存在
        if args["amount"] > order.get("paid_amount", 0):  # 物理：不能退超过实付
            raise ToolExecutionError("refund_amount_exceeds_paid_amount")
        policy = _find_policy(env, args.get("policy_id"))
        allowed = bool(_policy_rules(policy).get("refund_allowed", True)) if policy is not None else False
        return {
            "simulation_id": f"SIM_{args['order_id']}_{args['amount']}",
            "allowed": allowed,
            "policy_found": policy is not None,
            "amount": args["amount"],
            "currency": args["currency"],
            "requires_return": args["requires_return"],
            "reasons": [] if allowed else (["policy_denied"] if policy is not None else ["policy_not_found"]),
        }

    if tool_name == "finance.get_refund_status":
        # 查退款状态：合并读 refunds 只读表 + sandbox_refund_ledger 台账，
        # 这样本轮内 issue_refund 刚写入的退款也能查到（取最新一条作为 latest_status）。
        if not any(args.get(k) for k in ("order_id", "refund_id", "customer_id")):
            raise ToolExecutionError("lookup_key_required")
        refunds = []
        for row in _table(env, "refunds").values():
            if _refund_matches(row, args):
                refunds.append(_copy(row))
        for row in sandbox.state.get("sandbox_refund_ledger", []):
            if _refund_matches(row, args):
                refunds.append(_copy(row))
        if not refunds:
            raise ToolExecutionError("refund_not_found")
        latest = refunds[-1]
        return {
            "refunds": refunds,
            "latest_status": latest.get("status"),
            "eta_days": latest.get("eta_days"),
        }

    if tool_name == "payment.get_charge":
        # 读 charges 表：客户扣款记录，可按 order_id/charge_id 过滤，并带出重复扣款标记。
        row = _table(env, "charges").get(args["customer_id"])
        if not row:
            raise ToolExecutionError("charge_not_found")
        charges = row.get("charges", [])
        if args.get("order_id"):
            charges = [c for c in charges if c.get("order_id") == args["order_id"]]
        if args.get("charge_id"):
            charges = [c for c in charges if c.get("charge_id") == args["charge_id"]]
        return {"charges": _copy(charges), "duplicate_detected": row.get("duplicate_detected", False)}

    if tool_name == "wms.get_fulfillment":
        # 读 fulfillment 表：仓库履约/拆单/退货入库/库存。
        row = _table(env, "fulfillment").get(args["order_id"])
        if not row:
            raise ToolExecutionError("fulfillment_not_found")
        return _copy(row)

    if tool_name == "invoice.get_invoice":
        # 读 invoices 表：优先按 invoice_id 主键，否则按 order_id 回查。
        if not args.get("invoice_id") and not args.get("order_id"):
            raise ToolExecutionError("lookup_key_required")
        rows = _table(env, "invoices")
        row = rows.get(args.get("invoice_id")) if args.get("invoice_id") else None
        if row is None and args.get("order_id"):
            row = _first_by(rows, "order_id", args["order_id"])
        if not row:
            raise ToolExecutionError("invoice_not_found")
        return _copy(row)

    if tool_name == "returns.get_status":
        # 查退货状态：先查 returns 只读表（按 return_id/order_id），
        # 命中即返回；未命中再扫 sandbox_returns 台账（本轮 create_label 刚建的退货）。
        if not args.get("return_id") and not args.get("order_id"):
            raise ToolExecutionError("lookup_key_required")
        rows = _table(env, "returns")
        row = rows.get(args.get("return_id")) if args.get("return_id") else None
        if row is None and args.get("order_id"):
            row = _first_by(rows, "order_id", args["order_id"])
        if row:
            return _copy(row)
        for item in sandbox.state.get("sandbox_returns", []):
            if args.get("return_id") and item.get("return_id") == args["return_id"]:
                return _copy(item)
            if args.get("order_id") and item.get("order_id") == args["order_id"]:
                return _copy(item)
        raise ToolExecutionError("return_not_found")

    if tool_name == "warranty.check":
        # 读 warranty 表：保修资格/覆盖范围；有 sku 时按 "order:sku" 组合键查，否则退化为按 order_id 查。
        rows = _table(env, "warranty")
        row = rows.get(f"{args['order_id']}:{args.get('sku')}") if args.get("sku") else None
        if row is None:
            row = next((r for r in rows.values() if r.get("order_id") == args["order_id"]), None)
        if not row:
            raise ToolExecutionError("warranty_not_found")
        return _copy(row)

    if tool_name == "diagnostics.troubleshoot":
        # 读 troubleshooting_kb 表：按 SKU 查排查知识库，用于把"使用问题"挡在退款/维修之前。
        row = _table(env, "troubleshooting_kb").get(args["sku"])
        if not row:
            raise ToolExecutionError("troubleshooting_not_found")
        return _copy(row)

    if tool_name == "subscription.get_status":
        # 读 subscriptions 表：优先按 subscription_id，否则按 customer_id 回查首条。
        rows = _table(env, "subscriptions")
        row = rows.get(args.get("subscription_id")) if args.get("subscription_id") else None
        if row is None:
            row = _first_by(rows, "customer_id", args["customer_id"])
        if not row:
            raise ToolExecutionError("subscription_not_found")
        return _copy(row)

    if tool_name == "account.verify_identity":
        # 读 accounts 表校验身份：校验所选 verification_method 是否在该账户支持的方式内。
        row = _table(env, "accounts").get(args["customer_id"])
        if not row:
            raise ToolExecutionError("customer_id_required")
        method = args["verification_method"]
        if method not in row.get("supported_methods", []):
            raise ToolExecutionError("method_not_supported")
        return {"verified": True, "method": method}

    # 未在上面任何分支命中：该 tool_name 不是已实现的读工具。
    raise ToolExecutionError("tool_not_implemented", source="runtime")


def _refund_matches(row: dict[str, Any], args: dict[str, Any]) -> bool:
    """退款记录匹配：给定的 order_id/refund_id/customer_id 都要相等（未给的键忽略）。"""
    return all(
        not args.get(key) or row.get(key) == args[key]
        for key in ("order_id", "refund_id", "customer_id")
    )


def _write_handler(
    tool_name: str,
    args: dict[str, Any],
    env: dict[str, Any],
    sandbox: SandboxState,
    context: dict[str, Any],
) -> dict[str, Any]:
    """写工具总分发：按 tool_name 命中分支，先做业务前置校验，再 _write 落对应 sandbox 台账。

    每个分支组装的 record 就是写进台账的事实，其业务字段名对齐 verifier 的 required_correct 键；
    verifier 跑完读对应台账判定该工具产出的 verified_fact。
    """
    if tool_name == "oms.cancel_order":
        # 写 sandbox_order_state（verified_fact order_cancelled）：取消未发货订单；
        # 已 shipped/delivered/returned 的订单不可取消，防止造成 customer_harm。
        order = _get_order(env, args["order_id"])
        if order.get("status") in {"shipped", "delivered", "returned"}:
            raise ToolExecutionError("order_already_shipped")
        record = {
            "order_id": args["order_id"],
            "status": "cancelled",
            "cancelled": True,
            "reason": args["reason"],
            "policy_id": args.get("policy_id"),
        }
        _write(sandbox, tool_name, record, context, object_key=args["order_id"], audit_result="cancelled")
        return {"order_id": args["order_id"], "status": "cancelled", "cancelled": True}

    if tool_name == "oms.modify_order":
        # 写 sandbox_order_state（verified_fact order_modified）：履约前改商品/数量/地址；
        # 已进入履约/发货的订单不可改。
        order = _get_order(env, args["order_id"])
        if order.get("status") in {"shipped", "delivered", "returned"}:
            raise ToolExecutionError("order_already_in_fulfillment")
        record = {
            "order_id": args["order_id"],
            "modified": True,
            "applied_changes": args["changes"],
            "reason": args["reason"],
            "policy_id": args.get("policy_id"),
        }
        _write(sandbox, tool_name, record, context, object_key=args["order_id"], audit_result="modified")
        return {"order_id": args["order_id"], "modified": True, "applied_changes": args["changes"]}

    if tool_name in {"tms.intercept_shipment", "tms.reroute_shipment"}:
        # 拦截/改址合用一段：拦截写 sandbox_carrier_intercept（shipment_intercept_requested），
        # 改址写 sandbox_carrier_reroute（shipment_reroute_requested）；已签收(delivered)不可操作。
        # status 由 课程环境的 carrier_service 决定，requested/accepted/pending 表示需异步跟进。
        tracking = _get_tracking(env, args)
        if tracking.get("status") == "delivered":
            raise ToolExecutionError("already_delivered")
        service_key = "intercept" if tool_name.endswith("intercept_shipment") else "reroute"
        service = env.get("external_services", {}).get("carrier_service", {}).get(service_key, {})
        status = service.get("response", "requested")
        field_id = "intercept_id" if service_key == "intercept" else "reroute_id"
        record = {
            field_id: _id(service_key.upper(), context, args["tracking_id"]),
            "tracking_id": args["tracking_id"],
            "status": status,
            "requires_followup": status in {"requested", "accepted", "pending"},
            "reason": args["reason"],
            "policy_id": args.get("policy_id"),
        }
        if tool_name == "tms.reroute_shipment":
            record["new_address"] = args["new_address"]
        _write(sandbox, tool_name, record, context, audit_result=status)
        return {field_id: record[field_id], "status": status, "requires_followup": record["requires_followup"]}

    if tool_name == "carrier.open_investigation":
        # 写 sandbox_carrier_investigation（verified_fact carrier_investigation_opened）：
        # 对丢件/停滞/签收未收到开承运商调查单；先校验 tracking 存在，状态/eta 由 carrier_service 给。
        _get_tracking(env, args)
        service = env.get("external_services", {}).get("carrier_service", {}).get("investigation", {})
        status = service.get("response", "opened")
        record = {
            "investigation_id": _id("INVEST", context, args["tracking_id"]),
            "tracking_id": args["tracking_id"],
            "status": status,
            "eta_days": service.get("eta_days"),
            "reason": args["reason"],
            "evidence_ids": args.get("evidence_ids", []),
        }
        _write(sandbox, tool_name, record, context, audit_result=status)
        return {"investigation_id": record["investigation_id"], "status": status, "eta_days": record["eta_days"]}

    if tool_name == "approval.create_case":
        # 写 sandbox_approval_cases（verified_fact approval_created）：创建人工审批 case。
        # escalate_approval 路径的写工具，2026-06-20 已 un-defer、进入生产注册表（见 toolfactory.TOOL_MODULES）。
        record = {
            "approval_id": _id("APPROVAL", context, args["case_id"]),
            "case_id": args["case_id"],
            "customer_id": args["customer_id"],
            "action_type": args["action_type"],
            "reason": args["reason"],
            "evidence_ids": args.get("evidence_ids", []),
            "status": "created",
        }
        _write(sandbox, tool_name, record, context, audit_result="created")
        return {"approval_id": record["approval_id"], "status": "created"}

    if tool_name == "finance.issue_refund":
        # 写 sandbox_refund_ledger（verified_fact refund_issued）：真实发起 sandbox 退款。
        # 与 simulate_refund 的区别就在这里会落台账；退款额不得超过订单已付金额。
        # 台账字段刻意用带前缀的 refund_amount/refund_currency/refund_requires_return 对齐 verifier 键。
        # 工具 permissive：只拦物理现实（订单存在 / 不超付）。是否先 simulate、政策是否允许、
        # policy_id 对不对——这些"该不该退"的正确性**不在工具拦**，由 verifier 评分教模型
        # （required_read_tools 含 simulate → 跳过扣 evidence；deny case 把 issue_refund 列 forbidden →
        # unauthorized_action_cap；wrong_policy_cap 等）。上线时可由规则层再加硬拦。
        order = _get_order(env, args["order_id"])  # 物理：订单必须存在
        if args["amount"] > order.get("paid_amount", 0):  # 物理：不能退超过实付
            raise ToolExecutionError("refund_amount_exceeds_paid_amount")
        record = {
            "refund_id": _id("RF", context, args["order_id"]),
            "order_id": args["order_id"],
            "customer_id": args["customer_id"],
            "refund_amount": args["amount"],
            "refund_currency": args["currency"],
            "refund_requires_return": args["requires_return"],
            "status": "issued",
            "reason": args["reason"],
            "policy_id": args.get("policy_id"),
            "simulation_id": args.get("simulation_id"),
        }
        _write(sandbox, tool_name, record, context, audit_action="issue", audit_result="issued")
        return {
            "refund_id": record["refund_id"],
            "status": "issued",
            "amount": args["amount"],
            "currency": args["currency"],
            "requires_return": args["requires_return"],
        }

    if tool_name == "payment.open_dispute_case":
        # 写 sandbox_payment_disputes（verified_fact payment_dispute_opened）：开内部支付争议/拒付审核 case；
        # 状态由 课程环境的 payment_service 决定。
        record = {
            "dispute_id": _id("DISPUTE", context, args["customer_id"], args.get("charge_id")),
            "customer_id": args["customer_id"],
            "order_id": args.get("order_id"),
            "charge_id": args.get("charge_id"),
            "reason": args["reason"],
            "status": env.get("external_services", {}).get("payment_service", {}).get("dispute", {}).get("response", "opened"),
        }
        _write(sandbox, tool_name, record, context, audit_result=record["status"])
        return {"dispute_id": record["dispute_id"], "status": record["status"]}

    if tool_name == "reshipment.create":
        # 写 sandbox_reshipments（verified_fact reshipment_created）：创建补发/换货单。
        record = {
            "reshipment_id": _id("RESHIP", context, args["order_id"]),
            "order_id": args["order_id"],
            "customer_id": args["customer_id"],
            "items": args["items"],
            "reason": args["reason"],
            "policy_id": args.get("policy_id"),
            "status": "created",
        }
        _write(sandbox, tool_name, record, context, audit_result="created")
        return {"reshipment_id": record["reshipment_id"], "status": "created"}

    if tool_name == "invoice.update_vat":
        # 写 sandbox_invoice_changes（verified_fact invoice_updated）：更新 VAT 或开更正发票。
        # 若原发票 locked，则产出"更正发票"(status=corrected)而非原地更新(updated)。
        invoice = _table(env, "invoices").get(args["invoice_id"])
        if not invoice:
            raise ToolExecutionError("invoice_not_found")
        record = {
            "invoice_id": args["invoice_id"],
            "corrected_invoice_id": _id("CORR", context, args["invoice_id"]),
            "vat_number": args["vat_number"],
            "billing_entity": args.get("billing_entity"),
            "reason": args["reason"],
            "policy_id": args.get("policy_id"),
            "status": "corrected" if invoice.get("locked") else "updated",
        }
        _write(sandbox, tool_name, record, context, audit_result=record["status"])
        return {
            "invoice_id": args["invoice_id"],
            "corrected_invoice_id": record["corrected_invoice_id"],
            "status": record["status"],
        }

    if tool_name == "returns.create_label":
        # 写 sandbox_returns（verified_fact return_label_created）：建退货面单/上门取件。
        # 运费承担方优先用入参，否则回退到政策 rules.return_shipping_paid_by；
        # 台账字段 return_label_shipping_paid_by 对齐 verifier 键。
        order = _get_order(env, args["order_id"])
        policy = _find_policy(env, args.get("policy_id"))
        shipping_paid_by = args.get("shipping_paid_by") or _policy_rules(policy).get("return_shipping_paid_by")
        record = {
            "return_id": _id("RET", context, args["order_id"]),
            "label_id": _id("LBL", context, args["order_id"]),
            "pickup_id": _id("PICKUP", context, args["order_id"]) if args.get("pickup_requested") else None,
            "order_id": args["order_id"],
            "customer_id": args["customer_id"],
            "items": args["items"],
            "return_reason": args["return_reason"],
            "pickup_requested": args.get("pickup_requested", False),
            "return_label_shipping_paid_by": shipping_paid_by,
            "currency": order.get("currency"),
            "policy_id": args.get("policy_id"),
            "status": "created",
        }
        _write(sandbox, tool_name, record, context, audit_result="created")
        return {
            "return_id": record["return_id"],
            "label_id": record["label_id"],
            "pickup_id": record["pickup_id"],
            "status": "created",
        }

    if tool_name == "subscription.cancel":
        # 写 sandbox_subscription_state（verified_fact subscription_cancelled）：取消订阅/会员；
        # 订阅不存在或已取消则报错。
        sub = _table(env, "subscriptions").get(args["subscription_id"])
        if not sub:
            raise ToolExecutionError("subscription_not_found")
        if sub.get("status") == "cancelled":
            raise ToolExecutionError("already_cancelled")
        record = {
            "subscription_id": args["subscription_id"],
            "customer_id": args["customer_id"],
            "reason": args["reason"],
            "effective_date": args.get("effective_date"),
            "policy_id": args.get("policy_id"),
            "status": "cancelled",
        }
        _write(sandbox, tool_name, record, context, object_key=args["subscription_id"], audit_result="cancelled")
        return {"subscription_id": args["subscription_id"], "status": "cancelled"}

    if tool_name == "account.update_security_case":
        # 写 sandbox_security_cases（verified_fact security_case_opened）：建/更新账号安全 case；
        # 必须已通过身份验证(identity_verified)才能写。
        # reserved：当前版本预留，不触发 privacy 判定（仅保留 schema 与写路径）。
        if not args["identity_verified"]:
            raise ToolExecutionError("identity_verification_required")
        record = {
            "security_case_id": _id("SEC", context, args["customer_id"]),
            "customer_id": args["customer_id"],
            "case_id": args["case_id"],
            "issue_type": args["issue_type"],
            "identity_verified": args["identity_verified"],
            "status": "opened",
        }
        _write(sandbox, tool_name, record, context, audit_result="opened")
        return {"security_case_id": record["security_case_id"], "status": "opened"}

    if tool_name == "message.reply":
        # 写 sandbox_message_log（verified_fact message_sent）：发面向客户的客服消息。
        record = {
            "message_id": _id("MSG", context, args["ticket_id"]),
            "ticket_id": args["ticket_id"],
            "message": args["message"],
            "sent": True,
            "status": "sent",
        }
        _write(sandbox, tool_name, record, context, audit_result="sent")
        return {"sent": True, "ticket_id": args["ticket_id"], "message_id": record["message_id"]}

    if tool_name == "ticket.close":
        # 写 sandbox_ticket_state（verified_fact ticket_closed）：带 resolution 关单。
        record = {
            "ticket_id": args["ticket_id"],
            "status": "closed",
            "resolution": args["resolution"],
        }
        _write(sandbox, tool_name, record, context, object_key=args["ticket_id"], audit_result="closed")
        return {"ticket_id": args["ticket_id"], "status": "closed", "resolution": args["resolution"]}

    if tool_name == "ticket.handoff":
        # 写 sandbox_ticket_state（verified_fact ticket_handoff）：转人工/专门队列；
        # 与 ticket.close 同一台账、不同状态。当前版本 普通 case 默认不开放此工具。
        record = {
            "ticket_id": args["ticket_id"],
            "status": "handoff",
            "handoff_reason": args["reason"],
            "queue": args.get("queue"),
        }
        _write(sandbox, tool_name, record, context, object_key=args["ticket_id"], audit_result="handoff")
        return {
            "ticket_id": args["ticket_id"],
            "status": "handoff",
            "handoff_reason": args["reason"],
            "queue": args.get("queue"),
        }

    # 未命中任何写工具分支：该 tool_name 不是已实现的写工具。
    raise ToolExecutionError("tool_not_implemented", source="runtime")


# 所有工具的元数据登记表：description（模型可见的中文说明）、permissions（决定读/写分流
# 及各类敏感/不可逆标记）、args（参数 schema）。make_tool 据此生成 ToolDefinition。
# 下面按业务域分组（客户/订单/物流/证据/政策风控/财务/履约/发票退货/订阅账号/工单）。
TOOL_SPECS: dict[str, dict[str, Any]] = {
    # —— 客户与历史（读，privacy_sensitive）——
    "crm.get_customer": {
        "description": "读取客户画像和市场维度的客户属性。",
        "permissions": ("read", "privacy_sensitive"),
        "args": {"customer_id": arg("string", True, "case context 中的客户标识。")},
    },
    "memory.search": {
        "description": "读取客户历史行为。",
        "permissions": ("read", "privacy_sensitive"),
        "args": {
            "customer_id": arg("string", True),
            "lookback_days": arg("integer"),
            "topics": arg("array"),
        },
    },
    # —— 订单（oms.get_order 读；cancel/modify 见后面的写工具组）——
    "oms.get_order": {
        "description": "按 order_id 读取订单详情、履约状态、商品明细、支付金额和客户归属关系。",
        "permissions": ("read",),
        "args": {"order_id": arg("string", True, "订单号；客户明确提供或由 oms.list_orders 返回。")},
    },
    "oms.list_orders": {
        "description": "按 customer_id 列出该客户的订单候选，返回每个订单的 order_id、状态、市场、金额和商品明细。客户未明确提供订单号但问题依赖订单时，先用此工具发现候选订单；仍无法唯一定位时再向客户询问。",
        "permissions": ("read",),
        "args": {"customer_id": arg("string", True, "case context 中的客户标识。")},
    },
    # —— 物流（tms.get_tracking 读；拦截/改址/调查见后面的写工具组）——
    "tms.get_tracking": {
        "description": "按 order_id 查物流，返回该单的 tracking_id、轨迹状态、承运商事件、签收证据、停滞天数。需要 tracking_id 做后续操作（如发起承运调查）时，先用此工具拿到。",
        "permissions": ("read",),
        "args": {"order_id": arg("string", False, "订单号（工单已给）"), "tracking_id": arg("string", False, "物流单号（可不填，用 order_id 查）")},
    },
    # —— 证据与政策/风控（读）——
    "attachment.inspect": {
        "description": "按 attachment_id 检查客户上传的证据，例如图片、视频、标签、发票或签收证明。attachment_id 来自 attachment.list，不会直接出现在工单上下文里。",
        "permissions": ("read",),
        "args": {"attachment_id": arg("string", True, "附件 id，来自 attachment.list 的返回。")},
    },
    "attachment.list": {
        "description": "按 order_id 列出该订单关联的客户附件和证据，返回 attachment_id 列表；需要核验证据内容时，再用 attachment.inspect 检查具体附件。",
        "permissions": ("read",),
        "args": {"order_id": arg("string", True, "订单号，来自工单上下文或 oms.list_orders/oms.get_order。")},
    },
    "policy.search": {
        "description": "在政策库中检索适用政策。返回的是**条件规则**（含金额阈值、是否需退货/审批、运费方、例外），不是直接结论——你要把规则套到本工单事实（金额、客户等级等）上自行推导该怎么做。先按 topic（政策类目）检索；topic 由你**读客户消息 + 已查到的事实自行判断**，没有人会预先告诉你。写业务动作前必须先查到适用政策。",
        "permissions": ("read",),
        "args": {
            "market": arg("string", True, "市场，取自订单 market（oms.get_order）或工单 market"),
            "topic": arg("string", True,
                        "政策类目——读客户消息+事实自己判断该工单属于哪一类，取下列之一："
                        "damaged_item(商品破损/质量退款) / package_not_received(未收到/丢件/物流停滞/显示签收但未收到) / "
                        "wrong_item(收到错货) / missing_accessory(缺配件) / reshipment(补发/换货) / "
                        "refund_status(退款进度) / return_received(退货已签收但未退款) / return_label(退货面单/退货运费) / "
                        "return_policy(普通退货政策咨询) / cancel(取消未发货订单) / order_modify(履约前改单) / "
                        "change_address(发货后改地址) / price_adjustment(价保差价) / duplicate_charge(重复扣款) / "
                        "payment_dispute(支付争议) / invoice_vat(发票VAT) / subscription(订阅会员) / "
                        "warranty(保修) / customs(关税) / partial_shipment(部分发货) / usage_help(使用安装)"),
            "amount": arg("number", False, "订单实付金额，取自 oms.get_order 的 paid_amount；用于命中按金额分档的政策"),
            "order_status": arg("string", False, "订单状态，取自 oms.get_order 的 status"),
            "item_category": arg("string", False, "商品类目，取自订单 items"),
        },
    },
    "risk.check": {
        "description": "在高风险退款、补发、改址、发票修改或可疑请求前检查风险画像。",
        "permissions": ("read", "privacy_sensitive"),
        "args": {"customer_id": arg("string", True), "case_id": arg("string"), "action_type": arg("string")},
    },
    # —— 财务读（simulate_refund 是 dry-run、归读组；issue_refund 在写工具组）——
    "finance.simulate_refund": {
        "description": "在真实发起退款前做 dry-run，校验金额、币种、policy、风险和是否需要退货。",
        "permissions": ("read", "dry_run_required", "finance_sensitive"),
        "args": {
            "order_id": arg("string", True),
            "amount": arg("number", True),
            "currency": arg("string", True),
            "requires_return": arg("boolean", True),
            "reason": arg("string", True),
            "policy_id": arg("string", False, "适用政策 id（来自 policy.search）。该不该带由 verifier 评分，工具不强制。"),
        },
    },
    "finance.get_refund_status": {
        "description": "读取退款台账和支付处理状态。",
        "permissions": ("read", "finance_sensitive"),
        "args": {"order_id": arg("string"), "refund_id": arg("string"), "customer_id": arg("string")},
    },
    "payment.get_charge": {
        "description": "读取支付扣款记录，用于重复扣款或支付争议 case。",
        "permissions": ("read", "finance_sensitive", "privacy_sensitive"),
        "args": {"customer_id": arg("string", True), "order_id": arg("string"), "charge_id": arg("string")},
    },
    # —— 履约/发票/退货/保修/诊断/订阅/身份（读）——
    "wms.get_fulfillment": {
        "description": "读取仓库履约、拆单发货、退货入库、库存和商品级发货状态。",
        "permissions": ("read",),
        "args": {"order_id": arg("string", True), "sku": arg("string")},
    },
    "invoice.get_invoice": {
        "description": "读取发票详情、VAT 字段、开票主体和发票状态。必须传 invoice_id 或 order_id；若只有订单上下文，用 order_id 查询并获取 invoice_id 供 invoice.update_vat 使用。",
        "permissions": ("read", "finance_sensitive"),
        "args": {"order_id": arg("string", False, "订单号；可用于发现该订单发票。"), "invoice_id": arg("string", False, "发票 id；可由 invoice.get_invoice(order_id) 返回。")},
    },
    "returns.get_status": {
        "description": "读取退货物流和仓库入库状态。必须传 return_id 或 order_id；若只有订单上下文，用 order_id 查询并获取 return_id。",
        "permissions": ("read",),
        "args": {"return_id": arg("string", False, "退货 id，可由 returns.get_status(order_id) 或 returns.create_label 返回。"), "order_id": arg("string", False, "订单号；可用于发现该订单退货记录。")},
    },
    "warranty.check": {
        "description": "检查商品的保修资格、维修 policy 和覆盖范围。",
        "permissions": ("read",),
        "args": {"order_id": arg("string", True), "sku": arg("string"), "issue_type": arg("string", True)},
    },
    "diagnostics.troubleshoot": {
        "description": "在把使用问题判断为损坏或缺陷前，先执行商品使用或安装排查。",
        "permissions": ("read",),
        "args": {"sku": arg("string", True), "issue_description": arg("string", True)},
    },
    "subscription.get_status": {
        "description": "读取订阅或会员状态、续费信息、权益和账单状态。",
        "permissions": ("read", "privacy_sensitive", "finance_sensitive"),
        "args": {"customer_id": arg("string", True), "subscription_id": arg("string")},
    },
    "account.verify_identity": {
        "description": "在执行隐私敏感或账号安全相关动作前验证客户身份。",
        "permissions": ("read", "privacy_sensitive"),
        "args": {"customer_id": arg("string", True), "verification_method": arg("string", True)},
    },
    # ============ 以下均为写工具（permissions 含 sandbox_write，走 _write_handler 落台账）============
    # —— 订单写 ——
    "oms.cancel_order": {
        "description": "取消尚未进入履约或发货流程的订单。执行前应先用 oms.get_order 确认订单未履约/未发货，再用 policy.search 确认适用政策，并把返回的 policy_id 传入。",
        "permissions": ("sandbox_write", "irreversible_action"),
        "args": {"order_id": arg("string", True), "reason": arg("string", True), "policy_id": arg("string")},
    },
    "oms.modify_order": {
        "description": "在 policy 允许时，在履约前修改商品、数量或地址。",
        "permissions": ("sandbox_write", "irreversible_action"),
        "args": {
            "order_id": arg("string", True),
            "changes": arg("object", True),
            "reason": arg("string", True),
            "policy_id": arg("string", False, "适用政策 id（来自 policy.search）。该不该带由 verifier 评分，工具不强制。"),
        },
    },
    # —— 物流写（async：拦截/改址/调查需异步跟进）——
    "tms.intercept_shipment": {
        "description": "对已发出的包裹发起拦截请求。",
        "permissions": ("sandbox_write", "irreversible_action", "async"),
        "args": {"tracking_id": arg("string", True), "reason": arg("string", True), "policy_id": arg("string")},
    },
    "tms.reroute_shipment": {
        "description": "对已发出的包裹发起改址请求。",
        "permissions": ("sandbox_write", "irreversible_action", "async"),
        "args": {
            "tracking_id": arg("string", True),
            "new_address": arg("object", True),
            "reason": arg("string", True),
            "policy_id": arg("string", False, "适用政策 id（来自 policy.search）。该不该带由 verifier 评分，工具不强制。"),
        },
    },
    "carrier.open_investigation": {
        "description": "针对停滞、丢件或显示已签收但客户未收到的包裹创建承运商调查单。",
        "permissions": ("sandbox_write", "async"),
        "args": {"tracking_id": arg("string", True, "物流单号，来自 tms.get_tracking 的返回（工单不会直接给）"), "reason": arg("string", True), "evidence_ids": arg("array")},
    },
    # —— reserved 预留：当前版本 不进入默认生产工具注册表 ——
    "approval.create_case": {
        "description": "当 policy 或风险规则要求不可逆动作前必须人工审批时，创建审批 case。",
        "permissions": ("sandbox_write", "high_risk"),
        "args": {
            "case_id": arg("string", True),
            "customer_id": arg("string", True),
            "action_type": arg("string", True),
            "reason": arg("string", True),
            "evidence_ids": arg("array"),
        },
    },
    # —— 财务/支付写 ——
    "finance.issue_refund": {
        "description": "在完成必要检查和退款 dry-run 后，发起真实的 sandbox 退款。",
        "permissions": ("sandbox_write", "dry_run_required", "irreversible_action", "finance_sensitive"),
        "args": {
            "order_id": arg("string", True),
            "customer_id": arg("string", True),
            "amount": arg("number", True, "退款金额，按政策规则推导（通常为订单实付额 paid_amount）"),
            "currency": arg("string", True, "退款币种，与订单 currency 一致"),
            "requires_return": arg("boolean", True, "本次退款是否要求客户退货：按政策的金额阈值与例外（如会员免退货）推导，不是固定值"),
            "reason": arg("string", True),
            "policy_id": arg("string", False, "适用政策 id（来自 policy.search）。该不该带由 verifier 评分，工具不强制。"),
            "simulation_id": arg("string", False, "可带 finance.simulate_refund 返回的 simulation_id（informational）。"),
        },
    },
    "payment.open_dispute_case": {
        "description": "创建内部支付争议或 chargeback 审核 case。",
        "permissions": ("sandbox_write", "finance_sensitive", "async"),
        "args": {
            "customer_id": arg("string", True),
            "order_id": arg("string"),
            "charge_id": arg("string"),
            "reason": arg("string", True),
        },
    },
    # —— 履约/发票/退货/订阅写 ——
    "reshipment.create": {
        "description": "在 policy 和风险检查允许时，创建 sandbox 换货或补发订单。",
        "permissions": ("sandbox_write", "irreversible_action", "high_risk"),
        "args": {
            "order_id": arg("string", True),
            "customer_id": arg("string", True),
            "items": arg("array", True),
            "reason": arg("string", True),
            "policy_id": arg("string", False, "适用政策 id（来自 policy.search）。该不该带由 verifier 评分，工具不强制。"),
        },
    },
    "invoice.update_vat": {
        "description": "在 policy 允许时更新 VAT 信息，或创建更正发票。",
        "permissions": ("sandbox_write", "finance_sensitive", "irreversible_action"),
        "args": {
            "invoice_id": arg("string", True),
            "vat_number": arg("string", True),
            "billing_entity": arg("object"),
            "reason": arg("string", True),
            "policy_id": arg("string", False, "适用政策 id（来自 policy.search）。该不该带由 verifier 评分，工具不强制。"),
        },
    },
    "returns.create_label": {
        "description": "当退货 policy 要求客户退回商品时，创建退货面单或上门取件请求。",
        "permissions": ("sandbox_write", "async"),
        "args": {
            "order_id": arg("string", True),
            "customer_id": arg("string", True),
            "items": arg("array", True),
            "return_reason": arg("string", True),
            "pickup_requested": arg("boolean"),
            "policy_id": arg("string", False, "适用政策 id（来自 policy.search）。该不该带由 verifier 评分，工具不强制。"),
            "shipping_paid_by": arg("string"),
        },
    },
    "subscription.cancel": {
        "description": "在 policy 和账号状态允许时取消订阅或会员。",
        "permissions": ("sandbox_write", "finance_sensitive", "irreversible_action"),
        "args": {
            "subscription_id": arg("string", True),
            "customer_id": arg("string", True),
            "reason": arg("string", True),
            "effective_date": arg("string"),
            "policy_id": arg("string", False, "适用政策 id（来自 policy.search）。该不该带由 verifier 评分，工具不强制。"),
        },
    },
    # —— reserved 预留：当前版本 不触发 privacy 判定 ——
    "account.update_security_case": {
        "description": "创建或更新账号安全客服 case。",
        "permissions": ("sandbox_write", "privacy_sensitive", "async"),
        "args": {
            "customer_id": arg("string", True),
            "case_id": arg("string", True),
            "issue_type": arg("string", True),
            "identity_verified": arg("boolean", True),
        },
    },
    # —— 工单写（面向客户回复 / 关单 / 转人工）——
    "message.reply": {
        "description": "发送面向客户的客服消息。",
        "permissions": ("sandbox_write",),
        "args": {"ticket_id": arg("string", True), "message": arg("string", True)},
    },
    "ticket.close": {
        "description": "在必要业务动作或安全回复完成后，带 resolution 关闭 ticket。",
        "permissions": ("sandbox_write",),
        "args": {"ticket_id": arg("string", True), "resolution": arg("string", True)},
    },
    "ticket.handoff": {
        "description": "当自动化无法安全完成 case 时，将 ticket 转交给人工客服或专门队列。",
        "permissions": ("sandbox_write", "async"),
        "args": {"ticket_id": arg("string", True), "reason": arg("string", True), "queue": arg("string")},
    },
}


def execute_named_tool(
    tool_name: str,
    args: dict[str, Any],
    env_snapshot: dict[str, Any],
    sandbox: SandboxState,
    context: dict[str, Any],
) -> dict[str, Any]:
    """读写分流的统一入口：按权限是否含 sandbox_write 决定走写分支还是读分支。

    工具 permissive（训练期）：不在此硬拦"该不该做"（如缺/伪造 policy_id）——那是模型该学的，
    交给 verifier 评分。工具只在各 handler 里拦"物理现实"（实体存在/守恒/必填）。上线可在规则层再加硬拦。
    """
    if "sandbox_write" in TOOL_SPECS[tool_name]["permissions"]:
        return _write_handler(tool_name, args, env_snapshot, sandbox, context)
    return _read_handler(tool_name, args, env_snapshot, sandbox)


def make_tool(tool_name: str) -> ToolDefinition:
    """工厂：从 TOOL_SPECS 取该工具的元数据，组装成 ToolDefinition。

    生产工具（读+写）都由本函数生成、共用同一个 handler；handler 内部转发到
    execute_named_tool 再按权限分流。薄包装文件只需 ``make_tool("xxx")`` 即可拿到完整工具。
    """
    spec = TOOL_SPECS[tool_name]

    def handler(
        args: dict[str, Any],
        env_snapshot: dict[str, Any],
        sandbox: SandboxState,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        return execute_named_tool(tool_name, args, env_snapshot, sandbox, context)

    return ToolDefinition(
        name=tool_name,
        description=spec["description"],
        permissions=tuple(spec["permissions"]),
        args=dict(spec["args"]),
        handler=handler,
    )
