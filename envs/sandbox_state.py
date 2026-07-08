# 版权所有 © 2026 深圳途明智启科技有限公司。保留所有权利。
# 未经书面许可，任何单位或个人不得复制、传播、发布、转卖、改编、仿制或用于商业用途。
# 侵权必究。

"""按命名空间隔离的 sandbox 可写状态。

sandbox 是"模型可改的世界"——写工具（发退款、建退货面单、关单……）把记录写进这里的台账（ledger）。
只读业务表在 env_snapshot 里不可变，而所有副作用都落在 SandboxState；跑完后 export() 出来的
就是 环境数据规范 说的 sandbox_final_state，是 verifier 判 require/claim 的事实地基。

两条贯穿全模块的设计：
1. namespace 隔离：每条 rollout 的写都打上 namespace_id，读取时按它过滤。verl 并发跑同一 case 的
   多条 rollout 时，这保证 A 的退款不会被 B 读到。
2. 台账形状二分：多数台账是"追加列表"（每次写 append 一条）；少数表示"某对象的当前状态"
   （订单/订阅/工单），用 dict 按对象 key 存（见 DICT_LEDGERS），后写覆盖先写，反映最新状态。
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from schemas.env_schema import default_sandbox


# 写工具 -> {落哪个台账 ledger, 提供哪个 verified_fact_key} 的映射表。
# fact 是 verifier 用的"事实键"：某写工具成功执行后该 fact=True，require/claim 据此判 outcome。
# 例：finance.issue_refund 落 sandbox_refund_ledger，置 refund_issued=True；
#     若 trajectory 没成功调它，台账为空 -> refund_issued=false -> outcome=0（甚至触发 false_promise_cap）。
# 注意 oms.cancel_order/oms.modify_order 共用 sandbox_order_state，
# ticket.close/ticket.handoff 共用 sandbox_ticket_state（同表不同 fact，靠 fact 字段区分，见 records_for_tool）。
WRITE_TOOL_FACTS: dict[str, dict[str, str]] = {
    "finance.issue_refund": {"ledger": "sandbox_refund_ledger", "fact": "refund_issued"},
    "returns.create_label": {"ledger": "sandbox_returns", "fact": "return_label_created"},
    "reshipment.create": {"ledger": "sandbox_reshipments", "fact": "reshipment_created"},
    "oms.cancel_order": {"ledger": "sandbox_order_state", "fact": "order_cancelled"},
    "oms.modify_order": {"ledger": "sandbox_order_state", "fact": "order_modified"},
    "carrier.open_investigation": {
        "ledger": "sandbox_carrier_investigation",
        "fact": "carrier_investigation_opened",
    },
    "tms.intercept_shipment": {
        "ledger": "sandbox_carrier_intercept",
        "fact": "shipment_intercept_requested",
    },
    "tms.reroute_shipment": {
        "ledger": "sandbox_carrier_reroute",
        "fact": "shipment_reroute_requested",
    },
    "approval.create_case": {"ledger": "sandbox_approval_cases", "fact": "approval_created"},
    "payment.open_dispute_case": {
        "ledger": "sandbox_payment_disputes",
        "fact": "payment_dispute_opened",
    },
    "invoice.update_vat": {"ledger": "sandbox_invoice_changes", "fact": "invoice_updated"},
    "subscription.cancel": {
        "ledger": "sandbox_subscription_state",
        "fact": "subscription_cancelled",
    },
    "account.update_security_case": {
        "ledger": "sandbox_security_cases",
        "fact": "security_case_opened",
    },
    "message.reply": {"ledger": "sandbox_message_log", "fact": "message_sent"},
    "ticket.close": {"ledger": "sandbox_ticket_state", "fact": "ticket_closed"},
    "ticket.handoff": {"ledger": "sandbox_ticket_state", "fact": "ticket_handoff"},
}

# "状态型"台账：用 dict（按对象 key 存）而非 list。
# 为什么按 key 存：订单/订阅/工单关心的是"该对象的当前状态"（取消/改单/关单/转人工），
# 同一对象被多次写时应后写覆盖先写、只留最新态，而不是堆叠多条历史。
# 不在此集合内的台账都是"事件流"语义（每次写 append 一条，如退款、退货面单），保留全部历史。
DICT_LEDGERS = {
    "sandbox_order_state",
    "sandbox_subscription_state",
    "sandbox_ticket_state",
}


class SandboxState:
    """单条 rollout 命名空间内的可变 sandbox 状态。

    持有 self.state（全部台账的 dict）与 self.namespace_id（本 rollout 的隔离键）。
    构造时以 default_sandbox() 的空台账骨架打底，再叠加 case 的 sandbox_initial 初始值。
    """

    def __init__(self, initial: dict[str, Any] | None = None, namespace_id: str | None = None):
        # 先取标准的空台账骨架（保证所有台账键都存在，避免后续 KeyError）。
        base = default_sandbox()
        if initial:
            # 叠加 case 的初始状态；deepcopy 防止改动回流污染传入的字典/共享环境数据。
            base.update(deepcopy(initial))
        # state 是本 rollout 的可写世界。只读表永远不放这里，避免工具把 环境数据真值改掉。
        self.state = base
        # namespace_id 是读写过滤的默认命名空间；records_for_tool 不显式传时会使用它。
        self.namespace_id = namespace_id

    @classmethod
    def from_env_snapshot(cls, env_snapshot: dict[str, Any], namespace_id: str) -> "SandboxState":
        """从 env_snapshot 的 sandbox_initial 段构造本 rollout 的 SandboxState。

        sandbox_initial 是 case 预置的台账初值（一般为空列表/承自只读态的对象）；
        必须传入 namespace_id 以便后续所有写都打上隔离标记。
        """
        return cls(env_snapshot.get("sandbox_initial", {}), namespace_id)

    def export(self) -> dict[str, Any]:
        """深拷贝导出全部台账，即 sandbox_final_state（交给 verifier 判分的最终世界状态）。

        deepcopy 是为了导出后即使内部 state 再被改也不影响已导出的快照。
        """
        # 导出的是完整 sandbox，不只导出本 namespace 的记录；verifier 读取时再按 namespace 过滤。
        return deepcopy(self.state)

    def write_record(
        self,
        tool_name: str,
        record: dict[str, Any],
        *,
        namespace_id: str,
        run_id: str,
        case_id: str,
        rollout_id: str,
        tool_call_id: str,
        object_key: str | None = None,
        audit_action: str | None = None,
        audit_result: str | None = None,
    ) -> dict[str, Any]:
        """把一条写记录落进对应台账，并同步写一条审计日志；返回入账后的完整记录。

        流程：
          1. 据 WRITE_TOOL_FACTS 查到该写工具对应的 ledger 与 fact（无映射的工具直接报错）。
          2. 把业务 record 补全为 enriched：附上 tool 名、namespace/run/case/rollout/tool_call 等溯源字段，
             并置 <fact>=True 与 verified_fact_key=fact —— 这两项是 verifier 直接读的"该动作已发生"信号。
          3. 按台账形状落库：DICT_LEDGERS 按对象 key 覆盖式写（最新态）；其余 append（事件流）。
          4. 始终向 sandbox_audit_log 追加一条审计，记录这次写的来龙去脉。
        所有写都带 namespace_id，确保并发 rollout 之间互不污染。
        """
        if tool_name not in WRITE_TOOL_FACTS:
            # 非写工具/未登记映射的工具不应走到这里，属调用方逻辑错误。
            raise KeyError(f"tool has no sandbox write mapping: {tool_name}")
        # mapping 是写工具到台账/事实键的唯一来源，verifier 和写工具都共享这套口径。
        mapping = WRITE_TOOL_FACTS[tool_name]
        ledger = mapping["ledger"]  # 目标台账名
        fact = mapping["fact"]      # 该写工具对应的 verified_fact_key
        # enriched：业务字段 + 溯源元信息 + 事实标记。fact=True 与 verified_fact_key 让 verifier 一眼认出该事实。
        enriched = {
            **record,
            "tool": tool_name,
            "namespace_id": namespace_id,
            "run_id": run_id,
            "case_id": case_id,
            "rollout_id": rollout_id,
            "tool_call_id": tool_call_id,
            fact: True,
            "verified_fact_key": fact,
        }
        if ledger in DICT_LEDGERS:
            # 状态型台账：选一个稳定对象 key（优先显式 object_key，再按业务 id 兜底，最后用 tool_call_id 保底），
            # 同 key 后写覆盖先写，使台账只反映该对象的最新状态。
            key = object_key or record.get("order_id") or record.get("subscription_id")
            key = key or record.get("ticket_id") or tool_call_id
            bucket = self.state.setdefault(ledger, {})
            bucket[key] = enriched
        else:
            # 事件流台账：直接 append，保留每次写的完整历史。
            self.state.setdefault(ledger, []).append(enriched)
        # 审计日志：每次写都留痕。
        # action 缺省取工具名最后一段（如 finance.issue_refund -> issue_refund）；
        # result 缺省取记录的 status，否则记 "ok"。审计仅供追溯/调试，不作 outcome 真值。
        self.state.setdefault("sandbox_audit_log", []).append(
            {
                "tool": tool_name,
                "action": audit_action or tool_name.split(".")[-1],
                "args": deepcopy(record),
                "result": audit_result or record.get("status") or "ok",
                "namespace_id": namespace_id,
                "run_id": run_id,
                "case_id": case_id,
                "rollout_id": rollout_id,
                "tool_call_id": tool_call_id,
            }
        )
        return deepcopy(enriched)

    def records_for_tool(self, tool_name: str, namespace_id: str | None = None) -> list[dict[str, Any]]:
        """取某写工具在指定命名空间下写出的全部记录（已隔离过滤）。

        三层过滤：
          1. 取该工具对应的台账（dict 台账取 values，list 台账原样）。
          2. 按 namespace_id 过滤，只留本 rollout 的记录（默认用 self.namespace_id），保证并发隔离。
          3. 若该台账被多个工具共用（如 sandbox_order_state 被 cancel/modify 共用），
             再按 fact 字段 / tool 名细分，只留属于本工具的那条 fact——
             否则 cancel 的记录会被误算成 modify 的，verifier 判 order_cancelled vs order_modified 就会出错。
        独占台账（仅一个写工具）则跳过第 3 步，因为不存在混淆。
        """
        mapping = WRITE_TOOL_FACTS.get(tool_name)
        if not mapping:
            return []  # 非写工具，无台账记录
        # 取台账：dict 台账默认空 dict、list 台账默认空 list，避免类型不一致。
        ledger = self.state.get(mapping["ledger"], [] if mapping["ledger"] not in DICT_LEDGERS else {})
        if isinstance(ledger, dict):
            records = list(ledger.values())  # 状态型台账：取所有对象的当前状态
        else:
            records = list(ledger)
        # 命名空间过滤：只保留属于本 rollout 的写，实现并发隔离。
        ns = namespace_id if namespace_id is not None else self.namespace_id
        if ns is not None:
            records = [row for row in records if row.get("namespace_id") == ns]
        # 判断该台账是否被多个写工具共用。
        shared_ledger = sum(1 for item in WRITE_TOOL_FACTS.values() if item["ledger"] == mapping["ledger"]) > 1
        if shared_ledger:
            # 共用台账：按"本工具写的 tool 名"或"本工具的 fact=True"筛出真正属于本工具的记录。
            fact = mapping["fact"]
            records = [
                row
                for row in records
                if row.get("tool") == tool_name or row.get(fact) is True
            ]
        return deepcopy(records)

    def executed_write_tools(self, namespace_id: str | None = None) -> set[str]:
        """返回本命名空间内"确实执行过"的写工具集合。

        verifier 据此判断模型实际落地了哪些写动作（如是否真发起了退款、真关了单）。
        先建 ledger -> 工具列表 的反查表以识别独占/共用台账；再逐个写工具看其在本 namespace
        是否有记录，且记录确属本工具（fact=True / tool 名匹配 / 独占台账无歧义）即记为已执行。
        与 records_for_tool 同源的判定逻辑，保证共用台账下不把 cancel 误判成 modify。
        """
        executed = set()
        # ledger -> 写该台账的工具名列表，用于判断台账是独占还是共用。
        ledger_to_tools: dict[str, list[str]] = {}
        for tool_name, mapping in WRITE_TOOL_FACTS.items():
            ledger_to_tools.setdefault(mapping["ledger"], []).append(tool_name)
        for tool_name, mapping in WRITE_TOOL_FACTS.items():
            fact = mapping["fact"]
            unique_ledger = len(ledger_to_tools[mapping["ledger"]]) == 1  # 该台账是否仅本工具独占
            for row in self.records_for_tool(tool_name, namespace_id):
                # 命中任一条件即认定本工具执行过：本工具 fact=True / tool 名匹配 / 独占台账（必属本工具）。
                if row.get(fact) is True or row.get("tool") == tool_name or unique_ledger:
                    executed.add(tool_name)
                    break
        return executed
