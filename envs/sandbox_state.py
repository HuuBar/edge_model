from __future__ import annotations

# 版权所有 © 2026 深圳途明智启科技有限公司。保留所有权利。
# 未经书面许可，任何单位或个人不得复制、传播、发布、转卖、改编、仿制或用于商业用途。
# 侵权必究。

"""按命名空间隔离的 sandbox 可写状态。

sandbox 是"模型可改的世界"——写工具（开WiFi、改配置、设流量限制……）把记录写进这里的台账（ledger）。
只读设备表在 env_snapshot 里不可变，而所有副作用都落在 SandboxState；跑完后 export() 出来的
就是 环境数据规范 说的 sandbox_final_state，是 verifier 判 require/claim 的事实地基。

两条贯穿全模块的设计：
1. namespace 隔离：每条 rollout 的写都打上 namespace_id，读取时按它过滤。verl 并发跑同一 case 的
   多条 rollout 时，这保证 A 的改配置不会被 B 读到。
2. 台账形状二分：多数台账是"追加列表"（每次写 append 一条）；少数表示"某对象的当前状态"
   （用 dict 按对象 key 存），后写覆盖先写。
   
   WiFi 场景下所有 5 个台账都是"事件流"语义——每次操作产生一条记录，不存在"覆盖最新状态"的需求。
   因此 DICT_LEDGERS 为空集合。
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from schemas.env_schema import default_sandbox


# ============================================================================
# 写工具 -> {落哪个台账 ledger, 提供哪个 verified_fact_key} 的映射表。
# ============================================================================
# WiFi 客服场景共 13 个写工具，映射到 5 个台账：
#
#   wifi_config_log:    wifi.set_config, wifi.set_channel, wifi.set_bandwidth, wifi.hide_ssid
#   switch_log:         wifi.open, wifi.close, wifi.switch_5g_mode
#   data_limit_log:     data.set_limit, data.set_alert_threshold
#   ip_config_log:      network.set_ip_mode, network.set_ip_pool
#   operation_log:      device.restart, user.change_password
# ============================================================================
WRITE_TOOL_FACTORS: dict[str, dict[str, str]] = {
    # --- wifi_config_log（WiFi 参数配置类）---
    "wifi.set_config": {
        "ledger": "wifi_config_log",
        "fact": "wifi_config_set",
    },
    "wifi.set_channel": {
        "ledger": "wifi_config_log",
        "fact": "wifi_channel_set",
    },
    "wifi.set_bandwidth": {
        "ledger": "wifi_config_log",
        "fact": "wifi_bandwidth_set",
    },
    "wifi.hide_ssid": {
        "ledger": "wifi_config_log",
        "fact": "wifi_ssid_hidden",
    },
    # --- switch_log（开关与模式切换类）---
    "wifi.open": {
        "ledger": "switch_log",
        "fact": "wifi_opened",
    },
    "wifi.close": {
        "ledger": "switch_log",
        "fact": "wifi_closed",
    },
    "wifi.switch_5g_mode": {
        "ledger": "switch_log",
        "fact": "cellular_5g_mode_switched",
    },
    "wifi.switch_5g_priority": {
        "ledger": "switch_log",
        "fact": "wifi_5g_priority_switched",
    },
    # --- data_limit_log（流量限制类）---
    "data.set_limit": {
        "ledger": "data_limit_log",
        "fact": "data_limit_set",
    },
    "data.set_alert_threshold": {
        "ledger": "data_limit_log",
        "fact": "data_alert_threshold_set",
    },
    # --- ip_config_log（IP 配置类）---
    "network.set_ip_mode": {
        "ledger": "ip_config_log",
        "fact": "ip_mode_set",
    },
    "network.set_ip_pool": {
        "ledger": "ip_config_log",
        "fact": "ip_pool_set",
    },
    # --- operation_log（通用运维操作类）---
    "device.restart": {
        "ledger": "operation_log",
        "fact": "device_restarted",
    },
    "user.change_password": {
        "ledger": "operation_log",
        "fact": "password_changed",
    },
}

# "状态型"台账：用 dict（按对象 key 存）而非 list。
# WiFi 场景下所有操作都是"记录一次操作发生"的事件流语义，
# 不存在"某个对象的当前状态需要覆盖"的场景，因此 DICT_LEDGERS 为空集合。
DICT_LEDGERS: set[str] = set()


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

        sandbox_initial 是 case 预置的台账初值（一般为空列表）；
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
          1. 据 WRITE_TOOL_FACTORS 查到该写工具对应的 ledger 与 fact（无映射的工具直接报错）。
          2. 把业务 record 补全为 enriched：附上 tool 名、namespace/run/case/rollout/tool_call 等溯源字段，
             并置 <fact>=True 与 verified_fact_key=fact —— 这两项是 verifier 直接读的"该动作已发生"信号。
          3. 按台账形状落库：DICT_LEDGERS 按对象 key 覆盖式写；其余 append（事件流）。
             WiFi 场景下 DICT_LEDGERS 为空，全部走 append 分支。
          4. 始终向 operation_log 追加一条审计（若该工具映射到 operation_log 则同台账审计，
             否则向 operation_log 单独写审计记录），记录这次写的来龙去脉。
        所有写都带 namespace_id，确保并发 rollout 之间互不污染。
        """
        if tool_name not in WRITE_TOOL_FACTORS:
            # 非写工具/未登记映射的工具不应走到这里，属调用方逻辑错误。
            raise KeyError(f"tool has no sandbox write mapping: {tool_name}")
        # mapping 是写工具到台账/事实键的唯一来源，verifier 和写工具都共享这套口径。
        mapping = WRITE_TOOL_FACTORS[tool_name]
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
            # WiFi 场景下此分支不会命中（DICT_LEDGERS 为空），保留以备未来扩展。
            key = object_key or record.get("device_id") or record.get("config_id")
            key = key or tool_call_id
            bucket = self.state.setdefault(ledger, {})
            bucket[key] = enriched
        else:
            # 事件流台账：直接 append，保留每次写的完整历史。
            # WiFi 场景下全部 5 个台账都走此分支。
            self.state.setdefault(ledger, []).append(enriched)
        # 审计日志：每次写都留痕。优先写入该工具自身的台账作为审计，否则写入 operation_log。
        # action 缺省取工具名最后一段（如 wifi.set_config -> set_config）；
        # result 缺省取记录的 status，否则记 "ok"。审计仅供追溯/调试，不作 outcome 真值。
        audit_entry = {
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
        # 向操作工具自身的 operation_log 追加审计（如果是操作类工具），
        # 同时也始终向 operation_log 写一条全局审计记录
        self.state.setdefault("operation_log", []).append(audit_entry)
        return deepcopy(enriched)

    def records_for_tool(self, tool_name: str, namespace_id: str | None = None) -> list[dict[str, Any]]:
        """取某写工具在指定命名空间下写出的全部记录（已隔离过滤）。

        三层过滤：
          1. 取该工具对应的台账（dict 台账取 values，list 台账原样）。
             WiFi 场景下全部是 list 台账。
          2. 按 namespace_id 过滤，只留本 rollout 的记录（默认用 self.namespace_id），保证并发隔离。
          3. 若该台账被多个工具共用，再按 fact 字段 / tool 名细分，只留属于本工具的那条 fact。
             独占台账则跳过第 3 步。
        """
        mapping = WRITE_TOOL_FACTORS.get(tool_name)
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
        shared_ledger = sum(1 for item in WRITE_TOOL_FACTORS.values() if item["ledger"] == mapping["ledger"]) > 1
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

        verifier 据此判断模型实际落地了哪些写动作（如是否真改了 WiFi 配置、真重启了设备）。
        先建 ledger -> 工具列表 的反查表以识别独占/共用台账；再逐个写工具看其在本 namespace
        是否有记录，且记录确属本工具（fact=True / tool 名匹配 / 独占台账无歧义）即记为已执行。
        与 records_for_tool 同源的判定逻辑，保证共用台账下不把 set_config 误判成 set_channel。
        """
        executed = set()
        # ledger -> 写该台账的工具名列表，用于判断台账是独占还是共用。
        ledger_to_tools: dict[str, list[str]] = {}
        for tool_name, mapping in WRITE_TOOL_FACTORS.items():
            ledger_to_tools.setdefault(mapping["ledger"], []).append(tool_name)
        for tool_name, mapping in WRITE_TOOL_FACTORS.items():
            fact = mapping["fact"]
            unique_ledger = len(ledger_to_tools[mapping["ledger"]]) == 1  # 该台账是否仅本工具独占
            for row in self.records_for_tool(tool_name, namespace_id):
                # 命中任一条件即认定本工具执行过：本工具 fact=True / tool 名匹配 / 独占台账（必属本工具）。
                if row.get(fact) is True or row.get("tool") == tool_name or unique_ledger:
                    executed.add(tool_name)
                    break
        return executed
