"""WiFi 客服环境工具的共享实现。

三层调用结构（toolfactory → common → sandbox）：
- 上层 toolfactory：每个工具有一个薄包装文件（如 wifi_get_info.py、wifi_set_config.py），
  里面只有一句 ``TOOL = make_tool("xxx")``。
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


# ============================================================================
# 辅助函数
# ============================================================================

def arg(type_: str, required: bool = False, description: str = "") -> ToolArg:
    """构造一个参数 schema 项（ToolArg），供 TOOL_SPECS 里声明每个 arg 的类型/是否必填。"""
    return ToolArg(type=type_, required=required, description=description)


def _tables(env_snapshot: dict[str, Any]) -> dict[str, Any]:
    """取整个只读 数据表集合（readonly_tables），所有读工具的数据来源。"""
    return env_snapshot.get("readonly_tables", {})


def _table(env_snapshot: dict[str, Any], name: str) -> dict[str, Any]:
    """按名字取单张只读表（如 device_info、wifi_config）；缺失或类型不对时回退成空 dict 而非报错。"""
    value = _tables(env_snapshot).get(name, {})
    return value if isinstance(value, dict) else {}


def _first_by(items: dict[str, Any], key: str, value: Any) -> dict[str, Any] | None:
    """在以 id 为键的表里按某字段值线性查第一条匹配行。"""
    for row in items.values():
        if isinstance(row, dict) and row.get(key) == value:
            return row
    return None


def _copy(row: Any) -> Any:
    """深拷贝返回值，避免把只读课程数据 的内部对象直接暴露给调用方被改写。"""
    return deepcopy(row)


def _policy_rules(policy: dict[str, Any] | None) -> dict[str, Any]:
    """取政策的结构化规则块；写工具据此判断操作是否允许。

    优先读 decision_rule；rules 仅作旧字段回退。
    """
    if not policy:
        return {}
    rules = policy.get("decision_rule") or policy.get("rules", {})
    return rules if isinstance(rules, dict) else {}


def _find_policy(env_snapshot: dict[str, Any], policy_id: str | None) -> dict[str, Any] | None:
    """按 policy_id 精确取政策；写工具携带的 policy_id 在这里反查。"""
    if not policy_id:
        return None
    for policy in env_snapshot.get("policies", []):
        if policy.get("policy_id") == policy_id:
            return policy
    return None


def _policy_matches(policy: dict[str, Any], args: dict[str, Any]) -> bool:
    """policy.search 的匹配逻辑：按 topic/设备型号/操作类型逐项过滤，任一不符即排除。"""
    # topic 匹配：政策类目必须一致
    topic = args.get("topic")
    if topic is not None and topic != policy.get("topic"):
        return False
    # 设备型号匹配（如果指定）
    device_model = args.get("device_model")
    if device_model and policy.get("device_model"):
        if device_model != policy["device_model"]:
            return False
    # 操作类型匹配（如果指定）
    action_type = args.get("action_type")
    if action_type and policy.get("action_type"):
        if action_type != policy["action_type"]:
            return False
    # 金额上限匹配（如果政策有金额限制）
    match = policy.get("match", {}) or {}
    amount = args.get("amount")
    if amount is not None and match.get("amount_max") is not None and amount > match["amount_max"]:
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
    device_id: str | None = None,
    object_key: str | None = None,
    audit_action: str | None = None,
    audit_result: str | None = None,
) -> dict[str, Any]:
    """统一的写台账入口：所有写工具都经此把 record append 进对应 sandbox 台账并写审计日志。

    tool_name 决定写哪张台账；device_id 用于write_consistency_caps校验（确保写操作针对正确设备）；
    object_key 用于按对象去重/定位；audit_action/audit_result 落进 operation_log 供留痕。
    """
    if device_id is not None:
        record["device_id"] = device_id
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
    """生成确定性的对象 id（如 CFG_2.4G_abc123）。

    带上 tool_call_id 后缀保证同一次调用幂等、可 replay，且不同调用不会撞 id。
    """
    clean = "_".join(str(p) for p in parts if p is not None)
    suffix = context.get("tool_call_id", "tc")
    return f"{prefix}_{clean}_{suffix}" if clean else f"{prefix}_{suffix}"


# ============================================================================
# _read_handler：读工具总分发（9个读分支）
# ============================================================================

def _read_handler(tool_name: str, args: dict[str, Any], env: dict[str, Any], sandbox: SandboxState) -> dict[str, Any]:
    """读工具总分发：按 tool_name 命中分支，读对应只读表，不写任何台账、无副作用。"""

    # --- 1. wifi.get_info：读取 WiFi 当前配置 ---
    if tool_name == "wifi.get_info":
        config = _table(env, "wifi_config")
        if not config:
            raise ToolExecutionError("wifi_config_not_found")
        return _copy(config)

    # --- 2. wifi.list_clients：读取已连接客户端列表 ---
    if tool_name == "wifi.list_clients":
        clients = _table(env, "connected_clients")
        # 支持按条件过滤
        result = {"clients": []}
        for client_id, client in clients.items():
            if not isinstance(client, dict):
                continue
            # MAC 地址过滤
            if args.get("mac_address") and client.get("mac") != args["mac_address"]:
                continue
            # IP 过滤
            if args.get("ip_address") and client.get("ip") != args["ip_address"]:
                continue
            # 频段过滤（2.4G/5G）
            if args.get("band") and client.get("band") != args["band"]:
                continue
            item = _copy(client)
            item.setdefault("client_id", client_id)
            result["clients"].append(item)
        result["total_count"] = len(result["clients"])
        return result

    # --- 3. device.get_info：读取设备基本信息 ---
    if tool_name == "device.get_info":
        info = _table(env, "device_info")
        if not info:
            raise ToolExecutionError("device_info_not_found")
        return _copy(info)

    # --- 4. data.get_usage：读取流量使用统计 ---
    if tool_name == "data.get_usage":
        usage = _table(env, "data_usage")
        if not usage:
            raise ToolExecutionError("data_usage_not_found")
        # 支持按 client_id 过滤特定客户端的流量
        if args.get("client_id"):
            clients = usage.get("clients", {})
            client_data = clients.get(args["client_id"])
            if not client_data:
                raise ToolExecutionError("client_not_found")
            return {
                "client_id": args["client_id"],
                **_copy(client_data),
            }
        return _copy(usage)

    # --- 5. network.get_status：读取网络实时状态 ---
    if tool_name == "network.get_status":
        status = _table(env, "network_status")
        if not status:
            raise ToolExecutionError("network_status_not_found")
        return _copy(status)

    # --- 6. network.get_settings：读取网络高级设置 ---
    if tool_name == "network.get_settings":
        settings = _table(env, "network_settings")
        if not settings:
            raise ToolExecutionError("network_settings_not_found")
        return _copy(settings)

    # --- 7. system.get_logs：读取系统日志 ---
    if tool_name == "system.get_logs":
        logs = _table(env, "system_logs")
        if not logs:
            raise ToolExecutionError("system_logs_not_found")
        # 支持按 log_type 过滤（error/warning/info）
        log_type = args.get("log_type")
        entries = logs.get("entries", [])
        if log_type:
            entries = [e for e in entries if e.get("type") == log_type]
        # 支持按 limit 限制条数
        limit = args.get("limit", 50)
        if limit and isinstance(limit, int):
            entries = entries[:limit]
        return {
            "entries": _copy(entries),
            "total_count": len(entries),
        }

    # --- 8. policy.search：检索客服策略/规则 ---
    if tool_name == "policy.search":
        if not args.get("topic"):
            raise ToolExecutionError("topic_required")
        policy = _search_policy(env, args)
        if not policy:
            raise ToolExecutionError("policy_not_found")
        return _copy(policy)

    # 未在上面任何分支命中：该 tool_name 不是已实现的读工具。
    raise ToolExecutionError("tool_not_implemented", source="runtime")



# ============================================================================
# _write_handler：写工具总分发（13个写分支）
# ============================================================================

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

    WiFi 场景下所有写操作都是"事件流"语义，每次操作产生一条独立记录。
    """
    # 统一提取device_id，传给每个_write调用（用于verifier的write_consistency_caps校验）
    device_info = _table(env, "device_info")
    device_id = device_info.get("device_id") if device_info else None

    # ========================================================================
    # 组 A：wifi_config_log（WiFi 参数配置类，4 个工具）
    # ========================================================================

    # --- A1. wifi.set_config：设置 WiFi SSID/密码/加密方式 ---
    if tool_name == "wifi.set_config":
        # 前置校验：SSID 不能为空
        ssid = args.get("ssid", "")
        if not ssid or not isinstance(ssid, str):
            raise ToolExecutionError("ssid_required")
        # SSID 长度校验（1-32 字节）
        if len(ssid.encode("utf-8")) > 32:
            raise ToolExecutionError("ssid_too_long")
        # 密码长度校验（8-63 字符，若提供）
        password = args.get("password", "")
        if password and len(password) < 8:
            raise ToolExecutionError("password_too_short")
        record = {
            "config_id": _id("CFG", context, ssid),
            "ssid": ssid,
            "password": password if password else None,
            "encryption": args.get("encryption", "WPA2-PSK"),
            "band": args.get("band", "dual"),
            "status": "applied",
        }
        _write(sandbox, tool_name, record, context, device_id=device_id, audit_action="set_config", audit_result="applied")
        return {
            "config_id": record["config_id"],
            "ssid": ssid,
            "encryption": record["encryption"],
            "band": record["band"],
            "status": "applied",
        }

    # --- A2. wifi.set_channel：设置 WiFi 信道 ---
    if tool_name == "wifi.set_channel":
        channel = args.get("channel")
        if channel is None:
            raise ToolExecutionError("channel_required")
        # 信道范围校验（2.4G: 1-14, 5G: 36-165）
        band = args.get("band", "2.4G")
        valid_2g = list(range(1, 15))  # 1-14
        valid_5g = [36, 40, 44, 48, 52, 56, 60, 64, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140, 144, 149, 153, 157, 161, 165]
        if band == "2.4G" and channel not in valid_2g:
            raise ToolExecutionError("invalid_channel_for_2.4g")
        if band == "5G" and channel not in valid_5g:
            raise ToolExecutionError("invalid_channel_for_5g")
        record = {
            "channel_id": _id("CH", context, band, channel),
            "band": band,
            "channel": channel,
            "status": "applied",
        }
        _write(sandbox, tool_name, record, context, device_id=device_id, audit_action="set_channel", audit_result="applied")
        return {
            "channel_id": record["channel_id"],
            "band": band,
            "channel": channel,
            "status": "applied",
        }

    # --- A3. wifi.set_bandwidth：设置 WiFi 频带宽度 ---
    if tool_name == "wifi.set_bandwidth":
        bandwidth = args.get("bandwidth")
        if not bandwidth:
            raise ToolExecutionError("bandwidth_required")
        # 支持的带宽值
        valid_bandwidths = {"20MHz", "40MHz", "80MHz", "160MHz", "20/40MHz"}
        if bandwidth not in valid_bandwidths:
            raise ToolExecutionError("invalid_bandwidth")
        band = args.get("band", "2.4G")
        record = {
            "bw_id": _id("BW", context, band, bandwidth),
            "band": band,
            "bandwidth": bandwidth,
            "status": "applied",
        }
        _write(sandbox, tool_name, record, context, device_id=device_id, audit_action="set_bandwidth", audit_result="applied")
        return {
            "bw_id": record["bw_id"],
            "band": band,
            "bandwidth": bandwidth,
            "status": "applied",
        }

    # --- A4. wifi.hide_ssid：设置 SSID 隐藏/可见 ---
    if tool_name == "wifi.hide_ssid":
        hidden = args.get("hidden")
        if hidden is None:
            raise ToolExecutionError("hidden_required")
        if not isinstance(hidden, bool):
            raise ToolExecutionError("hidden_must_be_boolean")
        ssid = args.get("ssid", "")
        record = {
            "hide_id": _id("HIDE", context, ssid or "default"),
            "ssid": ssid,
            "hidden": hidden,
            "status": "applied",
        }
        _write(sandbox, tool_name, record, context, device_id=device_id, audit_action="hide_ssid", audit_result="applied")
        return {
            "hide_id": record["hide_id"],
            "hidden": hidden,
            "status": "applied",
        }

    # ========================================================================
    # 组 B：switch_log（开关与模式切换类，3 个工具）
    # ========================================================================

    # --- B1. wifi.open：开启 WiFi ---
    if tool_name == "wifi.open":
        # 产品决策 D2：open_wifi 只开不关（安全考量）
        # 前置校验：读取当前 WiFi 状态
        wifi_config = _table(env, "wifi_config")
        if wifi_config and wifi_config.get("enabled") is True:
            raise ToolExecutionError("wifi_already_open")
        band = args.get("band", "all")
        record = {
            "switch_id": _id("ON", context, band),
            "action": "open",
            "band": band,
            "status": "enabled",
        }
        _write(sandbox, tool_name, record, context, device_id=device_id, audit_action="open_wifi", audit_result="enabled")
        return {
            "switch_id": record["switch_id"],
            "action": "open",
            "band": band,
            "status": "enabled",
        }

    # --- B2. wifi.close：关闭 WiFi ---
    if tool_name == "wifi.close":
        # 前置校验：读取当前 WiFi 状态
        wifi_config = _table(env, "wifi_config")
        if wifi_config and wifi_config.get("enabled") is False:
            raise ToolExecutionError("wifi_already_closed")
        band = args.get("band", "all")
        # 产品决策 D2：open_wifi 只开不关——但保留 close 工具供管理员级 case 使用
        # close 需要额外确认（通过 policy_id 证明已查政策）
        record = {
            "switch_id": _id("OFF", context, band),
            "action": "close",
            "band": band,
            "status": "disabled",
        }
        _write(sandbox, tool_name, record, context, device_id=device_id, audit_action="close_wifi", audit_result="disabled")
        return {
            "switch_id": record["switch_id"],
            "action": "close",
            "band": band,
            "status": "disabled",
        }

    # --- B3. wifi.switch_5g_mode：切换蜂窝网络 5G 注册模式 ---
    if tool_name == "wifi.switch_5g_mode":
        mode = args.get("mode")
        if not mode:
            raise ToolExecutionError("mode_required")
        # 蜂窝网络 5G 注册模式：sa_nsa=同时支持SA和NSA, 4g_only=仅4G, auto=自动
        valid_modes = {"sa_nsa", "4g_only", "auto"}
        if mode not in valid_modes:
            raise ToolExecutionError("invalid_mode")
        record = {
            "switch_id": _id("5G", context, mode),
            "cellular_5g_mode": mode,
            "status": "applied",
        }
        _write(sandbox, tool_name, record, context, device_id=device_id, audit_action="switch_5g_mode", audit_result="applied")
        return {
            "switch_id": record["switch_id"],
            "cellular_5g_mode": mode,
            "status": "applied",
        }

    # --- B4. wifi.switch_5g_priority：开关 WiFi 5GHz 频段优选 ---
    if tool_name == "wifi.switch_5g_priority":
        enabled = args.get("enabled")
        if enabled is None:
            raise ToolExecutionError("enabled_required")
        if not isinstance(enabled, bool):
            raise ToolExecutionError("enabled_must_be_boolean")
        record = {
            "switch_id": _id("5GP", context, "on" if enabled else "off"),
            "wifi_5g_priority_enabled": enabled,
            "status": "enabled" if enabled else "disabled",
        }
        _write(sandbox, tool_name, record, context, device_id=device_id, audit_action="switch_5g_priority", audit_result=record["status"])
        return {
            "switch_id": record["switch_id"],
            "wifi_5g_priority_enabled": enabled,
            "status": record["status"],
        }

    # ========================================================================
    # 组 C：data_limit_log（流量限制类，2 个工具）
    # ========================================================================

    # --- C1. data.set_limit：设置流量上限 ---
    if tool_name == "data.set_limit":
        limit_mb = args.get("limit_mb")
        if limit_mb is None:
            raise ToolExecutionError("limit_mb_required")
        if not isinstance(limit_mb, (int, float)) or limit_mb < 0:
            raise ToolExecutionError("invalid_limit_value")
        # 转换为整数 MB
        limit_mb = int(limit_mb)
        target = args.get("target", "device")  # device / specific_client
        client_id = args.get("client_id") if target == "specific_client" else None
        record = {
            "limit_id": _id("DLIM", context, target, limit_mb),
            "target": target,
            "client_id": client_id,
            "limit_mb": limit_mb,
            "status": "applied",
        }
        _write(sandbox, tool_name, record, context, device_id=device_id, audit_action="set_limit", audit_result="applied")
        return {
            "limit_id": record["limit_id"],
            "target": target,
            "client_id": client_id,
            "limit_mb": limit_mb,
            "status": "applied",
        }

    # --- C2. data.set_alert_threshold：设置流量告警阈值 ---
    if tool_name == "data.set_alert_threshold":
        threshold_percent = args.get("threshold_percent")
        if threshold_percent is None:
            raise ToolExecutionError("threshold_percent_required")
        if not isinstance(threshold_percent, (int, float)) or threshold_percent < 1 or threshold_percent > 100:
            raise ToolExecutionError("invalid_threshold_value")
        threshold_percent = int(threshold_percent)
        record = {
            "alert_id": _id("ALERT", context, threshold_percent),
            "threshold_percent": threshold_percent,
            "status": "applied",
        }
        _write(sandbox, tool_name, record, context, device_id=device_id, audit_action="set_alert", audit_result="applied")
        return {
            "alert_id": record["alert_id"],
            "threshold_percent": threshold_percent,
            "status": "applied",
        }

    # ========================================================================
    # 组 D：ip_config_log（IP 配置类，2 个工具）
    # ========================================================================

    # --- D1. network.set_ip_mode：设置 IP 分配模式 ---
    if tool_name == "network.set_ip_mode":
        mode = args.get("mode")
        if not mode:
            raise ToolExecutionError("mode_required")
        valid_modes = {"dhcp", "static"}
        if mode not in valid_modes:
            raise ToolExecutionError("invalid_ip_mode")
        # 产品决策 D3：静态 IP 必须在设备局域网网段内（默认 192.168.8.x）
        # 注意：172.31.x.x 是设备内部保留网段，不可用于局域网
        static_ip = args.get("static_ip")
        if mode == "static" and static_ip:
            if not static_ip.startswith("192.168.8."):
                raise ToolExecutionError("static_ip_out_of_range")
        record = {
            "ipcfg_id": _id("IPM", context, mode),
            "mode": mode,
            "static_ip": static_ip,
            "status": "applied",
        }
        _write(sandbox, tool_name, record, context, device_id=device_id, audit_action="set_ip_mode", audit_result="applied")
        return {
            "ipcfg_id": record["ipcfg_id"],
            "mode": mode,
            "static_ip": static_ip,
            "status": "applied",
        }

    # --- D2. network.set_ip_pool：设置 DHCP 地址池范围 ---
    if tool_name == "network.set_ip_pool":
        start_ip = args.get("start_ip")
        end_ip = args.get("end_ip")
        if not start_ip or not end_ip:
            raise ToolExecutionError("start_ip_and_end_ip_required")
        # 产品决策 D3：地址池必须在设备局域网网段内（默认 192.168.8.x）
        # 注意：172.31.x.x 是设备内部保留网段，不可用于局域网
        if not start_ip.startswith("192.168.8.") or not end_ip.startswith("192.168.8."):
            raise ToolExecutionError("ip_pool_out_of_range")
        record = {
            "pool_id": _id("POOL", context, start_ip, end_ip),
            "start_ip": start_ip,
            "end_ip": end_ip,
            "status": "applied",
        }
        _write(sandbox, tool_name, record, context, device_id=device_id, audit_action="set_pool", audit_result="applied")
        return {
            "pool_id": record["pool_id"],
            "start_ip": start_ip,
            "end_ip": end_ip,
            "status": "applied",
        }

    # ========================================================================
    # 组 E：operation_log（运维操作类，2 个工具）
    # ========================================================================

    # --- E1. device.restart：重启设备 ---
    if tool_name == "device.restart":
        restart_type = args.get("restart_type", "soft")
        valid_types = {"soft", "hard"}
        if restart_type not in valid_types:
            raise ToolExecutionError("invalid_restart_type")
        record = {
            "op_id": _id("RST", context, restart_type),
            "action": "restart",
            "restart_type": restart_type,
            "status": "initiated",
        }
        _write(sandbox, tool_name, record, context, device_id=device_id, audit_action="restart", audit_result="initiated")
        return {
            "op_id": record["op_id"],
            "restart_type": restart_type,
            "status": "initiated",
            "eta_seconds": 60 if restart_type == "soft" else 120,
        }

    # --- E2. user.change_password：修改管理密码 ---
    if tool_name == "user.change_password":
        new_password = args.get("new_password")
        if not new_password:
            raise ToolExecutionError("new_password_required")
        # 密码强度校验
        if len(new_password) < 8:
            raise ToolExecutionError("password_too_short")
        confirm = args.get("confirm_password")
        if confirm and confirm != new_password:
            raise ToolExecutionError("passwords_do_not_match")
        record = {
            "op_id": _id("PWD", context),
            "action": "change_password",
            "password_changed": True,
            "status": "completed",
        }
        # 密码不记录明文，只记操作发生
        _write(sandbox, tool_name, record, context, device_id=device_id, audit_action="change_password", audit_result="completed")
        return {
            "op_id": record["op_id"],
            "password_changed": True,
            "status": "completed",
        }

    # 未命中任何写工具分支：该 tool_name 不是已实现的写工具。
    raise ToolExecutionError("tool_not_implemented", source="runtime")



# ============================================================================
# 所有工具的元数据登记表
# ============================================================================
# 包含：description（模型可见的中文说明）、permissions（决定读/写分流
# 及各类敏感/不可逆标记）、args（参数 schema）。make_tool 据此生成 ToolDefinition。
# 按业务域分组（WiFi 查询/设备查询/网络查询/流量查询/系统查询/政策查询 / WiFi 配置写 /
# 开关模式写 / 流量限制写 / IP 配置写 / 运维操作写）。
# ============================================================================

TOOL_SPECS: dict[str, dict[str, Any]] = {
    # =========================================================================
    # 读工具（8个）
    # =========================================================================

    # --- WiFi 查询 ---
    "wifi.get_info": {
        "description": "读取 WiFi 当前配置，包括 SSID、密码（通常不返回明文）、加密方式、信道、频段（2.4G/5G）、隐藏状态、带宽、最大连接数等。"
                      "当客户询问 WiFi 相关问题时，先调用此工具获取当前配置作为诊断基准。",
        "permissions": ("read",),
        "args": {},
    },
    "wifi.list_clients": {
        "description": "读取当前连接到 WiFi 的客户端列表，包括每个客户端的 MAC 地址、IP、设备名称、连接时长、实时流量、所属频段（2.4G/5G）。"
                      "支持按 MAC、IP 或频段过滤。当客户反映'网速慢'或'有人蹭网'时使用此工具排查。",
        "permissions": ("read",),
        "args": {
            "mac_address": arg("string", False, "客户端 MAC 地址，精确过滤"),
            "ip_address": arg("string", False, "客户端 IP 地址，精确过滤"),
            "band": arg("string", False, "频段过滤，取值为 2.4G 或 5G"),
        },
    },

    # --- 设备查询 ---
    "device.get_info": {
        "description": "读取设备基本信息，包括型号、固件版本、IMEI、运行时长、当前时间、存储使用情况等。"
                      "当需要了解设备硬件状态或确认固件是否需要升级时使用。",
        "permissions": ("read",),
        "args": {},
    },

    # --- 流量查询 ---
    "data.get_usage": {
        "description": "读取流量使用统计，包括总流量、本月/今日用量、各客户端流量明细、剩余额度、限速状态等。"
                      "当客户反映'流量超标'或'被限速'时使用此工具查询。支持按 client_id 查单个客户端。",
        "permissions": ("read",),
        "args": {
            "client_id": arg("string", False, "特定客户端 ID，查该客户端的流量明细"),
        },
    },

    # --- 网络查询 ---
    "network.get_status": {
        "description": "读取网络实时状态，包括连接状态、信号强度（RSRP/SINR）、上下行速率、延迟、丢包率、当前连接基站信息等。"
                      "当客户反映'连不上网'或'网速慢'时使用此工具做网络层诊断。",
        "permissions": ("read",),
        "args": {},
    },
    "network.get_settings": {
        "description": "读取网络高级设置，包括 MTU、IPv6 开关、UPnP、端口映射、防火墙规则、APN 配置、IP 分配模式、DHCP 地址池等。"
                      "当诊断网络连通性或 IP 相关问题时使用。",
        "permissions": ("read",),
        "args": {},
    },

    # --- 系统查询 ---
    "system.get_logs": {
        "description": "读取系统日志，返回近期事件、告警、错误等时间序列数据。"
                      "支持按日志类型（error/warning/info）过滤和限制返回条数。"
                      "当需要深入排查设备异常或查看操作历史时使用。",
        "permissions": ("read",),
        "args": {
            "log_type": arg("string", False, "日志类型过滤：error（错误）、warning（警告）、info（信息）"),
            "limit": arg("integer", False, "返回最大条数，默认 50"),
        },
    },

    # --- 政策查询 ---
    "policy.search": {
        "description": "在客服策略库中检索适用政策。返回的是**条件规则**（含操作权限、阈值、限制、例外），"
                      "不是直接结论——你要把规则套到本工单事实（设备型号、客户等级等）上自行推导该怎么做。"
                      "先按 topic（政策类目）检索；topic 由你**读客户消息 + 已查到的事实自行判断**。"
                      "执行写业务动作前必须先查到适用政策。",
        "permissions": ("read",),
        "args": {
            "topic": arg("string", True,
                        "政策类目——读客户消息+事实自己判断该工单属于哪一类，取下列之一："
                        "wifi_connect(WiFi连接问题) / wifi_speed(网速慢) / wifi_password(WiFi密码相关) / "
                        "wifi_switch(WiFi开关/模式切换) / data_limit(流量限制/超标) / data_usage(流量查询/统计) / "
                        "network_config(网络配置/IP/DHCP) / device_restart(设备重启) / device_reset(恢复出厂) / "
                        "client_manage(客户端管理/踢人) / firmware_upgrade(固件升级) / hardware_fault(硬件故障) / "
                        "password_change(修改管理密码) / blacklist(黑白名单) / port_forward(端口映射)"),
            "device_model": arg("string", False, "设备型号，取自 device.get_info"),
            "action_type": arg("string", False, "操作类型，如 restart/reset/config_change 等"),
            "amount": arg("number", False, "涉及金额（如流量套餐费用），用于命中按金额分档的政策"),
        },
    },

    # =========================================================================
    # 写工具（13个）
    # =========================================================================

    # --- WiFi 配置写（4个）---
    "wifi.set_config": {
        "description": "设置 WiFi 的 SSID、密码和加密方式。"
                      "执行前应先查询当前配置（wifi.get_info）确认状态，再用 policy.search 确认适用政策。"
                      "SSID 长度 1-32 字节，密码至少 8 位字符。修改后会影响所有已连接客户端。",
        "permissions": ("sandbox_write", "irreversible_action"),
        "args": {
            "ssid": arg("string", True, "新的 WiFi 名称（SSID），1-32 字节"),
            "password": arg("string", False, "新的 WiFi 密码，至少 8 位字符；不填则保持原密码"),
            "encryption": arg("string", False, "加密方式：WPA2-PSK（默认）、WPA3-SAE、WPA2/WPA3-Mixed"),
            "band": arg("string", False, "适用频段：dual（双频，默认）、2.4G、5G"),
        },
    },
    "wifi.set_channel": {
        "description": "设置 WiFi 的信道号。2.4G 频段使用 1-14，5G 频段使用 36/40/44/48/52/56/60/64 等。"
                      "当客户反映'网速慢'且诊断发现信道拥塞时使用此工具切换信道。"
                      "执行前先用 wifi.get_info 确认当前信道和频段，再用 policy.search 确认允许操作。",
        "permissions": ("sandbox_write",),
        "args": {
            "channel": arg("integer", True, "信道号：2.4G 用 1-14，5G 用 36/40/44/48/52/56/60/64/100/104...165"),
            "band": arg("string", False, "目标频段：2.4G（默认）或 5G"),
        },
    },
    "wifi.set_bandwidth": {
        "description": "设置 WiFi 频带宽度。20MHz 抗干扰强但速率低，40/80/160MHz 速率高但易干扰。"
                      "当客户需要优化速率或稳定性时使用。",
        "permissions": ("sandbox_write",),
        "args": {
            "bandwidth": arg("string", True, "带宽值：20MHz、40MHz、80MHz、160MHz、20/40MHz"),
            "band": arg("string", False, "目标频段：2.4G 或 5G（默认 2.4G）"),
        },
    },
    "wifi.hide_ssid": {
        "description": "设置是否隐藏 WiFi SSID（关闭广播）。隐藏后新设备需手动输入 SSID 才能连接。"
                      "产品安全考量：此工具只控制广播开关，不修改 SSID 本身。",
        "permissions": ("sandbox_write",),
        "args": {
            "hidden": arg("boolean", True, "true=隐藏 SSID，false=显示（广播）SSID"),
            "ssid": arg("string", False, "要操作的 SSID，不填则对默认 SSID 操作"),
        },
    },

    # --- 开关与模式切换写（3个）---
    "wifi.open": {
        "description": "开启 WiFi 广播。当客户反映'搜不到 WiFi'且诊断发现 WiFi 被关闭时使用。"
                      "产品决策：此工具只负责开启，不负责关闭（安全考量）。"
                      "执行前先用 wifi.get_info 确认 WiFi 当前为关闭状态。",
        "permissions": ("sandbox_write",),
        "args": {
            "band": arg("string", False, "要开启的频段：all（全部，默认）、2.4G、5G"),
        },
    },
    "wifi.close": {
        "description": "关闭 WiFi 广播。产品决策：此工具仅对管理员级别的 case 开放（需 policy.search 确认权限）。"
                      "普通客服 case 不应调用此工具。关闭后所有客户端将断开连接。",
        "permissions": ("sandbox_write", "irreversible_action"),
        "args": {
            "band": arg("string", False, "要关闭的频段：all（全部，默认）、2.4G、5G"),
            "reason": arg("string", True, "关闭原因说明"),
            "policy_id": arg("string", False, "适用政策 id（来自 policy.search），管理员级操作需带"),
        },
    },
    "wifi.switch_5g_mode": {
        "description": "切换蜂窝移动网络的 5G 注册模式：sa_nsa（同时支持 5G SA 和 NSA 组网）、"
                      "4g_only（仅使用 4G 网络）、auto（自动选择）。"
                      "当客户反映'无法注册到5G网络'或'5G速度比4G慢'时使用。"
                      "注意：这是控制蜂窝数据网络的 5G 模式，不是 WiFi 的 5GHz 频段。"
                      " WiFi 5GHz 优选请用 wifi.switch_5g_priority。",
        "permissions": ("sandbox_write",),
        "args": {
            "mode": arg("string", True, "目标模式：sa_nsa（5G SA+NSA）、4g_only（仅4G）、auto（自动）"),
        },
    },
    "wifi.switch_5g_priority": {
        "description": "开关 WiFi 的 5GHz 频段优选功能。开启后设备会优先连接 5GHz WiFi（速率快、干扰少），"
                      "关闭后设备优先连接 2.4GHz WiFi（覆盖广、穿墙好）。"
                      "当客户反映'WiFi 不稳定、经常掉线'或'网速慢'时使用。"
                      "注意：这是控制 WiFi 5GHz 频段的优选开关，不是蜂窝网络的 5G 模式。"
                      "蜂窝 5G 模式请用 wifi.switch_5g_mode。",
        "permissions": ("sandbox_write",),
        "args": {
            "enabled": arg("boolean", True, "true=开启 5GHz 优选，false=关闭 5GHz 优选"),
        },
    },

    # --- 流量限制写（2个）---
    "data.set_limit": {
        "description": "设置流量上限（MB）。达到上限后可选择断网或限速。"
                      "可对整台设备或特定客户端设置。当客户要求'控制流量使用'或'防止超额'时使用。",
        "permissions": ("sandbox_write",),
        "args": {
            "limit_mb": arg("number", True, "流量上限值（MB），0 表示取消限制"),
            "target": arg("string", False, "限制对象：device（整设备，默认）、specific_client（特定客户端）"),
            "client_id": arg("string", False, "当 target=specific_client 时必填，指定客户端 ID/MAC"),
        },
    },
    "data.set_alert_threshold": {
        "description": "设置流量告警阈值（百分比，1-100）。当使用量达到设定百分比时触发告警通知。"
                      "当客户希望在流量快用完时收到提醒时使用。",
        "permissions": ("sandbox_write",),
        "args": {
            "threshold_percent": arg("integer", True, "告警阈值百分比，范围 1-100"),
        },
    },

    # --- IP 配置写（2个）---
    "network.set_ip_mode": {
        "description": "设置 IP 地址分配模式：dhcp（自动获取，默认）或 static（静态分配）。"
                      "产品决策 D3：静态 IP 必须在设备局域网网段（默认 192.168.8.x）内；"
                      "172.31.x.x 是设备内部保留网段，不可用于局域网。"
                      "当客户需要固定 IP 或排查 IP 冲突时使用。",
        "permissions": ("sandbox_write", "irreversible_action"),
        "args": {
            "mode": arg("string", True, "IP 模式：dhcp（自动分配）或 static（静态）"),
            "static_ip": arg("string", False, "静态 IP 地址，必须在 192.168.8.x 段内；mode=dhcp 时不填"),
        },
    },
    "network.set_ip_pool": {
        "description": "设置 DHCP 地址池范围（起始 IP 和结束 IP）。"
                      "产品决策 D3：地址池必须在设备局域网网段（默认 192.168.8.x）；"
                      "172.31.x.x 是设备内部保留网段，不可用于局域网。"
                      "当客户反映'IP 冲突'或需要调整可用 IP 范围时使用。",
        "permissions": ("sandbox_write",),
        "args": {
            "start_ip": arg("string", True, "地址池起始 IP，必须在 192.168.8.x 段"),
            "end_ip": arg("string", True, "地址池结束 IP，必须在 192.168.8.x 段"),
        },
    },

    # --- 运维操作写（2个）---
    "device.restart": {
        "description": "重启设备。soft（软重启，保留配置，约 60 秒）或 hard（硬重启，类似断电重启，约 120 秒）。"
                      "当客户遇到无法通过配置解决的问题（如系统卡顿、内存泄漏、异常发热）时使用。"
                      "执行前需向客户确认重启将导致短暂断网，并建议保存正在进行的工作。",
        "permissions": ("sandbox_write", "irreversible_action"),
        "args": {
            "restart_type": arg("string", False, "重启类型：soft（软重启，默认）、hard（硬重启）"),
        },
    },
    "user.change_password": {
        "description": "修改设备管理密码（登录后台的密码，非 WiFi 密码）。"
                      "新密码至少 8 位字符。当客户反映'忘记管理密码'或'需要修改密码'时使用。"
                      "此操作不可逆，修改后旧密码立即失效。",
        "permissions": ("sandbox_write", "irreversible_action"),
        "args": {
            "new_password": arg("string", True, "新管理密码，至少 8 位字符"),
            "confirm_password": arg("string", False, "确认密码，若提供则校验两次输入一致"),
        },
    },
}


# ============================================================================
# execute_named_tool + make_tool：统一入口与工厂函数
# ============================================================================

def execute_named_tool(
    tool_name: str,
    args: dict[str, Any],
    env_snapshot: dict[str, Any],
    sandbox: SandboxState,
    context: dict[str, Any],
) -> dict[str, Any]:
    """读写分流的统一入口：按权限是否含 sandbox_write 决定走写分支还是读分支。

    工具 permissive（训练期）：不在此硬拦"该不该做"——那是模型该学的，
    交给 verifier 评分。工具只在各 handler 里拦"物理现实"（实体存在/守恒/必填/范围）。
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
        permissions=spec["permissions"],
        args=spec["args"],
        handler=handler,
    )