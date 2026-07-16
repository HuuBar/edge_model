"""工具名映射表：旧数据工具名 → 新代码工具名。

旧数据来自原始JSON数据集（8821条），使用了一套较早的工具命名。
新代码（envs/toollist/）使用规范化的WiFi工具命名（如 wifi.get_info）。

映射规则：
- 尽量保持语义一致
- 一个旧名可能映射到多个新名（按上下文选择）
- 旧数据中不存在的工具保留为None（需人工确认）
"""

from __future__ import annotations

# 主映射表：旧工具名 → 新工具名
# 旧名来自原始数据的 react_label.tool_name 和 categories 字段
# 新名来自 envs/toolfactory.py 的 TOOL_MODULES
TOOL_NAME_MAP: dict[str, str | None] = {
    # ── 读工具（从原始数据 → 新代码） ──
    "get_traffic_statistics": "data.get_usage",           # 查流量统计
    "get_wifi_info": "wifi.get_info",                     # 查WiFi配置
    "get_wifi_config": "wifi.get_info",                   # 同get_wifi_info
    "list_wifi_clients": "wifi.list_clients",             # 查客户端列表
    "get_client_list": "wifi.list_clients",               # 同list_wifi_clients
    "get_device_info": "device.get_info",                 # 查设备信息
    "get_serial_number": "device.get_info",               # 序列号在device_info中
    "get_antenna_type": "device.get_info",                # 天线类型在device_info中
    "get_network_status": "network.get_status",           # 查网络状态
    "get_ethernet_speed": "network.get_status",           # 网速在网络状态中
    "get_frequency_band": "network.get_status",           # 频段在网络状态中
    "get_network_settings": "network.get_settings",       # 查网络设置
    "get_dns_info": "network.get_settings",               # DNS在网络设置中
    "get_module_switch": "network.get_settings",          # 模块开关在网络设置中
    "get_system_logs": "system.get_logs",                 # 查系统日志
    "search_policy": "policy.search",                     # 查客服策略
    "get_policy": "policy.search",                        # 同search_policy
    "get_wifi_diagnosis_info": "wifi.get_info",           # WiFi诊断信息 → get_info

    # ── 写工具（从原始数据 → 新代码） ──
    "set_data_limit": "data.set_limit",                   # 设流量上限
    "set_data_alert_threshold": "data.set_alert_threshold",  # 设流量告警阈值
    "set_wifi_name": "wifi.set_config",                   # 设WiFi名称 → set_config
    "set_wifi_ssid": "wifi.set_config",                   # 同set_wifi_name
    "switch_wifi_broadcast": "wifi.hide_ssid",            # 开关广播 → hide_ssid
    "hide_ssid": "wifi.hide_ssid",                        # 同switch_wifi_broadcast
    "set_wifi_channel": "wifi.set_channel",               # 设信道
    "set_wifi_bandwidth": "wifi.set_bandwidth",           # 设带宽
    "switch_5G_mode": "wifi.switch_5g_mode",              # 切5G模式
    "switch_5G_priority": "wifi.switch_5g_priority",       # 切5G优选
    "set_network_ip_mode": "network.set_ip_mode",         # 设IP模式
    "set_network_ip_pool": "network.set_ip_pool",         # 设IP地址池
    "restart_device": "device.restart",                   # 重启设备
    "change_password": "user.change_password",            # 改密码
    "change_user_password": "user.change_password",       # 同change_password
    "wifi.open": "wifi.open",                             # 开WiFi（已对齐）
    "open_wifi": "wifi.open",                             # 同wifi.open
    "enable_wifi": "wifi.open",                           # 同wifi.open
    "wifi.close": "wifi.close",                           # 关WiFi（已对齐）
    "close_wifi": "wifi.close",                           # 同wifi.close
    "disable_wifi": "wifi.close",                         # 同wifi.close

    # ── 特殊映射（需要根据参数判断） ──
    # "switch_wifi_enable" → 根据 action=ON/OFF 判断为 wifi.open / wifi.close
    # 这个不在TOOL_NAME_MAP中，由 map_tool_name_with_args() 特殊处理

    # ── 以下工具在新代码中未定义，映射为None（需人工确认） ──
    "switch_firewall": None,                              # 防火墙开关
    "switch_game_turbo": None,                            # 游戏加速
    "switch_intelligent_func": None,                      # 智能覆盖
    "switch_data_mode": None,                             # 数据模式开关
    "get_battery_status": None,                           # 电池状态
    "get_temperature": None,                              # 温度
    "set_apn": None,                                      # 设置APN
}


def map_tool_name_with_args(old_name: str, args: dict | None = None) -> str | None:
    """带参数的工具名映射（处理需要根据参数判断的情况）。

    特殊情况:
      - "switch_wifi_enable" + {"action": "ON"}  → "wifi.open"
      - "switch_wifi_enable" + {"action": "OFF"} → "wifi.close"
      - "switch_wifi_enable" + 无args             → None（无法判断）
    """
    if not old_name:
        return None

    # 特殊处理：switch_wifi_enable 需要根据 action 参数判断
    if old_name in ("switch_wifi_enable",):
        if args:
            action = str(args.get("action", "")).upper()
            if action == "ON":
                return "wifi.open"
            elif action == "OFF":
                return "wifi.close"
        # 无法判断时返回None
        return None

    # 普通映射
    return map_tool_name(old_name)


def map_tool_name(old_name: str) -> str | None:
    """把旧工具名映射为新工具名。

    参数
    ----
    old_name: 原始数据中的工具名（如 "get_traffic_statistics"）

    返回
    ----
    新工具名（如 "data.get_usage"），如果无法映射则返回 None
    """
    if not old_name:
        return None
    # 精确匹配
    if old_name in TOOL_NAME_MAP:
        return TOOL_NAME_MAP[old_name]
    # 尝试去掉空格/下划线变体
    normalized = old_name.strip().lower().replace(" ", "_")
    if normalized in TOOL_NAME_MAP:
        return TOOL_NAME_MAP[normalized]
    # 未找到映射
    return None


def map_trajectory_tools(trajectory: list[dict]) -> list[dict]:
    """把一条trajectory中所有旧工具名替换为新工具名（原地修改）。

    参数
    ----
    trajectory: react_label_en 格式的trajectory列表

    返回
    ----
    工具名已替换的trajectory；无法映射的step保留原值但标记warning
    """
    result = []
    for step in trajectory:
        new_step = dict(step)
        content = new_step.get("content", {})
        if isinstance(content, dict):
            old_tool = content.get("tool_name", "")
            new_tool = map_tool_name(old_tool)
            if new_tool:
                content["tool_name"] = new_tool
            elif old_tool and old_tool not in ("", "sandbox.write_record"):
                # 保留原值但加标记，供后续排查
                content["_tool_name_unmapped"] = old_tool
        result.append(new_step)
    return result


def map_categories(categories: list[str]) -> list[str]:
    """把categories列表中的旧工具名全部替换为新工具名。

    无法映射的category跳过（不保留None）
    """
    mapped = []
    for cat in categories:
        new_cat = map_tool_name(cat)
        if new_cat:
            mapped.append(new_cat)
        # 无法映射的跳过（避免训练数据中出现无效工具名）
    return mapped


# 反向映射：新工具名 → 可能的旧工具名列表（用于调试/排查）
REVERSE_MAP: dict[str, list[str]] = {}
for old, new in TOOL_NAME_MAP.items():
    if new:
        REVERSE_MAP.setdefault(new, []).append(old)


def get_reverse_map(new_name: str) -> list[str]:
    """查询某个新工具名对应的所有旧名（调试用）。"""
    return REVERSE_MAP.get(new_name, [])


# 统计信息
TOTAL_MAPPED = sum(1 for v in TOOL_NAME_MAP.values() if v is not None)
TOTAL_UNMAPPED = sum(1 for v in TOOL_NAME_MAP.values() if v is None)
TOTAL_ENTRIES = len(TOOL_NAME_MAP)

if __name__ == "__main__":
    print(f"工具名映射表统计:")
    print(f"  总条目: {TOTAL_ENTRIES}")
    print(f"  已映射: {TOTAL_MAPPED}")
    print(f"  未映射(需人工确认): {TOTAL_UNMAPPED}")
    print(f"\n已映射工具列表:")
    for old, new in sorted(TOOL_NAME_MAP.items()):
        if new:
            print(f"  {old:35s} → {new}")
    if TOTAL_UNMAPPED > 0:
        print(f"\n未映射工具列表(需人工确认):")
        for old, new in sorted(TOOL_NAME_MAP.items()):
            if new is None:
                print(f"  {old:35s} → [未定义]")
