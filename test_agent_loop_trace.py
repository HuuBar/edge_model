#!/usr/bin/env python3
"""
Agent Runtime 逐步追踪测试

功能：
1. 初始化环境（展示5个空台账 + 9张只读表）
2. 手动模拟模型调用工具（绕过模型推理，直接注入tool_call）
3. 每步打印：messages变化、台账变化、工具执行结果
4. 最终导出完整sandbox状态

运行方式:
    cd /home/z50061485/edge_model
    python test_agent_loop_trace.py 2>&1 | tee /tmp/agent_loop_trace_$(date +%m%d_%H%M).log
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from envs.toolfactory import ToolFactory
from envs.sandbox_state import SandboxState, WRITE_TOOL_FACTORS
from envs.namespace import build_namespace_id
from schemas.env_schema import SANDBOX_KEYS


TEST_CASE = {
    "case_id": "TRACE_TEST_001",
    "customer_message": "我手机搜不到WiFi信号了，帮我看看",
    "entities": {"device_id": "DEV_TRACE_001"},
    "primary_intent": "wifi_not_found",
}

TEST_ENV = {
    "case_id": "TRACE_TEST_001",
    "reference_now": "2026-07-15T10:00:00",
    "readonly_tables": {
        "device_info": {
            "DEV_TRACE_001": {
                "device_id": "DEV_TRACE_001",
                "model": "HW-5G-CPE-Pro",
                "firmware_version": "V3.2.1",
                "imei": "860000011112222",
                "uptime_seconds": 86400,
            }
        },
        "wifi_config": {
            "DEV_TRACE_001": {
                "device_id": "DEV_TRACE_001",
                "ssid": "MyWiFi_5G",
                "password": "********",
                "encryption": "WPA2-PSK",
                "channel": 36,
                "band": "5G",
                "bandwidth": "80MHz",
                "hidden": False,
                "enabled": False,
                "max_clients": 32,
            }
        },
        "network_status": {
            "DEV_TRACE_001": {
                "device_id": "DEV_TRACE_001",
                "connected": True,
                "signal_strength": -75,
                "rsrp": -85,
                "sinr": 15,
                "download_speed_kbps": 51200,
                "upload_speed_kbps": 10240,
                "latency_ms": 35,
                "packet_loss_percent": 0.5,
                "network_type": "5G",
            }
        },
        "connected_clients": {},
        "data_usage": {
            "DEV_TRACE_001": {
                "device_id": "DEV_TRACE_001",
                "total_upload_mb": 10240,
                "total_download_mb": 51200,
                "current_month_upload_mb": 2048,
                "current_month_download_mb": 10240,
                "remaining_quota_mb": 20480,
            }
        },
    },
    "policies": [
        {
            "policy_id": "P_WIFI_OPEN",
            "topic": "wifi_switch",
            "device_model": "HW-5G-CPE-Pro",
            "action_allowed": True,
        }
    ],
}


def print_header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def print_json(label, data):
    print(f"\n--- {label} ---")
    print(json.dumps(data, ensure_ascii=False, indent=2))


def print_sandbox(sandbox, label="当前台账状态"):
    print(f"\n>>> {label}")
    state = sandbox.state
    for key in SANDBOX_KEYS:
        records = state.get(key, [])
        print(f"  [{key}]: {len(records)} 条记录")
        for i, r in enumerate(records):
            tool = r.get('tool', 'N/A')
            fact = r.get('verified_fact_key', 'N/A')
            print(f"    [{i}] tool={tool}, fact={fact}")


def simulate_tool_call(tool_factory, tool_name, arguments, env, sandbox, context):
    print(f"\n  [模型调用] {tool_name}({json.dumps(arguments, ensure_ascii=False)})")

    observation = tool_factory.execute(
        tool_name,
        arguments,
        env_snapshot=env,
        sandbox=sandbox,
        context=context,
    )

    ok = observation.get("ok")
    if ok:
        print(f"  [执行结果] 成功")
        result_str = json.dumps(observation.get('result'), ensure_ascii=False, indent=2)
        print(f"             result = {result_str[:200]}")
    else:
        print(f"  [执行结果] 失败")
        print(f"             error = {observation.get('error')}")
        print(f"             message = {observation.get('message')}")
        print(f"             source = {observation.get('source')}")

    return observation


def main():
    print_header("Agent Runtime 逐步追踪测试")
    print(f"\n测试Case: {TEST_CASE['case_id']}")
    print(f"客户诉求: {TEST_CASE['customer_message']}")

    # Step 0: 环境初始化
    print_header("Step 0: 环境初始化")

    print("\n[0.1] 创建 ToolFactory，加载22个WiFi工具...")
    tool_factory = ToolFactory()
    tool_names = sorted(tool_factory.tools.keys())
    print(f"      已加载 {len(tool_names)} 个工具:")
    for name in tool_names:
        tool = tool_factory.tools[name]
        perm = "写" if tool.is_write else "读"
        print(f"        - {name} ({perm}): {tool.description[:50]}...")

    print("\n[0.2] 创建 SandboxState（5个空台账）...")
    namespace_id = build_namespace_id("run_trace", "TRACE_TEST_001", "rollout_0001")
    sandbox = SandboxState.from_env_snapshot(TEST_ENV, namespace_id)
    print_sandbox(sandbox, "初始状态（全部为空）")

    context = {
        "run_id": "run_trace",
        "case_id": "TRACE_TEST_001",
        "rollout_id": "rollout_0001",
        "namespace_id": namespace_id,
        "tool_call_id": "tc_1",
    }

    print("\n[0.3] 生成 tool schemas（下发给模型的22个工具定义）...")
    schemas = tool_factory.tool_schemas()
    print(f"      共 {len(schemas)} 个schema")
    for s in schemas:
        name = s["function"]["name"]
        args = list(s["function"]["parameters"]["properties"].keys())
        print(f"        - {name}: args={args if args else '无'}")

    # Step 1: wifi.get_info（读工具）
    print_header("Step 1: 模型调用 wifi.get_info（读工具）")
    print("说明: 客户说搜不到WiFi，模型应该先查当前WiFi配置")

    context["tool_call_id"] = "tc_1"
    obs1 = simulate_tool_call(
        tool_factory, "wifi.get_info", {}, TEST_ENV, sandbox, context
    )
    print_json("wifi.get_info 返回的完整observation", obs1)
    print_sandbox(sandbox, "读工具执行后（台账不变，读工具无副作用）")

    # Step 2: wifi.open（写工具）
    print_header("Step 2: 模型调用 wifi.open（写工具）")
    print("说明: 查完发现WiFi是关闭的(enabled=False)，模型应该开启WiFi")

    context["tool_call_id"] = "tc_2"
    obs2 = simulate_tool_call(
        tool_factory, "wifi.open", {"band": "all"}, TEST_ENV, sandbox, context
    )
    print_json("wifi.open 返回的完整observation", obs2)
    print_sandbox(sandbox, "写工具执行后（switch_log + operation_log 各增加1条）")

    # Step 3: 台账详情
    print_header("Step 3: 台账详情分析")

    print("\n--- switch_log 台账内容 ---")
    for i, r in enumerate(sandbox.state.get("switch_log", [])):
        print(f"  记录[{i}]:")
        print(f"    tool: {r['tool']}")
        print(f"    action: {r.get('action')}")
        print(f"    band: {r.get('band')}")
        print(f"    status: {r.get('status')}")
        print(f"    wifi_opened: {r.get('wifi_opened')}")
        print(f"    verified_fact_key: {r.get('verified_fact_key')}")
        print(f"    namespace_id: {r.get('namespace_id')}")

    print("\n--- operation_log 审计日志内容 ---")
    for i, r in enumerate(sandbox.state.get("operation_log", [])):
        print(f"  记录[{i}]:")
        print(f"    tool: {r['tool']}")
        print(f"    action: {r.get('action')}")
        print(f"    result: {r.get('result')}")

    # Step 4: 重复调用（测试幂等性）
    print_header("Step 4: 再次调用 wifi.open（测试幂等性）")
    print("说明: WiFi已经开启了，再次调用应该报错")

    context["tool_call_id"] = "tc_3"
    obs4 = simulate_tool_call(
        tool_factory, "wifi.open", {"band": "all"}, TEST_ENV, sandbox, context
    )
    print_json("重复调用的observation", obs4)
    print_sandbox(sandbox, "报错后台账不变（没有新记录产生）")

    # Step 5: 参数校验测试
    print_header("Step 5: 调用 wifi.set_channel（测试参数校验）")

    print("\n  [5a] 缺少必填参数 channel...")
    context["tool_call_id"] = "tc_4a"
    obs5a = simulate_tool_call(
        tool_factory, "wifi.set_channel", {}, TEST_ENV, sandbox, context
    )

    print("\n  [5b] 无效信道号 99...")
    context["tool_call_id"] = "tc_4b"
    obs5b = simulate_tool_call(
        tool_factory, "wifi.set_channel", {"channel": 99, "band": "2.4G"}, TEST_ENV, sandbox, context
    )

    print("\n  [5c] 正确调用 channel=6, band=2.4G...")
    context["tool_call_id"] = "tc_4c"
    obs5c = simulate_tool_call(
        tool_factory, "wifi.set_channel", {"channel": 6, "band": "2.4G"}, TEST_ENV, sandbox, context
    )
    print_sandbox(sandbox, "正确调用后（wifi_config_log + operation_log 各增加1条）")

    # Step 6: 导出最终sandbox
    print_header("Step 6: 最终 sandbox 导出")
    final_state = sandbox.export()
    print_json("sandbox_final_state", final_state)

    print("\n--- 统计 ---")
    for key in SANDBOX_KEYS:
        count = len(final_state.get(key, []))
        print(f"  {key}: {count} 条记录")

    # Step 7: 验证查询
    print_header("Step 7: 查询已执行的写工具")
    executed = sandbox.executed_write_tools(namespace_id)
    print(f"本rollout执行过的写工具: {sorted(executed)}")

    print("\n--- wifi.open 的记录 ---")
    records = sandbox.records_for_tool("wifi.open", namespace_id)
    print(f"共 {len(records)} 条")
    for r in records:
        print(f"  band={r.get('band')}, status={r.get('status')}, wifi_opened={r.get('wifi_opened')}")

    print("\n--- wifi.set_channel 的记录 ---")
    records = sandbox.records_for_tool("wifi.set_channel", namespace_id)
    print(f"共 {len(records)} 条")
    for r in records:
        print(f"  band={r.get('band')}, channel={r.get('channel')}, status={r.get('status')}")

    print_header("测试完成")
    print("""
总结:
- 读工具（wifi.get_info）: 查只读表，无副作用，台账不变
- 写工具（wifi.open）: 校验 -> 落switch_log台账 -> 审计日志，台账+2条
- 参数校验: 缺必填参数/无效值会报错，台账不变
- 重复调用: 业务校验失败会报错（如wifi_already_open），台账不变
- 所有写操作都带namespace_id隔离，支持并发rollout
""")


if __name__ == "__main__":
    main()
