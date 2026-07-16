"""
核心转换模块：从预处理后的JSON数据 → 四件套(case/env/gold/verifier)。

输入格式（预处理后的JSON）：
  query_cn/en/ar, history, intent_label, rag_en/ar, react_label_en/ar, categories, dataset

输出格式（四件套）：
  case.json — 用户query + 上下文
  env_snapshot.json — 设备状态 + 可用工具
  gold.json — 标准工具调用轨迹
  verifier_spec.json — 评分规则

关键逻辑：
  1. 从 react_label_en 提取 ReAct 轨迹（Thought → Action → Observation → Final Answer）
  2. 从 rag.tools 提取旧工具定义 → 映射为新工具名
  3. 从 rag.knowledge + tool_result 反推 env_snapshot
  4. 从 react_label 中的工具调用序列推断 verifier_spec
"""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any

from data_pipeline.tool_name_mapping import (
    map_tool_name, map_tool_name_with_args, map_categories,
)


# ═══════════════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════════════

def convert_record_to_quartet(
    record: dict[str, Any],
    source_file: str = "",
    index: int = 0,
) -> dict[str, Any] | None:
    """把单条预处理JSON记录转换为四件套。

    参数
    ----
    record: 预处理后的单条JSON数据
    source_file: 来源文件名（用于生成case_id）
    index: 记录在文件中的索引

    返回
    ----
    {"case": case, "env_snapshot": env, "gold": gold, "verifier_spec": spec}
    或 None（数据无法转换时）
    """
    # 1. 基础信息
    case_id = _make_case_id(source_file, index)
    query = record.get("query_en", "") or record.get("query_cn", "")
    dataset_label = record.get("dataset", "")
    categories = record.get("categories", [])
    react_label = record.get("react_label_en", [])

    if not react_label:
        return None  # 无轨迹数据，跳过

    # 2. 解析ReAct轨迹，提取工具调用
    parsed_trajectory = _parse_react_trajectory(react_label)
    if not parsed_trajectory:
        return None

    # 3. 构建四件套
    case = _build_case(record, case_id)
    env = _build_env_snapshot(record, case_id)
    gold = _build_gold(parsed_trajectory, case_id)
    spec = _build_verifier_spec(parsed_trajectory, categories, dataset_label)

    return {
        "case": case,
        "env_snapshot": env,
        "gold": gold,
        "verifier_spec": spec,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 1. case.json
# ═══════════════════════════════════════════════════════════════════════════

def _build_case(record: dict, case_id: str) -> dict[str, Any]:
    """构建case.json：用户query + 上下文。"""
    # 判断语言
    language = "EN"
    if record.get("query_en"):
        customer_message = record["query_en"]
        language = "EN"
    elif record.get("query_cn"):
        customer_message = record["query_cn"]
        language = "CN"
    elif record.get("query_ar"):
        customer_message = record["query_ar"]
        language = "AR"
    else:
        customer_message = ""

    # 多轮：从历史对话构建上下文
    history = record.get("history_en", []) or []

    # entities: 从rag中提取设备ID（如果没有则用默认值）
    device_id = f"DEV_{case_id}"

    case = {
        "case_id": case_id,
        "customer_message": customer_message,
        "entities": {"device_id": device_id},
        "primary_intent": _infer_primary_intent(record),
        "language": language,
        "dataset": record.get("dataset", ""),
    }

    # 如果有历史对话，加入case
    if history:
        case["conversation_history"] = history

    return case


# ═══════════════════════════════════════════════════════════════════════════
# 2. env_snapshot.json
# ═══════════════════════════════════════════════════════════════════════════

def _build_env_snapshot(record: dict, case_id: str) -> dict[str, Any]:
    """构建env_snapshot.json：设备状态 + 可用工具。

    数据来源：
      - rag.knowledge → 作为背景知识
      - react_label 中 tool_result → 反推设备状态
      - rag.tools → 映射为新工具定义，作为 env 中的可用工具
    """
    # 从 rag_en 中提取信息
    rag = record.get("rag_en", {})
    knowledge = rag.get("knowledge", {})

    # 从 react_label 中提取 tool_result，反推设备状态
    react_label = record.get("react_label_en", [])
    inferred_state = _infer_state_from_tool_results(react_label)

    # 映射 rag.tools 中的工具定义
    old_tools = rag.get("tools", [])
    mapped_tools = _map_rag_tools(old_tools)

    env = {
        "case_id": case_id,
        "reference_now": "2026-07-15T10:00:00",
        "readonly_tables": {
            "device_info": {
                f"DEV_{case_id}": {
                    "device_id": f"DEV_{case_id}",
                    "model": "HW-5G-CPE-Pro",
                    "firmware_version": "V3.2.1",
                    "imei": f"86000001{case_id[-6:].zfill(6)}",
                    "uptime_seconds": 86400,
                }
            },
            "wifi_config": {
                f"DEV_{case_id}": inferred_state.get("wifi_config", {
                    "device_id": f"DEV_{case_id}",
                    "ssid": "MyWiFi_5G",
                    "password": "********",
                    "encryption": "WPA2-PSK",
                    "channel": 36,
                    "band": "5G",
                    "bandwidth": "80MHz",
                    "hidden": False,
                    "enabled": True,
                    "max_clients": 32,
                })
            },
            "network_status": {
                f"DEV_{case_id}": inferred_state.get("network_status", {
                    "device_id": f"DEV_{case_id}",
                    "connected": True,
                    "signal_strength": -75,
                    "rsrp": -85,
                    "sinr": 15,
                    "download_speed_kbps": 51200,
                    "upload_speed_kbps": 10240,
                    "latency_ms": 35,
                    "packet_loss_percent": 0.5,
                    "network_type": "5G",
                })
            },
            "connected_clients": {},
            "data_usage": {
                f"DEV_{case_id}": inferred_state.get("data_usage", {
                    "device_id": f"DEV_{case_id}",
                    "total_upload_mb": 10240,
                    "total_download_mb": 51200,
                    "current_month_upload_mb": 2048,
                    "current_month_download_mb": 10240,
                    "remaining_quota_mb": 20480,
                })
            },
            "network_settings": {},
            "dhcp_leases": {},
            "system_logs": {},
            "policies": _extract_policies(record),
        },
        "policies": _extract_policies(record),
        "knowledge": knowledge,  # 保留原始知识，供模型参考
        "available_tools": mapped_tools,  # 映射后的工具定义
    }

    return env


def _infer_state_from_tool_results(react_label: list[dict]) -> dict[str, dict]:
    """从 tool_result 中反推设备状态。

    例如：
      tool_result: "总共17.3GB,当月3.2GB" → data_usage.current_month_download_mb = 3200
      tool_result: "channel不在范围内" → wifi_config 中 channel 无效
    """
    state: dict[str, dict] = {}

    for step in react_label:
        content = step.get("content", {})
        if not isinstance(content, dict):
            continue

        tool_name = content.get("tool_name", "")
        tool_result = content.get("tool_result", "")
        args = content.get("args", {})

        mapped_tool = map_tool_name_with_args(tool_name, args)

        # 从 data.get_usage 的 tool_result 提取流量
        if mapped_tool == "data.get_usage" and tool_result:
            data = _parse_data_usage_result(tool_result)
            if data:
                state["data_usage"] = data

        # 从 wifi.set_channel 的 args 提取信道
        if mapped_tool == "wifi.set_channel" and args:
            channel = args.get("channel")
            if channel is not None:
                state.setdefault("wifi_config", {})
                state["wifi_config"]["channel"] = int(channel) if str(channel).isdigit() else channel

        # 从 wifi.set_config 的 args 提取SSID
        if mapped_tool == "wifi.set_config" and args:
            ssid = args.get("ssid") or args.get("name")
            if ssid:
                state.setdefault("wifi_config", {})
                state["wifi_config"]["ssid"] = ssid

        # 从 data.set_limit 的 args 提取限额
        if mapped_tool == "data.set_limit" and args:
            limit = args.get("limit_mb") or args.get("limit")
            if limit:
                state.setdefault("data_usage", {})
                state["data_usage"]["limit_mb"] = _parse_limit_value(limit)

    return state


def _parse_data_usage_result(result: str) -> dict[str, Any] | None:
    """从 tool_result 字符串解析流量数据。

    示例: "总共17.3GB,当月3.2GB" → {total_download_mb: 17700, current_month_download_mb: 3200}
    """
    try:
        total_match = re.search(r"总共([\d.]+)\s*GB", result)
        month_match = re.search(r"当月([\d.]+)\s*GB", result)

        data = {}
        if total_match:
            data["total_download_mb"] = int(float(total_match.group(1)) * 1024)
        if month_match:
            data["current_month_download_mb"] = int(float(month_match.group(1)) * 1024)

        return data if data else None
    except Exception:
        return None


def _parse_limit_value(value: Any) -> int:
    """解析限额值（支持 '60GB', 61440, '61440' 等格式）。"""
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        # 尝试提取数字
        match = re.search(r"(\d+)", value)
        if match:
            num = int(match.group(1))
            # 如果原始字符串含GB且数字较小，可能是GB单位
            if "GB" in value.upper() and num < 1024:
                return num * 1024  # GB → MB
            return num
    return 0


def _map_rag_tools(old_tools: list[dict]) -> list[dict]:
    """把 rag.tools 中的旧工具定义映射为新工具名。"""
    mapped = []
    for tool in old_tools:
        old_name = tool.get("function", {}).get("name", "")
        new_name = map_tool_name(old_name)
        if new_name:
            new_tool = dict(tool)
            new_tool["function"]["name"] = new_name
            mapped.append(new_tool)
        # 无法映射的跳过
    return mapped


def _extract_policies(record: dict) -> list[dict]:
    """从 record 中提取 policy 信息。"""
    # 默认：所有写操作都允许
    return [
        {
            "policy_id": "P_ALLOW_ALL",
            "topic": "general",
            "device_model": "HW-5G-CPE-Pro",
            "action_allowed": True,
        }
    ]


# ═══════════════════════════════════════════════════════════════════════════
# 3. gold.json
# ═══════════════════════════════════════════════════════════════════════════

def _build_gold(parsed_trajectory: list[dict], case_id: str) -> dict[str, Any]:
    """构建gold.json：标准工具调用轨迹。

    从 parsed_trajectory 中构建完整的 trajectory 结构。
    """
    parsed_actions = []
    tool_observations = []

    for i, step in enumerate(parsed_trajectory):
        tool_call_id = f"tc_{i + 1}"

        # parsed_action
        parsed_actions.append({
            "step": i + 1,
            "tool_call_id": tool_call_id,
            "name": step["tool_name"],
            "arguments": step.get("args", {}),
            "timestamp": f"2026-07-15T10:{i:02d}:{i * 10:02d}",
        })

        # tool_observation
        observation = {
            "tool_call_id": tool_call_id,
            "tool_name": step["tool_name"],
            "ok": step.get("success", True),
            "result": step.get("tool_result") if step.get("success") else None,
            "error": None if step.get("success") else step.get("tool_result"),
        }
        tool_observations.append(observation)

    # final_text 从最后一步的 final_answer 提取
    final_text = ""
    for step in reversed(parsed_trajectory):
        if step.get("final_answer"):
            final_text = step["final_answer"]
            break

    return {
        "case_id": case_id,
        "gold_trajectory": {
            "namespace_id": f"gold:{case_id}:gold_001",
            "parsed_actions": parsed_actions,
            "tool_observations": tool_observations,
            "final_text": final_text,
            "tool_errors": [obs for obs in tool_observations if not obs.get("ok")],
        }
    }


# ═══════════════════════════════════════════════════════════════════════════
# 4. verifier_spec.json
# ═══════════════════════════════════════════════════════════════════════════

def _build_verifier_spec(
    parsed_trajectory: list[dict],
    categories: list[str],
    dataset_label: str,
) -> dict[str, Any]:
    """构建verifier_spec.json：评分规则。

    从trajectory中的工具调用序列推断评分规则。
    """
    # 从trajectory中提取读工具和写工具
    read_tools = []
    write_tools = []
    required_side_effects = []

    for step in parsed_trajectory:
        tool_name = step["tool_name"]
        args = step.get("args", {})
        success = step.get("success", True)

        # 判断是读还是写（根据工具名前缀）
        if tool_name.startswith(("wifi.get", "device.get", "data.get",
                                  "network.get", "system.get", "policy.search")):
            if tool_name not in read_tools:
                read_tools.append(tool_name)
        elif tool_name.startswith(("wifi.set", "wifi.open", "wifi.close",
                                    "wifi.hide", "wifi.switch",
                                    "data.set", "network.set",
                                    "device.restart", "user.change")):
            if tool_name not in write_tools:
                write_tools.append(tool_name)

            # 成功的写操作 → required_side_effects
            if success:
                required_side_effects.append({
                    "id": f"se_{len(required_side_effects) + 1}",
                    "tool": tool_name,
                    "required_correct": {},  # 简化版：发生即算对
                })

    # 从 dataset_label 推断 evidence_required
    evidence_required = "API" in str(dataset_label) or len(read_tools) > 0

    # 从 dataset_label 推断 max_steps
    if "多轮" in dataset_label:
        max_steps = 10
    elif "条件依赖" in dataset_label:
        max_steps = 8
    else:
        max_steps = 6

    # 构建 response_points（基于 final_answer 的内容）
    response_points = []
    final_answer = ""
    for step in reversed(parsed_trajectory):
        if step.get("final_answer"):
            final_answer = step["final_answer"]
            break

    if final_answer:
        response_points.append({
            "id": "rp_1",
            "description": f"最终回复应包含: {final_answer[:100]}",
        })

    return {
        "policy_required": False,
        "evidence_required": evidence_required,
        "required_read_tools": read_tools,
        "allowed_write_tools": write_tools,
        "required_side_effects": required_side_effects,
        "forbidden_side_effects": [],
        "required_response_points": response_points,
        "forbidden_text_points": [],
        "max_steps": max_steps,
        "version": "verifier_simple_v1",
    }


# ═══════════════════════════════════════════════════════════════════════════
# ReAct 轨迹解析
# ═══════════════════════════════════════════════════════════════════════════

def _parse_react_trajectory(react_label: list[dict]) -> list[dict] | None:
    """从 react_label_en 解析 ReAct 轨迹，提取工具调用序列。

    输入格式（ReAct）：
      [
        {"role": "assistant", "content": {
          "thought": "...",
          "tool_name": "set_wifi_channel",
          "args": {"channel": "22"},
          "tool_result": "channel不在范围内...",
          "final_answer": ""
        }},
        {"role": "assistant", "content": {
          "thought": "...",
          "tool_name": "",
          "args": {},
          "tool_result": "",
          "final_answer": "抱歉，您提供的信道22超出了有效范围..."
        }}
      ]

    输出格式：
      [
        {"tool_name": "wifi.set_channel", "args": {"channel": "22"},
         "tool_result": "channel不在范围内...", "success": False, "final_answer": ""},
        {"tool_name": "", "args": {}, "tool_result": "", "success": True,
         "final_answer": "抱歉..."}
      ]
    """
    parsed = []

    for step in react_label:
        if not isinstance(step, dict):
            continue

        content = step.get("content", {})
        if not isinstance(content, dict):
            continue

        tool_name = content.get("tool_name", "")
        args = content.get("args", {})
        tool_result = content.get("tool_result", "")
        final_answer = content.get("final_answer", "")

        # 映射工具名（带参数判断，处理 switch_wifi_enable 等特殊映射）
        mapped_tool = map_tool_name_with_args(tool_name, args) if tool_name else ""

        # 判断成功/失败
        success = True
        if tool_result and any(kw in str(tool_result) for kw in [
            "不在范围内", "错误", "失败", "invalid", "error",
            "not found", "超出", "不支持", "Forbidden"
        ]):
            success = False

        parsed.append({
            "tool_name": mapped_tool or tool_name,
            "original_tool_name": tool_name if mapped_tool != tool_name else "",
            "args": args,
            "tool_result": tool_result,
            "success": success,
            "final_answer": final_answer,
        })

    # 过滤掉完全没有内容的步骤
    parsed = [s for s in parsed if s["tool_name"] or s["final_answer"]]

    return parsed if parsed else None


# ═══════════════════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════════════════

def _make_case_id(source_file: str, index: int) -> str:
    """生成唯一的case_id。"""
    # 从文件名提取类别信息
    file_name = Path(source_file).stem if source_file else "unknown"
    # 清理文件名中的特殊字符
    clean_name = re.sub(r"[^\w\u4e00-\u9fff]", "_", file_name)
    return f"{clean_name}_{index:04d}"


def _infer_primary_intent(record: dict) -> str:
    """推断主意图。"""
    # 优先从 intent_label 提取
    intent_label = record.get("intent_label_en", {})
    tasks = intent_label.get("task_decomposition", [])
    if tasks:
        return tasks[0].get("sub_task", "unknown")

    # 从 categories 推断
    categories = record.get("categories", [])
    if categories:
        mapped = map_categories(categories)
        return mapped[0] if mapped else categories[0]

    # 从 dataset 推断
    return record.get("dataset", "unknown")


# ═══════════════════════════════════════════════════════════════════════════
# 批量转换入口
# ═══════════════════════════════════════════════════════════════════════════

def convert_json_file(input_path: Path, output_dir: Path) -> dict[str, int]:
    """转换单个JSON文件中的所有记录。

    参数
    ----
    input_path: 输入JSON文件路径
    output_dir: 输出目录（四件套将保存在 output_dir/<file_name>/ 下）

    返回
    ----
    {"total": N, "success": M, "failed": K}
    """
    with open(input_path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    records = data if isinstance(data, list) else [data]
    file_name = input_path.stem

    # 创建输出子目录
    quartet_dir = output_dir / file_name
    quartet_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    failed_count = 0

    for i, record in enumerate(records):
        try:
            quartet = convert_record_to_quartet(record, source_file=file_name, index=i)
            if quartet is None:
                failed_count += 1
                continue

            # 写入四件套
            case_id = quartet["case"]["case_id"]
            case_dir = quartet_dir / case_id
            case_dir.mkdir(exist_ok=True)

            _write_json(case_dir / "case.json", quartet["case"])
            _write_json(case_dir / "env_snapshot.json", quartet["env_snapshot"])
            _write_json(case_dir / "gold.json", quartet["gold"])
            _write_json(case_dir / "verifier_spec.json", quartet["verifier_spec"])

            success_count += 1
        except Exception as e:
            failed_count += 1
            print(f"  ⚠️ 转换失败 [{file_name} #{i}]: {e}")

    return {"total": len(records), "success": success_count, "failed": failed_count}


def _write_json(path: Path, data: dict):
    """写入JSON文件。"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ═══════════════════════════════════════════════════════════════════════════
# CLI入口
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("用法: python convert_react_to_quartet.py <input_json_file> <output_dir>")
        print("  或: python convert_react_to_quartet.py --batch <input_dir> <output_dir>")
        sys.exit(1)

    if sys.argv[1] == "--batch":
        # 批量模式：转换目录下所有JSON文件
        input_dir = Path(sys.argv[2])
        output_dir = Path(sys.argv[3])
        output_dir.mkdir(parents=True, exist_ok=True)

        json_files = sorted(input_dir.rglob("*.json"))
        print(f"📁 输入目录: {input_dir}")
        print(f"📁 输出目录: {output_dir}")
        print(f"📄 发现 {len(json_files)} 个JSON文件\n")

        total_all = 0
        success_all = 0
        failed_all = 0

        for json_file in json_files:
            print(f"  处理: {json_file.relative_to(input_dir)}")
            stats = convert_json_file(json_file, output_dir)
            total_all += stats["total"]
            success_all += stats["success"]
            failed_all += stats["failed"]
            print(f"    ✅ {stats['success']}/{stats['total']} 成功, ❌ {stats['failed']} 失败")

        print(f"\n📊 总计: {total_all}条, ✅ {success_all}成功, ❌ {failed_all}失败")
        print(f"📁 四件套保存在: {output_dir}")
    else:
        # 单文件模式
        input_path = Path(sys.argv[1])
        output_dir = Path(sys.argv[2])
        stats = convert_json_file(input_path, output_dir)
        print(f"📊 {stats['total']}条, ✅ {stats['success']}成功, ❌ {stats['failed']}失败")
