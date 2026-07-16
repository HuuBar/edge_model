"""阶段4：场景分类。

数据已经通过预处理有了 dataset 字段（场景分类标签）。
本阶段增加两个维度：
  1. 轨迹长度分桶（单步/2-3步/多步）→ 用于区分训练信号强度
  2. 意图分类（从 categories 映射后的工具名提取主意图）

输出：每个 quartet 目录下新增 classification.json
"""

from __future__ import annotations

import json
from pathlib import Path


def classify_quartet(quartet_dir: Path) -> dict | None:
    """对单个四件套进行分类，生成分类信息。"""
    gold_path = quartet_dir / "gold.json"
    case_path = quartet_dir / "case.json"

    if not gold_path.exists() or not case_path.exists():
        return None

    with open(gold_path, "r", encoding="utf-8") as f:
        gold = json.load(f)
    with open(case_path, "r", encoding="utf-8") as f:
        case = json.load(f)

    trajectory = gold.get("gold_trajectory", {})
    actions = trajectory.get("parsed_actions", [])

    # 1. 轨迹长度分桶
    n_actions = len(actions)
    if n_actions <= 1:
        trajectory_bucket = "single_step"      # 单步：简单工具调用
    elif n_actions <= 3:
        trajectory_bucket = "short_chain"      # 短链：2-3步
    else:
        trajectory_bucket = "long_chain"       # 长链：4步以上

    # 2. 主意图（从第一个工具调用推断）
    primary_tool = actions[0]["name"] if actions else "none"
    primary_intent = _tool_to_intent(primary_tool)

    # 3. 场景标签（从dataset字段）
    dataset_label = case.get("dataset", "")
    scene_type = _parse_dataset_label(dataset_label)

    # 4. 复杂度评分
    complexity = _compute_complexity(actions, scene_type)

    classification = {
        "case_id": case["case_id"],
        "trajectory_length": n_actions,
        "trajectory_bucket": trajectory_bucket,
        "primary_tool": primary_tool,
        "primary_intent": primary_intent,
        "dataset_label": dataset_label,
        "scene_type": scene_type,
        "complexity": complexity,
        "has_error": any(not obs.get("ok", True)
                         for obs in trajectory.get("tool_observations", [])),
        "is_multi_turn": "多轮" in dataset_label,
        "is_multi_intent": "多意图" in dataset_label,
        "has_clarification": "澄清" in dataset_label or "接续" in dataset_label,
    }

    # 写入
    with open(quartet_dir / "classification.json", "w", encoding="utf-8") as f:
        json.dump(classification, f, ensure_ascii=False, indent=2)

    return classification


def _tool_to_intent(tool_name: str) -> str:
    """从工具名推断意图类别。"""
    if not tool_name:
        return "none"
    prefix = tool_name.split(".")[0] if "." in tool_name else tool_name
    intent_map = {
        "wifi": "wifi_config",
        "data": "data_management",
        "network": "network_settings",
        "device": "device_management",
        "system": "system_info",
        "policy": "policy_query",
        "user": "user_management",
    }
    return intent_map.get(prefix, "other")


def _parse_dataset_label(label: str) -> dict:
    """解析 dataset 标签，提取轮次/意图/场景信息。"""
    result = {
        "turn_type": "single",       # single / multi
        "intent_type": "single",     # single / multi
        "scene_subtype": "",         # API / QA / 参数缺失 / 参数错误 / 模糊 / ...
    }

    if "多轮" in label:
        result["turn_type"] = "multi"
    if "多意图" in label:
        result["intent_type"] = "multi"

    # 提取子场景
    for subtype in ["API类", "QA类", "参数缺失", "参数错误", "模糊",
                    "业务无关", "条件依赖", "简单多任务", "QA转API",
                    "参数二次修改", "参数缺失接续", "参数错误接续",
                    "多轮切阈", "意图接续"]:
        if subtype in label:
            result["scene_subtype"] = subtype
            break

    return result


def _compute_complexity(actions: list, scene_type: dict) -> str:
    """计算复杂度等级。"""
    n = len(actions)
    has_error = any(not a.get("ok", True) for a in actions)
    is_multi = scene_type.get("intent_type") == "multi"

    if n >= 4 or (is_multi and n >= 2):
        return "complex"
    elif n >= 2 or has_error:
        return "medium"
    else:
        return "simple"


def classify_all_quartets(quartets_root: Path):
    """批量分类所有四件套。"""
    stats = {"simple": 0, "medium": 0, "complex": 0, "single_step": 0,
             "short_chain": 0, "long_chain": 0, "total": 0}

    for file_dir in sorted(quartets_root.iterdir()):
        if not file_dir.is_dir():
            continue
        for case_dir in sorted(file_dir.iterdir()):
            if not case_dir.is_dir():
                continue
            result = classify_quartet(case_dir)
            if result:
                stats["total"] += 1
                stats[result["complexity"]] += 1
                stats[result["trajectory_bucket"]] += 1

    print(f"\n📊 分类统计:")
    print(f"  总计: {stats['total']}")
    print(f"  复杂度: simple={stats['simple']}, medium={stats['medium']}, complex={stats['complex']}")
    print(f"  轨迹长度: single_step={stats['single_step']}, short_chain={stats['short_chain']}, long_chain={stats['long_chain']}")

    return stats


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法: python stage4_classify.py <quartets_root_dir>")
        sys.exit(1)
    classify_all_quartets(Path(sys.argv[1]))
