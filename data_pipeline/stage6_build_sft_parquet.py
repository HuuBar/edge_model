"""阶段6：构建SFT训练数据（parquet格式）。

从四件套 + 划分结果构建 verl MultiTurnSFTDataset 能读取的 parquet 文件。

核心逻辑：重放gold trajectory的完整messages序列：
  system prompt → user query → assistant tool call → tool observation → ... → final answer

输出：data/output/sft_train.parquet / sft_val.parquet / sft_test.parquet
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agent.prompts.templates import render_prompt
from agent.observations import project_observation_for_model
from envs.toolfactory import ToolFactory


def build_sft_row(case: dict, env_snapshot: dict, gold: dict,
                  verifier_spec: dict, tool_schemas: list,
                  split: str = "train") -> dict | None:
    """构建单条SFT训练样本。

    messages序列：
      system prompt → user query → [tool call → observation] × N → final answer
    """
    try:
        # Step 1: System prompt
        system_text = render_prompt("system.txt", {})

        # Step 2: User query
        step_text = render_prompt("step_user.txt", {"case": _case_context(case)})

        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": step_text},
        ]

        # Step 3-N: 重放gold trajectory
        trajectory = gold.get("gold_trajectory", {})
        observations_by_id = {
            obs["tool_call_id"]: obs
            for obs in trajectory.get("tool_observations", [])
        }

        for action in trajectory.get("parsed_actions", []):
            tc_id = action["tool_call_id"]

            # Assistant tool call (XML格式)
            tool_call_json = json.dumps({
                "name": action["name"],
                "arguments": action.get("arguments", {})
            }, ensure_ascii=False)
            messages.append({
                "role": "assistant",
                "content": f'<tool_call>{tool_call_json}</tool_call>'
            })

            # Tool observation
            obs = observations_by_id.get(tc_id, {"ok": False, "error": "missing"})
            messages.append({
                "role": "tool",
                "content": json.dumps(project_observation_for_model(obs), ensure_ascii=False)
            })

        # Final: Assistant final answer
        final_text = trajectory.get("final_text", "")
        if final_text:
            messages.append({"role": "assistant", "content": final_text})

        return {
            "messages": messages,
            "tools": tool_schemas,
            "case_id": case["case_id"],
            "primary_intent": case.get("primary_intent", ""),
            "dataset": case.get("dataset", ""),
            "language": case.get("language", "EN"),
            "split": split,
        }

    except Exception as e:
        print(f"    ⚠️ 构建SFT row失败 [{case.get('case_id', '?')}]: {e}")
        return None


def _case_context(case: dict) -> dict:
    """构建case上下文（与runtime中一致）。"""
    return {
        "ticket_id": f"TICKET_{case['case_id']}",
        "customer_message": case["customer_message"],
        "customer_id": case.get("entities", {}).get("customer_id", ""),
        "market": case.get("entities", {}).get("market", ""),
    }


def build_sft_parquet(
    quartets_root: Path,
    split_items: list[dict],
    output_path: Path,
    tool_schemas: list,
):
    """为单个split构建parquet文件。"""
    rows = []
    skipped = 0

    for item in split_items:
        case_dir = quartets_root / item["source_dir"] / item["case_id"]

        # 读取四件套
        try:
            with open(case_dir / "case.json", "r", encoding="utf-8") as f:
                case = json.load(f)
            with open(case_dir / "env_snapshot.json", "r", encoding="utf-8") as f:
                env = json.load(f)
            with open(case_dir / "gold.json", "r", encoding="utf-8") as f:
                gold = json.load(f)
            with open(case_dir / "verifier_spec.json", "r", encoding="utf-8") as f:
                spec = json.load(f)
        except Exception as e:
            skipped += 1
            continue

        row = build_sft_row(case, env, gold, spec, tool_schemas, item.get("split", "train"))
        if row:
            rows.append(row)
        else:
            skipped += 1

    if not rows:
        print(f"  ⚠️ 没有有效数据: {output_path}")
        return

    # 构建parquet
    table = _rows_to_parquet(rows)
    pq.write_table(table, output_path)

    print(f"  ✅ {output_path.name}: {len(rows)}条 (跳过{skipped}条)")
    return len(rows)


def _rows_to_parquet(rows: list[dict]) -> pa.Table:
    """将row列表转换为parquet table。"""
    return pa.Table.from_pylist(rows)


def build_all_splits(
    quartets_root: Path,
    output_dir: Path,
):
    """为所有split构建parquet文件。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    tool_schemas = ToolFactory().tool_schemas()

    splits_dir = quartets_root / "splits"

    for split_name in ["train", "val", "test"]:
        split_file = splits_dir / f"{split_name}.json"
        if not split_file.exists():
            print(f"  ⚠️ 划分文件不存在: {split_file}")
            continue

        with open(split_file, "r", encoding="utf-8") as f:
            split_items = json.load(f)

        output_path = output_dir / f"sft_{split_name}.parquet"
        build_sft_parquet(quartets_root, split_items, output_path, tool_schemas)

    print(f"\n📁 训练数据保存在: {output_dir}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法: python stage6_build_sft_parquet.py <quartets_root_dir> [output_dir]")
        sys.exit(1)

    quartets_root = Path(sys.argv[1])
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else quartets_root.parent / "output"

    build_all_splits(quartets_root, output_dir)
