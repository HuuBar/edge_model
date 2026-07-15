"""Build verl multi-turn SFT data from validated gold trajectories.

SFT 数据构造的核心思想是“重放 gold trajectory”：
case/env/verifier/gold 已经由上游 build 流程校验过，这里只把 gold 里的动作和 observation
转换成 verl ``MultiTurnSFTDataset`` 能读取的 messages/tools parquet。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from agent.observations import project_observation_for_model
from agent.prompts.templates import render_prompt
from agent.runtime import _case_context
from envs.toolfactory import ToolFactory


DEFAULT_BATCH_DIR = Path("data/batches/sft")
DEFAULT_OUT_DIR = Path("data/sft/stage5")


@dataclass(frozen=True)
class SFTBuildResult:
    """SFT 构造结果摘要，供 CLI 打印和测试断言使用。"""

    out_dir: Path
    train_path: Path
    val_path: Path
    train_count: int
    val_count: int
    jsonl_path: Path
    manifest_path: Path


def build_sft_dataset(
    *,
    batch_dir: Path = DEFAULT_BATCH_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    val_every: int = 10,
    write_parquet: bool = True,
) -> SFTBuildResult:
    """Build train/val parquet files for verl MultiTurnSFTDataset.

    Each example is a faithful replay of the validated gold path:
    system/user prompt -> assistant tool call -> tool observation ... -> final
    assistant answer. Original authoring data is only read; all training rows are
    generated under ``out_dir``.
    """

    if val_every < 2:
        raise ValueError("val_every must be >= 2")
    # manifest 是 batch 的总索引，里面记录每个 case 对应的 case/env/gold/verifier 文件。
    manifest = _read_json(batch_dir / "manifest.json")
    # 排序后再切分，保证每次构造 train/val 都稳定可复现。
    entries = sorted(manifest["entries"], key=lambda item: item["id"])
    # SFT 每条 row 都带同一份生产 tool schema；模型需要在监督样本里看到真实工具协议。
    tool_schemas = ToolFactory().tool_schemas()

    rows: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        # 简单确定性 split：第 0、10、20... 条进 val，其余进 train。
        split = "val" if index % val_every == 0 else "train"
        row = _build_row(batch_dir=batch_dir, entry=entry, tools=tool_schemas, split=split, index=index)
        rows.append(row)

    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "val"]
    out_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = out_dir / "all.jsonl"
    _write_jsonl(jsonl_path, rows)
    train_jsonl_path = out_dir / "train.jsonl"
    val_jsonl_path = out_dir / "val.jsonl"
    _write_jsonl(train_jsonl_path, train_rows)
    _write_jsonl(val_jsonl_path, val_rows)

    train_path = out_dir / "train.parquet"
    val_path = out_dir / "val.parquet"
    if write_parquet:
        # verl 训练直接读 parquet；JSONL 主要给人检查和 debug。
        _write_parquet(train_path, train_rows)
        _write_parquet(val_path, val_rows)

    # manifest 记录这份派生数据的来源、列名和 verl dataset 读法。
    out_manifest = {
        "dataset_id": out_dir.name,
        "type": "verl_multiturn_sft",
        "source_batch": _portable_path(batch_dir),
        "source_manifest_id": manifest.get("manifest_id"),
        "count": len(rows),
        "train_count": len(train_rows),
        "val_count": len(val_rows),
        "val_every": val_every,
        "columns": sorted(rows[0].keys()) if rows else [],
        "parquet": {
            "train": _portable_path(train_path),
            "val": _portable_path(val_path),
        },
        "jsonl": {
            "all": _portable_path(jsonl_path),
            "train": _portable_path(train_jsonl_path),
            "val": _portable_path(val_jsonl_path),
        },
        "verl_dataset": {
            "class": "verl.utils.dataset.multiturn_sft_dataset.MultiTurnSFTDataset",
            "messages_key": "messages",
            "tools_key": "tools",
            "enable_thinking_key": "enable_thinking",
        },
    }
    manifest_path = out_dir / "manifest.json"
    _write_json(manifest_path, out_manifest)
    return SFTBuildResult(
        out_dir=out_dir,
        train_path=train_path,
        val_path=val_path,
        train_count=len(train_rows),
        val_count=len(val_rows),
        jsonl_path=jsonl_path,
        manifest_path=manifest_path,
    )


def _build_row(
    *,
    batch_dir: Path,
    entry: dict[str, Any],
    tools: list[dict[str, Any]],
    split: str,
    index: int,
) -> dict[str, Any]:
    """把一条 batch entry 转成一条 verl MultiTurn SFT row。

    对真实 case 的转换顺序：
      1. 读取 case 和 gold trajectory。
      2. 用同一套 prompt 模板渲染 system/user 首轮消息。
      3. 对 gold 的每个 parsed_action 追加 assistant tool_call。
      4. 对对应 observation 追加 tool message。
      5. 最后追加 gold final_text 作为 assistant 监督答案。
    """

    files = entry["files"]
    case = _read_json(batch_dir / files["case"])
    gold = _read_json(batch_dir / files["gold"])
    trajectory = gold["gold_trajectory"]

    # 首轮 prompt 必须和 runtime/GRPO 同源：同一 system.txt、同一 step_user.txt、同一 _case_context 投影。
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": render_prompt("system.txt", {})},
        {"role": "user", "content": render_prompt("step_user.txt", {"case": _case_context(case)})},
    ]

    # gold trajectory 里 action 和 observation 分开存；这里先按 tool_call_id 建索引，方便一一配对。
    observations_by_id = {
        obs.get("tool_call_id"): obs
        for obs in trajectory.get("tool_observations", [])
    }
    for action in trajectory.get("parsed_actions", []):
        tool_call_id = action["tool_call_id"]
        # SFT 的 assistant 消息使用原生 function-calling 形态，让 verl chat template 训练模型学会工具调用格式。
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": action["name"],
                            "arguments": json.dumps(action.get("arguments", {}), ensure_ascii=False, sort_keys=True),
                        },
                    }
                ],
            }
        )
        observation = observations_by_id.get(tool_call_id)
        if observation is None:
            # gold 不完整时必须失败，不能生成“有动作无 observation”的半截监督样本。
            raise ValueError(f"gold trajectory missing observation for {entry['id']} {tool_call_id}")
        # 回放给模型的 tool content 只用模型可见投影；完整 observation 不进入 SFT prompt。
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": action["name"],
                "content": json.dumps(
                    project_observation_for_model(observation),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }
        )

    # gold final_text 是最后一条 assistant 监督目标，训练时 loss 会落在这类 assistant tokens 上。
    messages.append({"role": "assistant", "content": gold["final_text"]})
    metadata = entry.get("metadata") or {}
    return {
        "data_source": "industrial_posttrain_stage5_sft",
        "id": entry["id"],
        "case_id": entry["case_id"],
        "split": split,
        "index": index,
        "messages": messages,
        "tools": tools,
        "enable_thinking": False,
        "gold_reward": entry.get("gold_reward"),
        "topic": metadata.get("topic"),
        "primary_intent": metadata.get("primary_intent") or entry.get("primary_intent"),
        "secondary_intent": metadata.get("secondary_intent"),
        "composition": metadata.get("composition"),
        "difficulty": metadata.get("difficulty") or entry.get("difficulty"),
        "routing_bucket": _classification_bucket(batch_dir, entry["case_id"]),
        "source_files": files,
        # prompt_trace_hash 用于追踪 prompt/messages 是否发生漂移。
        "prompt_trace_hash": _stable_hash(messages),
    }


def _classification_bucket(batch_dir: Path, case_id: str) -> str | None:
    """从 batch classification.json 里取 routing bucket，作为训练样本 metadata。"""

    classification_path = batch_dir / "classification.json"
    if not classification_path.exists():
        return None
    rows = _read_json(classification_path).get("rows", [])
    for row in rows:
        if row.get("case_id") == case_id:
            return row.get("routing_bucket")
    return None


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    """写 verl 训练使用的 parquet 文件。"""

    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - depends on env
        raise RuntimeError("pandas/pyarrow are required to write verl parquet data") from exc
    frame = pd.DataFrame(rows)
    frame.to_parquet(path, index=False)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _stable_hash(payload: Any) -> str:
    """稳定短 hash：用于 metadata 审计，不作为安全哈希使用。"""

    return sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _portable_path(path: Path) -> str:
    """Return a package-relative path when possible."""

    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)
