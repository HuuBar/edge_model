"""Build verl RL/GRPO prompt data for the industrial tool runtime.

GRPO parquet 是 prompt-only：
- ``prompt`` 给 verl/vLLM 做首轮生成。
- ``extra_info`` 保存 case/env/verifier/gold 的文件路径和 rollout 落盘目录。
- gold 只作为审计/对照路径保存，不作为训练 target；reward 来自在线 rollout 后的 verifier。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from agent.prompts.templates import render_prompt
from agent.runtime import _case_context


DEFAULT_BATCH_DIR = Path("data/batches/rl")
DEFAULT_OUT_DIR = Path("data/rl/stage5")
DEFAULT_ROLLOUT_ROOT = Path("data/rollouts_verl")


@dataclass(frozen=True)
class GRPOBuildResult:
    """GRPO 构造结果摘要，供 CLI 打印和测试断言使用。"""

    out_dir: Path
    train_path: Path
    val_path: Path
    train_count: int
    val_count: int
    jsonl_path: Path
    manifest_path: Path


def build_grpo_dataset(
    *,
    batch_dir: Path = DEFAULT_BATCH_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    rollout_root: Path = DEFAULT_ROLLOUT_ROOT,
    val_every: int = 10,
    write_parquet: bool = True,
) -> GRPOBuildResult:
    """Build prompt-only verl RLHFDataset rows for custom AgentLoop rollout."""

    if val_every < 2:
        raise ValueError("val_every must be >= 2")
    # manifest 提供 batch 内每条 case 的四件套相对路径。
    manifest = _read_json(batch_dir / "manifest.json")
    # classification 只做 metadata，不参与 reward 计算。
    classification = _classification_by_case(batch_dir)
    # 排序后 deterministic split，保证多次构造 train/val 一致。
    entries = sorted(manifest["entries"], key=lambda item: item["id"])

    rows: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        split = "val" if index % val_every == 0 else "train"
        rows.append(
            _build_row(
                batch_dir=batch_dir,
                rollout_root=rollout_root,
                entry=entry,
                classification=classification.get(entry["case_id"], {}),
                split=split,
                index=index,
            )
        )

    train_rows = [row for row in rows if row["extra_info"]["split"] == "train"]
    val_rows = [row for row in rows if row["extra_info"]["split"] == "val"]
    out_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = out_dir / "all.jsonl"
    train_jsonl_path = out_dir / "train.jsonl"
    val_jsonl_path = out_dir / "val.jsonl"
    _write_jsonl(jsonl_path, rows)
    _write_jsonl(train_jsonl_path, train_rows)
    _write_jsonl(val_jsonl_path, val_rows)

    train_path = out_dir / "train.parquet"
    val_path = out_dir / "val.parquet"
    if write_parquet:
        # verl 的 RLHFDataset 直接读取 parquet；JSONL 用于人工检查。
        _write_parquet(train_path, train_rows)
        _write_parquet(val_path, val_rows)

    # manifest 明确告诉读者这份数据应由 RLHFDataset 读取，并且 prompt_key 是 prompt。
    out_manifest = {
        "dataset_id": out_dir.name,
        "type": "verl_grpo_agent_loop",
        "source_batch": _portable_path(batch_dir),
        "source_manifest_id": manifest.get("manifest_id"),
        "rollout_root": _portable_path(rollout_root),
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
            "class": "verl.utils.dataset.rl_dataset.RLHFDataset",
            "prompt_key": "prompt",
            "return_raw_chat": True,
        },
    }
    manifest_path = out_dir / "manifest.json"
    _write_json(manifest_path, out_manifest)
    return GRPOBuildResult(
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
    rollout_root: Path,
    entry: dict[str, Any],
    classification: dict[str, Any],
    split: str,
    index: int,
) -> dict[str, Any]:
    """把一条 batch entry 转成一条 GRPO prompt row。

    和 SFT builder 的区别：
    - 不重放 gold parsed_actions。
    - 不把 tool observation 放进 messages。
    - 只保存首轮 system/user prompt。
    - 把 case/env/verifier 路径写进 extra_info，供 AgentLoop 在线读取。
    """

    files = entry["files"]
    case = _read_json(batch_dir / files["case"])
    # 首轮 prompt 与 SFT/runtime 同源，保证训练前后看到的 case 投影一致。
    prompt = [
        {"role": "system", "content": render_prompt("system.txt", {})},
        {"role": "user", "content": render_prompt("step_user.txt", {"case": _case_context(case)})},
    ]
    metadata = entry.get("metadata") or {}
    # extra_info 是 GRPO 接线的核心：verl 会把它原样传给 AgentLoop.run(**kwargs)。
    # Adapter 再通过这些路径读取 case/env/verifier，并把 rollout artifact 写到 rollout_artifact_root。
    extra_info = {
        "index": index,
        "split": split,
        "case_id": entry["case_id"],
        "entry_id": entry["id"],
        "batch_dir": _portable_path(batch_dir),
        "case_path": _portable_path(batch_dir / files["case"]),
        "env_snapshot_path": _portable_path(batch_dir / files["env_snapshot"]),
        "verifier_spec_path": _portable_path(batch_dir / files["verifier_spec"]),
        "gold_path": _portable_path(batch_dir / files["gold"]),
        "rollout_artifact_root": _portable_path(rollout_root),
        "need_tools_kwargs": False,
        "routing_bucket": classification.get("routing_bucket"),
        "intent_type": classification.get("intent_type"),
        "topic": metadata.get("topic"),
        "primary_intent": metadata.get("primary_intent") or entry.get("primary_intent"),
        "secondary_intent": metadata.get("secondary_intent"),
        "composition": metadata.get("composition"),
        "difficulty": metadata.get("difficulty") or entry.get("difficulty"),
        "gold_reward": entry.get("gold_reward"),
        "prompt_trace_hash": _stable_hash(prompt),
    }
    return {
        "data_source": "industrial_posttrain_stage5_grpo",
        "prompt": prompt,
        "ability": "after_sales_tool_agent",
        "reward_model": {
            # 这里不是启用 verl 内置 reward model，而是给样本 metadata 标明 reward 口径。
            # train_grpo_verl.py 已设置 reward_model.enable=False。
            "style": "industrial_posttrain_verifier_with_deepseek_judge",
            "ground_truth": entry["case_id"],
        },
        "extra_info": extra_info,
    }


def _classification_by_case(batch_dir: Path) -> dict[str, dict[str, Any]]:
    """读取 routing/classification metadata；文件不存在时返回空映射。"""

    path = batch_dir / "classification.json"
    if not path.exists():
        return {}
    rows = _read_json(path).get("rows", [])
    return {row["case_id"]: row for row in rows}


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    """写 verl RLHFDataset 使用的 parquet 文件。"""

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
    """稳定短 hash：用于确认首轮 prompt 是否漂移。"""

    return sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _portable_path(path: Path) -> str:
    """Return a package-relative path when possible."""

    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)
