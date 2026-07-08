# 版权所有 © 2026 深圳途明智启科技有限公司。保留所有权利。
# 未经书面许可，任何单位或个人不得复制、传播、发布、转卖、改编、仿制或用于商业用途。
# 侵权必究。

"""Rollout artifact IDs and persistent storage.

runtime 只负责跑出一条内存里的 trajectory；本模块负责“工程落盘”：
- 生成 run_id / rollout_id。
- 计算单条 rollout 的 artifact 目录。
- 把 trajectory、prompt_history、工具调用、工具结果、score 等拆成可审计文件。

这样做的好处是：训练前后、debug、复算 reward 时，不需要重新跑模型，也能从磁盘
还原一条 rollout 当时看到的 prompt、执行过的工具、最终 sandbox 状态和分数。
"""

from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# 项目根目录：agent/rollout_store.py 的上一级父目录就是仓库根。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# 默认落盘位置。调用方也可传 root 覆盖，便于测试或临时实验写到隔离目录。
DEFAULT_ROLLOUT_ROOT = PROJECT_ROOT / "data" / "rollouts"


def make_run_id(prefix: str = "run", now: datetime | None = None) -> str:
    """生成一次运行的 run_id。

    格式为 ``run_UTC时间_随机后缀``。时间方便人读，随机后缀避免同一秒并发启动时目录冲突。
    ``now`` 参数只给测试用，生产调用通常不传。
    """

    # 统一用 UTC，避免不同机器/时区下同一次实验目录名不一致。
    current = now or datetime.now(timezone.utc)
    stamp = current.astimezone(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{stamp}_{uuid.uuid4().hex[:8]}"


def make_rollout_id(index: int, width: int = 4) -> str:
    """按 run 内序号生成确定性的 rollout_id。

    例如 index=1 时得到 ``rollout_0001``。确定性编号让同一个 case 多采样时便于排序和对比。
    """

    if index < 1:
        # rollout 编号从 1 开始，避免 rollout_0000 和“未初始化/空值”语义混淆。
        raise ValueError("rollout index must start at 1")
    return f"rollout_{index:0{width}d}"


def rollout_artifact_dir(
    *,
    root: str | Path | None = None,
    run_id: str,
    case_id: str,
    rollout_id: str,
) -> Path:
    """计算单条 rollout 的 artifact 目录。

    目录层级固定为 ``root/run_id/case_id/rollout_id``。每个 path part 都会清洗，
    防止 case_id 等外部字符串意外包含斜杠导致写到目录外。
    """

    base = Path(root) if root is not None else DEFAULT_ROLLOUT_ROOT
    return base / _safe_path_part(run_id) / _safe_path_part(case_id) / _safe_path_part(rollout_id)


def write_rollout_artifacts(
    *,
    trajectory: dict[str, Any],
    root: str | Path | None = None,
    case: dict[str, Any] | None = None,
    env_snapshot: dict[str, Any] | None = None,
    verifier_spec: dict[str, Any] | None = None,
    score: dict[str, Any] | None = None,
    extra_metadata: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> Path:
    """把一条 rollout 的所有可审计产物写入磁盘。

    写入文件分三类：
    - 主产物：trajectory.json、sandbox_final_state.json、final_text.txt。
    - 过程切片：prompt_history/raw_model_outputs/parsed_actions/tool_observations/tool_errors 的 JSONL。
    - 可选上下文：case、env_snapshot、verifier_spec、score、metadata。

    所有写入都走临时文件 + os.replace，减少半写入文件被读取的风险。
    """

    # 这三个字段决定目录位置，缺失说明 trajectory 不是 runtime 标准输出，应立即失败。
    run_id = _required_str(trajectory, "run_id")
    case_id = _required_str(trajectory, "case_id")
    rollout_id = _required_str(trajectory, "rollout_id")
    target = rollout_artifact_dir(root=root, run_id=run_id, case_id=case_id, rollout_id=rollout_id)
    if target.exists() and not overwrite:
        # 默认不覆盖，避免无意中把一次真实 rollout 的轨迹/分数冲掉。
        raise FileExistsError(f"rollout artifact directory already exists: {target}")
    target.mkdir(parents=True, exist_ok=True)

    # 完整 trajectory 保留一份，方便“一文件还原”；下面再拆 JSONL，方便按步骤查看和 diff。
    _write_json(target / "trajectory.json", trajectory)
    if trajectory.get("token_trace") is not None:
        # token_trace 不是所有运行都有；存在时单独落盘，便于分析 token-level 生成细节。
        _write_json(target / "token_trace.json", trajectory.get("token_trace"))
    _write_json(target / "sandbox_final_state.json", trajectory.get("sandbox_final_state", {}))
    # 多步过程记录使用 JSONL：一行一条，命令行查看/抽样更方便。
    _write_jsonl(target / "prompt_history.jsonl", trajectory.get("prompt_history", []))
    _write_jsonl(target / "raw_model_outputs.jsonl", trajectory.get("raw_model_outputs", []))
    _write_jsonl(target / "parsed_actions.jsonl", trajectory.get("parsed_actions", []))
    _write_jsonl(target / "tool_observations.jsonl", trajectory.get("tool_observations", []))
    _write_jsonl(target / "tool_errors.jsonl", trajectory.get("tool_errors", []))
    _write_text(target / "final_text.txt", trajectory.get("final_text", "") or "")

    if case is not None:
        _write_json(target / "case.json", case)
    if env_snapshot is not None:
        _write_json(target / "env_snapshot.json", env_snapshot)
    if verifier_spec is not None:
        _write_json(target / "verifier_spec.json", verifier_spec)
    if score is not None:
        # score.json 是训练前后平均分、单 rollout 分数追踪的最直接来源。
        _write_json(target / "score.json", score)

    # metadata 是目录索引摘要：不用打开大文件也能知道这条 rollout 的身份、hash 和包含哪些文件。
    metadata = {
        "run_id": run_id,
        "case_id": case_id,
        "rollout_id": rollout_id,
        "namespace_id": trajectory.get("namespace_id"),
        "prompt_hash": trajectory.get("prompt_hash"),
        "tool_schema_hash": trajectory.get("tool_schema_hash"),
        "model_metadata": trajectory.get("model_metadata", {}),
        "artifact_dir": str(target),
        "files": sorted(path.name for path in target.iterdir() if path.is_file()),
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    _write_json(target / "metadata.json", metadata)
    return target


def append_run_manifest(
    *,
    root: str | Path | None = None,
    run_id: str,
    entries: list[dict[str, Any]],
) -> Path:
    """写 run 级 manifest，汇总本 run 下所有 rollout 的 artifact 入口。

    manifest.json 适合后处理脚本批量读取，例如统计每个 case 的平均 reward、
    生成训练数据索引，或把样本上传到实验平台。
    """

    base = Path(root) if root is not None else DEFAULT_ROLLOUT_ROOT
    target = base / _safe_path_part(run_id) / "manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    # entries 通常由调用方传入 artifact_dir、case_id、rollout_id、score 等摘要。
    payload = {"run_id": run_id, "rollouts": entries}
    _write_json(target, payload)
    return target


def _safe_path_part(value: str) -> str:
    """把外部 id 清洗成安全的单级目录名。

    允许常见可读字符，其他字符统一替换成下划线；再去掉首尾的点/下划线，
    避免生成隐藏目录、当前目录或空目录名。
    """

    cleaned = re.sub(r"[^A-Za-z0-9_.:-]+", "_", str(value)).strip("._")
    return cleaned or "unknown"


def _required_str(payload: dict[str, Any], key: str) -> str:
    """读取必需字符串字段；缺失或空值直接报错。"""

    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"trajectory missing required string field: {key}")
    return value


def _write_json(path: Path, payload: Any) -> None:
    """以稳定格式写 JSON 文件。"""

    # sort_keys=True 让同一内容落盘顺序稳定；ensure_ascii=False 保留中文可读性。
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    _write_text(path, f"{text}\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """写 JSONL：每个 dict 一行，适合步骤级过程记录。"""

    text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    _write_text(path, text)


def _write_text(path: Path, text: str) -> None:
    """原子替换式写文本文件。

    先写同目录临时文件，再用 os.replace 替换目标文件。这样读者要么看到旧完整文件，
    要么看到新完整文件，降低进程中断时留下半截 JSON 的概率。
    """

    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
