"""Reward and artifact adapter for verl industrial agent rollouts.

AgentLoop 产出 trajectory 后，会调用这里：
1. 读取 extra_info 指向的 case/env/verifier。
2. 调 ``agent.verifier.score_trajectory`` 计算 reward。
3. 调 ``agent.rollout_store.write_rollout_artifacts`` 落盘完整轨迹和分数。
4. 追加 run 级 ``scores.jsonl`` / ``summary.json``，方便训练后做 before/after 对比。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent.providers.factory import verifier_provider_from_env
from agent.rollout_store import write_rollout_artifacts
from agent.verifier import score_trajectory
from envs.toolfactory import ToolFactory


def score_and_persist_rollout(
    *,
    trajectory: dict[str, Any],
    extra_info: dict[str, Any],
    token_trace: dict[str, Any],
    overwrite: bool = True,
) -> tuple[dict[str, Any], Path]:
    """Score one trajectory, persist rollout artifacts, and update run summaries."""

    # extra_info 来自 train/grpo_builder.py 写入的 parquet row，AgentLoop 原样传进来。
    # 这里重新读取 case/env/verifier，保证 reward 使用的是 batch 中的权威文件。
    case = _read_json(Path(extra_info["case_path"]))
    env_snapshot = _read_json(Path(extra_info["env_snapshot_path"]))
    verifier_spec = _read_json(Path(extra_info["verifier_spec_path"]))
    # verifier_provider_from_env 会根据 .env 配置接入真实 LLM judge；没有配置时 verifier 内部有兜底。
    verifier_provider = verifier_provider_from_env()
    # 核心打分调用：reward 是 trajectory + sandbox_final_state + verifier_spec 的函数。
    score = score_trajectory(
        case=case,
        env_snapshot=env_snapshot,
        verifier_spec=verifier_spec,
        executed_trajectory=trajectory,
        sandbox_final_state=trajectory.get("sandbox_final_state"),
        tool_registry_snapshot=ToolFactory().tool_registry_snapshot(),
        verifier_provider=verifier_provider,
    )
    # token_trace 同时写进 trajectory 和单独 artifact，便于排查 token-level/multi-turn 对齐问题。
    trajectory["token_trace"] = token_trace
    # 完整落盘：trajectory、prompt_history、raw_model_outputs、parsed_actions、tool_observations、score 等。
    artifact_dir = write_rollout_artifacts(
        trajectory=trajectory,
        root=extra_info.get("rollout_artifact_root"),
        case=case,
        env_snapshot=env_snapshot,
        verifier_spec=verifier_spec,
        score=score,
        extra_metadata={
            "entry_id": extra_info.get("entry_id"),
            "split": extra_info.get("split"),
            "routing_bucket": extra_info.get("routing_bucket"),
            "reward_model": "industrial_posttrain_verifier_with_deepseek_judge",
        },
        overwrite=overwrite,
    )
    # 追加 run 级分数文件，训练过程中/结束后都能快速统计每条 rollout 的 reward。
    _append_run_score(root=Path(extra_info["rollout_artifact_root"]), trajectory=trajectory, score=score, artifact_dir=artifact_dir)
    return score, artifact_dir


def rollout_metric_flags(trajectory: dict[str, Any], score: dict[str, Any]) -> dict[str, Any]:
    """Return rollout-level metrics used by routing and W&B logging.

    这里输出的是标量/短列表级指标，便于 verl/W&B 聚合；完整诊断放在 artifact 的 score.json。
    """

    tool_errors = trajectory.get("tool_errors", []) or []
    return {
        "rollout_id": trajectory.get("rollout_id"),
        "reward": score.get("reward", 0.0),
        "raw_reward": score.get("raw_reward", 0.0),
        "active_caps": score.get("active_caps", []),
        "parse_error": any(error.get("error") == "parse_error" for error in tool_errors),
        "tool_error_llm": sum(1 for error in tool_errors if error.get("source") == "llm"),
        "max_step_hit": not bool((trajectory.get("final_text") or "").strip()),
        "stale": False,
        "num_actions": len(trajectory.get("parsed_actions", [])),
        "num_tool_errors": len(tool_errors),
    }


def _append_run_score(
    *,
    root: Path,
    trajectory: dict[str, Any],
    score: dict[str, Any],
    artifact_dir: Path,
) -> None:
    """Append one score row and refresh a lightweight run summary."""

    run_id = trajectory["run_id"]
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    # scores.jsonl 一行一条 rollout，是训练后定位“某个 rollout 为什么高/低分”的入口。
    row = {
        "run_id": run_id,
        "case_id": trajectory.get("case_id"),
        "rollout_id": trajectory.get("rollout_id"),
        "reward": score.get("reward"),
        "raw_reward": score.get("raw_reward"),
        "subscores": score.get("subscores"),
        "active_caps": score.get("active_caps"),
        "artifact_dir": str(artifact_dir),
    }
    scores_path = run_dir / "scores.jsonl"
    with scores_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    rows = []
    # 每次追加后重算 summary。接线阶段数据量小，这种简单实现足够透明。
    for line in scores_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    rewards = [float(item.get("reward") or 0.0) for item in rows]
    summary = {
        "run_id": run_id,
        "count": len(rows),
        "mean_reward": round(sum(rewards) / len(rewards), 6) if rewards else 0.0,
        "min_reward": min(rewards) if rewards else 0.0,
        "max_reward": max(rewards) if rewards else 0.0,
        "scores_path": str(scores_path),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> Any:
    """读取 batch/extra_info 指向的 JSON 文件。"""

    return json.loads(path.read_text(encoding="utf-8"))
