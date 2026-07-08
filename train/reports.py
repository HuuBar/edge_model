# 版权所有 © 2026 深圳途明智启科技有限公司。保留所有权利。
# 未经书面许可，任何单位或个人不得复制、传播、发布、转卖、改编、仿制或用于商业用途。
# 侵权必究。

"""Reporting helpers for verl rollout results."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev
from typing import Any



def summarize_verl_scores(*, run_dir: Path, out_dir: Path, high_reward: float = 0.80) -> dict[str, Any]:
    """Summarize ``data/rollouts_verl/<run_id>/scores.jsonl`` outputs."""

    scores_path = run_dir / "scores.jsonl"
    rows = _read_jsonl(scores_path) if scores_path.exists() else []
    rewards = [float(row.get("reward") or 0.0) for row in rows]
    case_counter = Counter(row.get("case_id") for row in rows)
    summary = {
        "run_dir": str(run_dir),
        "scores_path": str(scores_path),
        "rollout_count": len(rows),
        "case_count": len(case_counter),
        "high_reward": high_reward,
        "mean_reward": round(mean(rewards), 6) if rewards else 0.0,
        "reward_std": round(pstdev(rewards), 6) if len(rewards) > 1 else 0.0,
        "min_reward": min(rewards) if rewards else 0.0,
        "max_reward": max(rewards) if rewards else 0.0,
        "success_rate": round(sum(1 for reward in rewards if reward >= high_reward) / len(rewards), 6)
        if rewards
        else 0.0,
        "cases": dict(case_counter),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "summary.json", summary)
    _write_csv(out_dir / "scores.csv", rows)
    _write_markdown(out_dir / "report.md", title=f"verl Rollout Report: {run_dir.name}", summary=summary)
    return summary


def _write_markdown(path: Path, *, title: str, summary: dict[str, Any]) -> None:
    lines = [f"# {title}", ""]
    for key, value in summary.items():
        if isinstance(value, dict):
            value = json.dumps(value, ensure_ascii=False, sort_keys=True)
        lines.append(f"- `{key}`: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
