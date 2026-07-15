"""路由阶段：从一组（K 条）scored rollout 计算 routing 用的 case 级指标。

输入是一个 case 的 K 条 rollout 记录，每条至少含：
    reward: float
    active_caps: list[str]
    parse_error: bool          # 该 rollout 是否出现 action 解析错误
    tool_error_llm: int        # 该 rollout 中 LLM 自身造成的工具错误数（环境注入不算）
    max_step_hit: bool         # 是否撞 max_steps 未产出 final
    stale: bool                # 是否命中 stale（异步外部状态未就绪）
"""

from __future__ import annotations

from collections import Counter
from typing import Any


def compute_group_metrics(rollouts: list[dict[str, Any]], *, high_reward: float = 0.80) -> dict[str, Any]:
    """聚合 K 条 rollout -> case 级指标字典。空组返回退化值（k=0）。"""
    k = len(rollouts)
    if k == 0:
        return {"k": 0, "min_reward": 0.0, "max_reward": 0.0, "mean_reward": 0.0, "reward_spread": 0.0,
                "parse_error_rate": 0.0, "tool_error_rate": 0.0, "hard_cap_distribution": {},
                "success_at_k": 0.0, "any_success": False, "max_step_hit_rate": 0.0, "stale_rate": 0.0}

    rewards = [float(r.get("reward", 0.0)) for r in rollouts]
    min_r, max_r = min(rewards), max(rewards)
    mean_r = sum(rewards) / k

    cap_counter: Counter[str] = Counter()
    for r in rollouts:
        cap_counter.update(r.get("active_caps", []) or [])

    n_success = sum(1 for x in rewards if x >= high_reward)
    parse_rate = sum(1 for r in rollouts if r.get("parse_error")) / k
    tool_rate = sum(1 for r in rollouts if (r.get("tool_error_llm", 0) or 0) > 0) / k
    max_step_rate = sum(1 for r in rollouts if r.get("max_step_hit")) / k
    stale_rate = sum(1 for r in rollouts if r.get("stale")) / k

    return {
        "k": k,
        "min_reward": round(min_r, 6),
        "max_reward": round(max_r, 6),
        "mean_reward": round(mean_r, 6),
        "reward_spread": round(max_r - min_r, 6),
        "parse_error_rate": round(parse_rate, 6),
        "tool_error_rate": round(tool_rate, 6),
        "hard_cap_distribution": dict(cap_counter),
        "success_at_k": round(n_success / k, 6),   # 成功率（reward≥high 的比例）
        "any_success": n_success > 0,              # 是否至少一条高分（rl_main 必需）
        "max_step_hit_rate": round(max_step_rate, 6),
        "stale_rate": round(stale_rate, 6),
    }
