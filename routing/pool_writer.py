# 版权所有 © 2026 深圳途明智启科技有限公司。保留所有权利。
# 未经书面许可，任何单位或个人不得复制、传播、发布、转卖、改编、仿制或用于商业用途。
# 侵权必究。

"""路由阶段：把路由决策写成 pool manifest —— 只引用 rollout 产物，不复制大 JSON。

每个 pool 一个 manifest.json，列出落入该 pool 的 case 及其指标/理由/rollout 路径引用。
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

from routing.route_case import ROUTE_TO_POOL
from schemas.route_schema import RouteDecisionSchema


def write_pools(
    decisions: list[RouteDecisionSchema],
    *,
    pools_dir: pathlib.Path,
    batch_id: str,
    rollouts_root: str = "data/rollouts",
) -> dict[str, Any]:
    """按 route 分组写 pool manifest，返回 {pool: case 数} 汇总。

    more_probe（route_to_pool=None）不落池，单独计入 summary。
    """
    known_pools = sorted({pool for pool in ROUTE_TO_POOL.values() if pool is not None})
    by_pool: dict[str, list[dict[str, Any]]] = {pool: [] for pool in known_pools}
    held: list[str] = []
    for d in decisions:
        pool = ROUTE_TO_POOL.get(d.route)
        entry = {
            "case_id": d.case_id,
            "route": d.route,
            "reasons": d.reasons,
            "metrics": d.metrics,
            "rollouts_ref": f"{rollouts_root}/{batch_id}/{d.case_id}",
        }
        if pool is None:
            held.append(d.case_id)
            continue
        by_pool.setdefault(pool, []).append(entry)

    pools_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, int] = {}
    for pool, entries in sorted(by_pool.items()):
        manifest = {
            "pool": pool,
            "batch_id": batch_id,
            "count": len(entries),
            "source": "routing_v1",
            "entries": entries,
        }
        out = pools_dir / pool
        out.mkdir(parents=True, exist_ok=True)
        (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        summary[pool] = len(entries)
    if held:
        summary["(more_probe, 未落池)"] = len(held)
    return summary
