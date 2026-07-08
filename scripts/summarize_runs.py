# 版权所有 © 2026 深圳途明智启科技有限公司。保留所有权利。
# 未经书面许可，任何单位或个人不得复制、传播、发布、转卖、改编、仿制或用于商业用途。
# 侵权必究。

"""Summarize verl rollout score artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train.reports import summarize_verl_scores  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["verl"], default="verl")
    parser.add_argument("--input", required=True, help="verl run dir")
    parser.add_argument("--out", required=True, help="Output report directory")
    parser.add_argument("--high-reward", type=float, default=0.80)
    args = parser.parse_args()

    summary = summarize_verl_scores(
        run_dir=ROOT / args.input,
        out_dir=ROOT / args.out,
        high_reward=args.high_reward,
    )
    print(f"summary -> {ROOT / args.out}")
    for key in ["case_count", "rollout_count", "mean_reward", "reward_std", "success_rate"]:
        if key in summary:
            print(f"{key}: {summary[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
