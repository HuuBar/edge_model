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
