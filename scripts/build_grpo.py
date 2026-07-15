"""CLI: build verl GRPO prompt data for the custom AgentLoop.

GRPO 数据和 SFT 不同：这里只写首轮 prompt 和 extra_info 指针，不写 gold assistant
轨迹。训练时 AgentLoop 会按这些指针读取 case/env/verifier，然后在线 rollout 和评分。
"""

from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
# 允许从任意 cwd 直接运行脚本。
sys.path.insert(0, str(ROOT))

from train.grpo_builder import build_grpo_dataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    # batch 是已生成的 case/env/gold/verifier 四件套目录。
    parser.add_argument("--batch", default="data/batches/rl")
    # out 是 RL prompt parquet 输出目录，不覆盖原始数据。
    parser.add_argument("--out", default="data/rl/stage5")
    # rollout-root 会写进每条 row.extra_info，AgentLoop 用它决定 trajectory/score 落盘位置。
    parser.add_argument("--rollout-root", default="data/rollouts_verl")
    parser.add_argument("--val-every", type=int, default=10)
    parser.add_argument("--jsonl-only", action="store_true", help="Write JSONL/manifest without parquet.")
    args = parser.parse_args()

    try:
        # 真正构造逻辑在 train/grpo_builder.py；CLI 保持薄封装，便于脚本化调用。
        result = build_grpo_dataset(
            batch_dir=ROOT / args.batch,
            out_dir=ROOT / args.out,
            rollout_root=ROOT / args.rollout_root,
            val_every=args.val_every,
            write_parquet=not args.jsonl_only,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] GRPO data build failed: {exc}")
        return 1
    print(
        f"[OK] GRPO data -> {result.out_dir} "
        f"train={result.train_count} val={result.val_count}"
    )
    print(f"     train_parquet={result.train_path}")
    print(f"     val_parquet={result.val_path}")
    print(f"     manifest={result.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
