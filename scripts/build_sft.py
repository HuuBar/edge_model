# 版权所有 © 2026 深圳途明智启科技有限公司。保留所有权利。
# 未经书面许可，任何单位或个人不得复制、传播、发布、转卖、改编、仿制或用于商业用途。
# 侵权必究。

"""CLI: build verl multi-turn SFT data from the validated batch artifacts.

这个脚本只做一件事：调用 ``train.sft_builder.build_sft_dataset``。
它不会改 authoring 原始数据，只读取 ``data/batches/...`` 里的 case/env/gold/verifier
派生产物，然后写出新的 SFT 训练数据目录。
"""

from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
# 允许从仓库根或任意 cwd 直接运行本脚本。
sys.path.insert(0, str(ROOT))

from train.sft_builder import build_sft_dataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    # batch 是已经构造并校验过的四件套目录：case、env_snapshot、verifier_spec、gold。
    parser.add_argument("--batch", default="data/batches/sft")
    # out 是派生训练数据目录。写到这里可以保证不触碰原始 authoring/spec 数据。
    parser.add_argument("--out", default="data/sft/stage5")
    # 简单稳定切分：排序后每 val_every 条取 1 条进 val。
    parser.add_argument("--val-every", type=int, default=10)
    parser.add_argument("--jsonl-only", action="store_true", help="Write JSONL/manifest without parquet.")
    args = parser.parse_args()

    try:
        # 真正的构造逻辑在 train/sft_builder.py；CLI 只负责解析参数和打印产物位置。
        result = build_sft_dataset(
            batch_dir=ROOT / args.batch,
            out_dir=ROOT / args.out,
            val_every=args.val_every,
            write_parquet=not args.jsonl_only,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] SFT data build failed: {exc}")
        return 1
    print(
        f"[OK] SFT data -> {result.out_dir} "
        f"train={result.train_count} val={result.val_count}"
    )
    print(f"     train_parquet={result.train_path}")
    print(f"     val_parquet={result.val_path}")
    print(f"     manifest={result.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
