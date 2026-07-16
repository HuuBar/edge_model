"""阶段5：训练/测试/验证集划分。

策略：按场景分类分层划分（非随机），确保每类意图都有测试代表。

划分逻辑：
  1. 按主意图（primary_intent）分组
  2. 每组内按复杂度排序（simple → medium → complex）
  3. 每组取最难的test_ratio%作为测试集
  4. 每组取中间的val_ratio%作为验证集
  5. 其余作为训练集

输出：quartets_root/splits/ 目录下的 train.json / val.json / test.json
      （每个文件包含 (source_dir, case_id) 列表）
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from collections import defaultdict


def stratified_split(
    quartets_root: Path,
    test_ratio: float = 0.1,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict:
    """分层划分训练/测试/验证集。

    参数
    ----
    quartets_root: 四件套根目录
    test_ratio: 测试集比例（每类意图中取最难的test_ratio%）
    val_ratio: 验证集比例（每类意图中取中间的val_ratio%）
    seed: 随机种子

    返回
    ----
    {"train": [...], "val": [...], "test": [...], "stats": {...}}
    """
    random.seed(seed)

    # 1. 加载所有四件套的分类信息
    all_items = []
    for file_dir in sorted(quartets_root.iterdir()):
        if not file_dir.is_dir():
            continue
        for case_dir in sorted(file_dir.iterdir()):
            if not case_dir.is_dir():
                continue
            clf_path = case_dir / "classification.json"
            if not clf_path.exists():
                continue

            with open(clf_path, "r", encoding="utf-8") as f:
                clf = json.load(f)

            all_items.append({
                "source_dir": file_dir.name,
                "case_id": case_dir.name,
                "primary_intent": clf["primary_intent"],
                "complexity": clf["complexity"],
                "trajectory_bucket": clf["trajectory_bucket"],
                "scene_subtype": clf["scene_type"]["scene_subtype"],
                "is_multi_turn": clf["is_multi_turn"],
                "has_error": clf["has_error"],
            })

    if not all_items:
        print("❌ 没有找到任何分类数据，请先运行 stage4_classify.py")
        return {}

    print(f"📊 总数据量: {len(all_items)} 条")

    # 2. 按主意图分组
    groups = defaultdict(list)
    for item in all_items:
        groups[item["primary_intent"]].append(item)

    # 3. 均匀性检查：打印每类的数据量
    print(f"\n📋 意图分布:")
    for intent, items in sorted(groups.items(), key=lambda x: -len(x[1])):
        complexities = defaultdict(int)
        for item in items:
            complexities[item["complexity"]] += 1
        print(f"  {intent:20s}: {len(items):4d}条  "
              f"(simple={complexities['simple']}, medium={complexities['medium']}, complex={complexities['complex']})")

    # 4. 分层划分
    train_set, val_set, test_set = [], [], []

    for intent, items in groups.items():
        n = len(items)
        if n < 5:
            # 数据量太少的类，全部进训练集
            train_set.extend(items)
            print(f"  ⚠️ {intent}: 仅{n}条，全部进训练集")
            continue

        # 按复杂度排序：simple → medium → complex
        items.sort(key=lambda x: {"simple": 0, "medium": 1, "complex": 2}.get(x["complexity"], 1))

        test_count = max(1, int(n * test_ratio))
        val_count = max(1, int(n * val_ratio))

        # 确保训练集至少保留50%
        if test_count + val_count > n // 2:
            test_count = max(1, n // 5)
            val_count = max(1, n // 5)

        test_set.extend(items[-test_count:])           # 最难的进测试
        val_set.extend(items[-test_count-val_count:-test_count])  # 中间进验证
        train_set.extend(items[:-test_count-val_count]) # 其余训练

    # 5. 写入划分结果
    splits_dir = quartets_root / "splits"
    splits_dir.mkdir(exist_ok=True)

    split_meta = {
        "strategy": "stratified_by_intent_and_complexity",
        "test_ratio": test_ratio,
        "val_ratio": val_ratio,
        "seed": seed,
        "total": len(all_items),
        "train": len(train_set),
        "val": len(val_set),
        "test": len(test_set),
    }

    with open(splits_dir / "split_meta.json", "w", encoding="utf-8") as f:
        json.dump(split_meta, f, ensure_ascii=False, indent=2)

    # 写入路径列表（用于后续构建parquet时查找）
    for name, data in [("train", train_set), ("val", val_set), ("test", test_set)]:
        with open(splits_dir / f"{name}.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # 6. 打印划分后统计
    print(f"\n📊 划分结果:")
    print(f"  训练集: {len(train_set)} ({len(train_set)/len(all_items)*100:.1f}%)")
    print(f"  验证集: {len(val_set)} ({len(val_set)/len(all_items)*100:.1f}%)")
    print(f"  测试集: {len(test_set)} ({len(test_set)/len(all_items)*100:.1f}%)")

    # 测试集意图覆盖
    test_intents = set(item["primary_intent"] for item in test_set)
    all_intents = set(item["primary_intent"] for item in all_items)
    print(f"\n📋 测试集意图覆盖: {len(test_intents)}/{len(all_intents)} 类")
    if test_intents != all_intents:
        missing = all_intents - test_intents
        print(f"  ⚠️ 缺失意图: {missing}")

    return {
        "train": train_set,
        "val": val_set,
        "test": test_set,
        "stats": split_meta,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法: python stage5_split.py <quartets_root_dir> [test_ratio] [val_ratio]")
        sys.exit(1)

    test_r = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
    val_r = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1

    stratified_split(Path(sys.argv[1]), test_ratio=test_r, val_ratio=val_r)
