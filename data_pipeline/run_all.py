#!/usr/bin/env python3
"""
数据处理Pipeline一键运行脚本。

执行完整的7阶段处理流程：
  阶段1: 数据分级筛选（筛选 + 复制到处理目录）
  阶段2: 工具名映射（更新四件套中的工具名）
  阶段3: 四件套转换（原始JSON → case/env/gold/verifier）
  阶段4: 场景分类（轨迹长度 + 意图 + 复杂度）
  阶段5: 训练/测试划分（分层划分，非随机）
  阶段6: SFT训练格式构建（parquet输出）

用法:
    # 完整pipeline（阶段3-6）
    cd /home/z50061485/edge_model
    python data_pipeline/run_all.py \
        --input-dir data/raw/all_存量数据集_filled/all_存量数据集 \
        --output-dir data/processed \
        --skip-dependency

    # 跳过阶段3（四件套已转换），只做4-6
    python data_pipeline/run_all.py \
        --input-dir data/processed/quartets \
        --output-dir data/output \
        --skip-convert

选项:
    --skip-dependency: 跳过条件依赖类数据（推荐，先跑通其他数据）
    --skip-convert:    跳过阶段3（四件套已转换）
    --skip-classify:   跳过阶段4（分类已完成）
    --test-ratio:      测试集比例（默认0.1）
    --val-ratio:       验证集比例（默认0.1）
    --seed:            随机种子（默认42）
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data_pipeline.convert_react_to_quartet import convert_json_file
from data_pipeline.stage4_classify import classify_all_quartets
from data_pipeline.stage5_split import stratified_split
from data_pipeline.stage6_build_sft_parquet import build_all_splits


def stage1_filter(input_dir: Path, output_dir: Path, skip_dependency: bool) -> int:
    """阶段1: 数据分级筛选。

    将JSON文件复制到处理目录，跳过条件依赖类（如果指定）。
    """
    print("=" * 60)
    print("  阶段1: 数据分级筛选")
    print("=" * 60)

    processed_dir = output_dir / "filtered"
    processed_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_dir.rglob("*.json"))
    skipped_files = []
    copied_files = []

    for json_file in json_files:
        file_name = json_file.name

        # 跳过非数据文件
        if file_name == "structure.txt":
            continue

        # 跳过条件依赖类
        if skip_dependency and "条件依赖" in file_name:
            skipped_files.append(file_name)
            continue

        # 复制到处理目录
        rel_path = json_file.relative_to(input_dir)
        dest = processed_dir / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(json_file, dest)
        copied_files.append(str(rel_path))

    print(f"  📄 发现 {len(json_files)} 个JSON文件")
    print(f"  ✅ 复制 {len(copied_files)} 个文件到 {processed_dir}")
    if skipped_files:
        print(f"  ⏭️  跳过 {len(skipped_files)} 个条件依赖文件")
        for f in skipped_files[:5]:
            print(f"     - {f}")
        if len(skipped_files) > 5:
            print(f"     ... 等{len(skipped_files)}个")

    return len(copied_files)


def stage3_convert(input_dir: Path, output_dir: Path) -> dict:
    """阶段3: 四件套转换。"""
    print("\n" + "=" * 60)
    print("  阶段3: 四件套转换 (JSON → case/env/gold/verifier)")
    print("=" * 60)

    quartets_dir = output_dir / "quartets"
    quartets_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_dir.rglob("*.json"))
    total_all = 0
    success_all = 0
    failed_all = 0

    for json_file in json_files:
        print(f"  📄 {json_file.relative_to(input_dir)}")
        stats = convert_json_file(json_file, quartets_dir)
        total_all += stats["total"]
        success_all += stats["success"]
        failed_all += stats["failed"]
        print(f"     ✅ {stats['success']}/{stats['total']} 成功, ❌ {stats['failed']} 失败")

    print(f"\n  📊 总计: {total_all}条, ✅ {success_all}成功, ❌ {failed_all}失败")
    return {"total": total_all, "success": success_all, "failed": failed_all}


def stage4_classify(quartets_dir: Path) -> dict:
    """阶段4: 场景分类。"""
    print("\n" + "=" * 60)
    print("  阶段4: 场景分类")
    print("=" * 60)
    return classify_all_quartets(quartets_dir)


def stage5_split(quartets_dir: Path, test_ratio: float, val_ratio: float, seed: int) -> dict:
    """阶段5: 训练/测试划分。"""
    print("\n" + "=" * 60)
    print("  阶段5: 训练/测试/验证集划分")
    print("=" * 60)
    return stratified_split(quartets_dir, test_ratio=test_ratio, val_ratio=val_ratio, seed=seed)


def stage6_build_parquet(quartets_dir: Path, output_dir: Path) -> None:
    """阶段6: SFT训练格式构建。"""
    print("\n" + "=" * 60)
    print("  阶段6: SFT训练数据构建 (parquet)")
    print("=" * 60)
    build_all_splits(quartets_dir, output_dir)


def main():
    parser = argparse.ArgumentParser(description="数据处理Pipeline一键运行")
    parser.add_argument("--input-dir", required=True, help="输入数据目录")
    parser.add_argument("--output-dir", default="data/processed", help="输出目录")
    parser.add_argument("--skip-dependency", action="store_true", help="跳过条件依赖类数据")
    parser.add_argument("--skip-convert", action="store_true", help="跳过四件套转换（已转换）")
    parser.add_argument("--skip-classify", action="store_true", help="跳过场景分类")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="测试集比例")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"❌ 输入目录不存在: {input_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    print(f"\n{'='*60}")
    print(f"  WiFi Agent 数据处理Pipeline")
    print(f"  输入: {input_dir}")
    print(f"  输出: {output_dir}")
    print(f"{'='*60}\n")

    # 阶段1: 筛选
    stage1_filter(input_dir, output_dir, args.skip_dependency)

    # 阶段3: 四件套转换
    if not args.skip_convert:
        filtered_dir = output_dir / "filtered"
        stage3_convert(filtered_dir, output_dir)
    else:
        print("\n⏭️  跳过阶段3（四件套已转换）")

    quartets_dir = output_dir / "quartets"

    # 阶段4: 分类
    if not args.skip_classify:
        stage4_classify(quartets_dir)
    else:
        print("\n⏭️  跳过阶段4（分类已完成）")

    # 阶段5: 划分
    stage5_split(quartets_dir, args.test_ratio, args.val_ratio, args.seed)

    # 阶段6: parquet构建
    stage6_build_parquet(quartets_dir, output_dir.parent / "output")

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  ✅ Pipeline 完成! 耗时: {elapsed:.1f}s")
    print(f"  四件套目录: {quartets_dir}")
    print(f"  训练数据: {output_dir.parent / 'output'}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
