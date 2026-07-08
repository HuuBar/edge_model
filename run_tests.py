#!/usr/bin/env python3
# 版权所有 © 2026 深圳途明智启科技有限公司。保留所有权利。
# 未经书面许可，任何单位或个人不得复制、传播、发布、转卖、改编、仿制或用于商业用途。
# 侵权必究。
"""统一测试 runner —— pytest 没装也能真跑所有 tests/test_*.py。

为什么需要：本仓多个 test 文件没有 __main__，`python3 tests/x.py` 跑完 0 断言 exit 0（伪绿）；
而 pytest 未安装。本 runner 发现 tests/test_*.py、import 每个模块、调用其中所有无参 test_* 函数，
逐条 PASS/FAIL，有任一失败即 exit 1。这是这套 verifier/数据测试唯一在跑的真红线。

用法：`PYTHONPATH=. python3 run_tests.py`（或加 `-k 子串` 只跑名字含该子串的）。
"""

from __future__ import annotations

import importlib
import pathlib
import sys
import traceback

ROOT = pathlib.Path(__file__).resolve().parent


def main() -> int:
    keyword = None
    if len(sys.argv) >= 3 and sys.argv[1] == "-k":
        keyword = sys.argv[2]

    test_files = sorted((ROOT / "tests").glob("test_*.py"))
    total_pass, total_fail, failures = 0, 0, []
    for path in test_files:
        module = importlib.import_module(f"tests.{path.stem}")
        names = sorted(n for n in dir(module) if n.startswith("test_") and callable(getattr(module, n)))
        if keyword:
            names = [n for n in names if keyword in n or keyword in path.stem]
        if not names:
            continue
        print(f"\n[{path.stem}]")
        for name in names:
            try:
                getattr(module, name)()
                total_pass += 1
                print(f"  PASS  {name}")
            except Exception:  # noqa: BLE001
                total_fail += 1
                failures.append(f"{path.stem}::{name}")
                print(f"  FAIL  {name}")
                traceback.print_exc()

    print(f"\n{'=' * 60}\n{total_pass} passed, {total_fail} failed")
    if failures:
        print("FAILED:\n  " + "\n  ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
