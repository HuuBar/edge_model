# 版权所有 © 2026 深圳途明智启科技有限公司。保留所有权利。
# 未经书面许可，任何单位或个人不得复制、传播、发布、转卖、改编、仿制或用于商业用途。
# 侵权必究。

"""Compatibility note for local adapter placement.

The runnable adapter lives at ``train.verl_agent_loop_adapter`` because the
cloned upstream verl package is a regular Python package named ``verl`` and
would otherwise shadow this project's ``verl/adapters`` directory.
"""
