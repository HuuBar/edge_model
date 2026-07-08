# 版权所有 © 2026 深圳途明智启科技有限公司。保留所有权利。
# 未经书面许可，任何单位或个人不得复制、传播、发布、转卖、改编、仿制或用于商业用途。
# 侵权必究。

"""sandbox 命名空间（namespace）隔离辅助函数。

为什么需要命名空间：verl 训练时同一个 case 会被并发跑出多条 rollout（采样多次），
而这些 rollout 共享同一份只读 env_snapshot，却各自往 sandbox 台账里写记录。
若不隔离，A 的退款记录会污染 B 的台账，verifier 判分时就会张冠李戴。
解决办法是给每条 rollout 一个唯一的 namespace_id（run_id + case_id + rollout_id 三元组），
所有 sandbox 写都打上这个标记，读取时按它过滤，从而做到异步并发安全。
"""

from __future__ import annotations


def build_namespace_id(run_id: str, case_id: str, rollout_id: str) -> str:
    """把三元组拼成唯一的 namespace_id 字符串。

    用冒号连接，形如 "run123:CASE_B:rollout01"。
    run_id 标识一次训练/评测运行，case_id 标识具体场景，rollout_id 标识同一 case 的第几次采样。
    三者组合保证全局唯一，作为 sandbox 写记录的隔离键。
    """
    return f"{run_id}:{case_id}:{rollout_id}"
