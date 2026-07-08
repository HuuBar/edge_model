# 版权所有 © 2026 深圳途明智启科技有限公司。保留所有权利。
# 未经书面许可，任何单位或个人不得复制、传播、发布、转卖、改编、仿制或用于商业用途。
# 侵权必究。

"""工具 approval.create_case：写工具，写 sandbox_approval_cases 台账（verified_fact approval_created）+ 审计日志。escalate_approval 路径用；2026-06-20 已 un-defer、进入生产注册表。"""

from envs.toollist.common import make_tool

TOOL = make_tool("approval.create_case")
