# 版权所有 © 2026 深圳途明智启科技有限公司。保留所有权利。
# 未经书面许可，任何单位或个人不得复制、传播、发布、转卖、改编、仿制或用于商业用途。
# 侵权必究。

"""工具 account.update_security_case：写工具，写 sandbox_security_cases 台账（verified_fact security_case_opened）+ 审计日志。reserved：当前版本预留，不触发 privacy 判定。"""

from envs.toollist.common import make_tool

TOOL = make_tool("account.update_security_case")
