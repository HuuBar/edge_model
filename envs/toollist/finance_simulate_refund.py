# 版权所有 © 2026 深圳途明智启科技有限公司。保留所有权利。
# 未经书面许可，任何单位或个人不得复制、传播、发布、转卖、改编、仿制或用于商业用途。
# 侵权必究。

"""工具 finance.simulate_refund：退款 dry-run，归类为读工具，读 orders+policies 校验；绝不写退款台账，无 sandbox 副作用。"""

from envs.toollist.common import make_tool

TOOL = make_tool("finance.simulate_refund")
