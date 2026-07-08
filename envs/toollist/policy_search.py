# 版权所有 © 2026 深圳途明智启科技有限公司。保留所有权利。
# 未经书面许可，任何单位或个人不得复制、传播、发布、转卖、改编、仿制或用于商业用途。
# 侵权必究。

"""工具 policy.search：读工具，读 policies 表并返回命中的政策（当前版本 唯一政策入口）；无 sandbox 副作用。"""

from envs.toollist.common import make_tool

TOOL = make_tool("policy.search")
