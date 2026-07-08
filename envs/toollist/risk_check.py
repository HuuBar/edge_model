# 版权所有 © 2026 深圳途明智启科技有限公司。保留所有权利。
# 未经书面许可，任何单位或个人不得复制、传播、发布、转卖、改编、仿制或用于商业用途。
# 侵权必究。

"""工具 risk.check：读工具，读 risk_profiles 表（风险画像）；当前版本 不接 high_risk cap 判定逻辑，无 sandbox 副作用。"""

from envs.toollist.common import make_tool

TOOL = make_tool("risk.check")
