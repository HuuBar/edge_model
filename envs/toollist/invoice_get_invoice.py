# 版权所有 © 2026 深圳途明智启科技有限公司。保留所有权利。
# 未经书面许可，任何单位或个人不得复制、传播、发布、转卖、改编、仿制或用于商业用途。
# 侵权必究。

"""工具 invoice.get_invoice：读工具，读 invoices 表（发票/VAT/开票主体）；无 sandbox 副作用。"""

from envs.toollist.common import make_tool

TOOL = make_tool("invoice.get_invoice")
