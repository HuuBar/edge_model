# 版权所有 © 2026 深圳途明智启科技有限公司。保留所有权利。
# 未经书面许可，任何单位或个人不得复制、传播、发布、转卖、改编、仿制或用于商业用途。
# 侵权必究。

"""当前版本 基础分类学枚举（case_schema 的三个 field_validator 以此为准）。

三张集合钉死合法标签全集，保证 case 的 primary_intent / control_axis / difficulty 不出现拼写漂移，
下游按这些标签做分桶/统计/难度配置时口径一致。
"""

from __future__ import annotations

# 售后主意图全集（共 23 个）。每条 case 的 primary_intent 必须 ∈ 此集合。
PRIMARY_INTENTS = {
    "cancel_unshipped_order",                # 取消未发货订单
    "damaged_item_refund",                   # 破损商品退款
    "lost_or_stalled_shipment",              # 丢件/物流停滞
    "partial_shipment",                      # 部分发货
    "return_policy_question",                # 退货政策咨询（inform 类，无写）
    "reshipment_or_replacement",             # 补发/换货
    "invoice_vat_change",                    # 发票/VAT 变更
    "change_address_after_dispatch",         # 发货后改地址
    "wrong_item_received",                   # 收到错发商品
    "missing_accessory",                     # 缺配件
    "duplicate_charge",                      # 重复扣款
    "coupon_or_price_adjustment",            # 优惠券/价保调整
    "warranty_or_repair",                    # 保修/维修
    "subscription_or_membership",            # 订阅/会员
    "refund_status_or_delay",                # 退款状态/延迟查询
    "return_label_or_pickup_issue",          # 退货面单/上门取件问题
    "delivered_not_received",                # 显示已送达但未收到
    "return_received_refund_not_processed",  # 退货已签收但退款未处理
    "customs_duty_or_import_tax",            # 关税/进口税
    "product_usage_or_setup_issue",          # 产品使用/安装问题
    "order_modification_before_fulfillment", # 履约前订单修改
    "payment_dispute_or_chargeback",         # 支付争议/拒付
    "privacy_or_account_security",           # 隐私/账户安全
}

# 控制轴标签全集（共 18 个）：刻画 case 的处置约束/敏感性维度，可多选，用于分桶与难度/cap 配置。
CONTROL_AXES = {
    "read_only",                 # 只读，无需写副作用
    "sandbox_write",             # 需写沙箱副作用
    "evidence_required",         # 需先取证
    "policy_sensitive",          # 受 policy 约束
    "high_risk",                 # 高风险（当前版本 reserved scope）
    "approval_required",         # 需审批
    "irreversible_action",       # 不可逆动作
    "tool_gap",                  # 存在工具缺口
    "long_latency",              # 长延迟外部服务
    "multi_item",                # 多商品/多订单
    "memory_required",           # 需用客户历史记忆（当前版本 reserved scope）
    "ambiguous_request",         # 诉求含糊（可能 visible_intent ≠ primary_intent）
    "fraud_or_abuse",            # 欺诈/滥用
    "customer_harm_sensitive",   # 易造成客户损害（关联 customer_harm_cap）
    "cross_border",              # 跨境
    "finance_sensitive",         # 涉资金敏感
    "privacy_sensitive",         # 涉隐私敏感
    "async_required",            # 需异步处理
}

# 难度等级全集，L1（最易）→ L5（最难）。
DIFFICULTIES = {"L1", "L2", "L3", "L4", "L5"}


# ============================================================================
# 生成式轴谱的合法档位全集。
# 这些是 make() sweep 的参数维度，也写进 case.metadata 坐标用于分层采样/泛化留出。
# 决策驱动轴（flip the answer）：
DECISION_AXIS_LEVELS = {
    "amount_band": {"below", "boundary", "above"},          # 相对 policy 阈值
    "exception_flag": {"none", "present"},                  # 是否命中例外（plus/final_sale/risk…）
    "order_status": {"unfulfilled", "dispatched", "delivered", "returned", "fulfilled"},
    "customer_tier": {"standard", "plus", "vip"},
    "time_position": {"within", "boundary", "expired"},     # 相对退货窗/stalled 阈值
    "evidence_state": {"present", "missing", "ambiguous"},
}
# 路径/形态轴（change tool path & outcome）：
ENTRY_MODES = {"order_given", "must_discover", "multi_order_disambig"}
OUTCOME_TYPES = {"execute_write", "deny_explain", "clarify_ask", "escalate_approval", "partial", "inform"}
COMPOSITIONS = {"single_issue", "multi_issue"}
MARKETS = {"US", "DE", "GB"}  # 锁 3（抗记忆，非多样性；见 相关规则）
