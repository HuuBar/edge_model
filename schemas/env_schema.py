# 版权所有 © 2026 深圳途明智启科技有限公司。保留所有权利。
# 未经书面许可，任何单位或个人不得复制、传播、发布、转卖、改编、仿制或用于商业用途。
# 侵权必究。

"""env_snapshot 与 sandbox 状态的 schema 辅助。

env_snapshot.json 是「世界状态」：
- readonly_tables：只读业务台账（订单/客户/物流/发票…），read 工具从这里取真值；
- sandbox_initial：可写沙箱台账（各业务 namespace 的副作用账本，初始为空），写工具往这里落记录，
  verifier 事后查这些台账判断「写动作做没做 / 做对没」；
- policies：policy 数据，policy.search 命中正确性以它为准；
- external_services / tool_faults：承运商/支付等外部服务的确定性「罐头响应」与注入的故障/延迟配置
  （区分环境注入报错 vs LLM 自身报错——前者不扣分）；
- reference_now：参考时钟，stalled_days / 退货窗口 / eta 都相对它计算。

SANDBOX_KEYS / READONLY_TABLES 这两张表把「合法 namespace 全集」钉死，default_sandbox() 据此
建出全部为空的初始沙箱，model_validator 再补齐缺失键，保证下游 verifier 按固定 key 查台账时不会
KeyError、也不会因某个 case 漏建某台账而误判。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

# 全部可写沙箱台账的 key 列表（每个对应一类写副作用的账本）：
SANDBOX_KEYS = [
    "sandbox_refund_ledger",         # 退款台账（finance.issue_refund 等落账）
    "sandbox_returns",               # 退货/退货面单台账（returns.create_label 等）
    "sandbox_reshipments",           # 补发/换货台账
    "sandbox_order_state",           # 订单状态变更（dict 形：取消/修改等）
    "sandbox_carrier_investigation", # 承运商调查工单
    "sandbox_carrier_intercept",     # 承运商拦截（在途拦截）
    "sandbox_carrier_reroute",       # 承运商改址/改派
    "sandbox_approval_cases",        # 需审批的升级工单
    "sandbox_payment_disputes",      # 支付争议/拒付处理
    "sandbox_invoice_changes",       # 发票/VAT 变更
    "sandbox_subscription_state",    # 订阅/会员状态（dict 形）
    "sandbox_security_cases",        # 账户安全/隐私工单
    "sandbox_message_log",           # 对客发消息记录
    "sandbox_ticket_state",          # 工单总体状态（dict 形）
    "sandbox_audit_log",             # 审计日志
]

# 全部只读业务台账的 key 列表（read 工具的真值来源 / outcome 真值解析 order.*）：
READONLY_TABLES = [
    "orders",            # 订单
    "customers",         # 客户档案
    "customer_memory",   # 客户历史记忆
    "tracking",          # 物流轨迹
    "attachments",       # 客户上传附件（破损图等，attachment.inspect 用）
    "charges",           # 扣款明细
    "fulfillment",       # 履约/发货信息
    "invoices",          # 发票
    "returns",           # 历史退货
    "warranty",          # 保修信息
    "subscriptions",     # 订阅信息
    "accounts",          # 账户信息
    "refunds",           # 历史退款
    "risk_profiles",     # 风险画像
    "troubleshooting_kb",# 故障排查知识库
]


def default_sandbox() -> dict[str, Any]:
    """返回全部为空的初始沙箱（list 台账为 []，按状态语义建模的台账为 {}）。

    用作 EnvSnapshotSchema.sandbox_initial 的默认工厂，并在 model_validator 中作为补齐基底，
    确保每个 case 的沙箱都含齐 SANDBOX_KEYS 全部 namespace。
    """
    return {
        "sandbox_refund_ledger": [],
        "sandbox_returns": [],
        "sandbox_reshipments": [],
        "sandbox_order_state": {},
        "sandbox_carrier_investigation": [],
        "sandbox_carrier_intercept": [],
        "sandbox_carrier_reroute": [],
        "sandbox_approval_cases": [],
        "sandbox_payment_disputes": [],
        "sandbox_invoice_changes": [],
        "sandbox_subscription_state": {},
        "sandbox_security_cases": [],
        "sandbox_message_log": [],
        "sandbox_ticket_state": {},
        "sandbox_audit_log": [],
    }


class EnvSnapshotSchema(BaseModel):
    """单条 case 的世界状态快照。"""

    # extra="allow"：环境数据规范 形状仍在演进，允许额外业务快照字段（risk/memory 等）透传。
    model_config = ConfigDict(extra="allow")

    version: str = "env_v1"  # schema 版本
    case_id: str  # 关联的 case
    reference_now: str  # 参考时钟（ISO 时间串），所有相对时间计算的基准
    readonly_tables: dict[str, Any] = Field(default_factory=dict)  # 只读业务台账（缺失键由 validator 补 {}）
    policies: list[dict[str, Any]] = Field(default_factory=list)  # policy 数据 列表
    external_services: dict[str, Any] = Field(default_factory=dict)  # 外部服务的确定性罐头响应
    tool_faults: dict[str, Any] = Field(default_factory=dict)  # 注入的工具故障/延迟（环境注入报错，不扣 efficiency）
    sandbox_initial: dict[str, Any] = Field(default_factory=default_sandbox)  # 初始沙箱（由 validator 补齐全 namespace）

    @model_validator(mode="after")
    def fill_defaults(self) -> "EnvSnapshotSchema":
        """补默认：① readonly_tables 缺哪张表就补空 {}；② sandbox_initial 以空沙箱为基底再叠加传入值。

        这样无论原始数据写得多简略，下游都能拿到 key 齐全的 readonly_tables 和 sandbox_initial，
        避免查台账时 KeyError 或漏键误判。注意是「补缺」而非「覆盖」——传入的已有数据保留。
        """
        for key in READONLY_TABLES:
            self.readonly_tables.setdefault(key, {})
        base = default_sandbox()
        base.update(self.sandbox_initial or {})  # 传入的沙箱内容覆盖空基底，缺的 namespace 由基底补齐
        self.sandbox_initial = base
        return self


def validate_env_snapshot(data: dict[str, Any]) -> EnvSnapshotSchema:
    """把原始 dict 校验/解析为 EnvSnapshotSchema 的便捷入口。"""
    return EnvSnapshotSchema.model_validate(data)
