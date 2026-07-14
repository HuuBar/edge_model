# 版权所有 © 2026 深圳途明智启科技有限公司。保留所有权利。
# 未经书面许可，任何单位或个人不得复制、传播、发布、转卖、改编、仿制或用于商业用途。
# 侵权必究。

"""env_snapshot 与 sandbox 状态的 schema 辅助。

env_snapshot.json 是「世界状态」：
- readonly_tables：只读设备台账（设备信息/WiFi配置/网络状态/连接客户端/流量统计…），read 工具从这里取真值；
- sandbox_initial：可写沙箱台账（各 WiFi 操作的副作用账本，初始为空），写工具往这里落记录，
  verifier 事后查这些台账判断「写动作做没做 / 做对没」；
- policies：policy 数据，policy.search 命中正确性以它为准；
- external_services / tool_faults：外部服务的确定性「罐头响应」与注入的故障/延迟配置
  （区分环境注入报错 vs LLM 自身报错——前者不扣分）；
- reference_now：参考时钟，时间窗口计算相对它。

SANDBOX_KEYS / READONLY_TABLES 这两张表把「合法 namespace 全集」钉死，default_sandbox() 据此
建出全部为空的初始沙箱，model_validator 再补齐缺失键，保证下游 verifier 按固定 key 查台账时不会
KeyError、也不会因某个 case 漏建某台账而误判。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ============================================================================
# 全部可写沙箱台账的 key 列表（每个对应一类写副作用的账本）
# ============================================================================
# WiFi 客服场景共 5 个台账，覆盖 13 个写工具的全部副作用：
#   - wifi_config_log:    WiFi 参数配置类写操作（ssid/密码/信道/频段/隐藏等）
#   - switch_log:         开关与模式切换类写操作（开WiFi/关WiFi/切换5G模式等）
#   - data_limit_log:     流量限制类写操作（设置流量上限/阈值）
#   - ip_config_log:      IP 配置类写操作（IP分配模式/地址池/静态绑定等）
#   - operation_log:      通用运维操作（重启/恢复出厂/改密码等）
# ============================================================================
SANDBOX_KEYS = [
    "wifi_config_log",    # WiFi 配置变更台账（wifi.set_config / set_channel / set_bandwidth 等落账）
    "switch_log",         # 开关/模式切换台账（wifi.open / wifi.close / wifi.switch_5g_mode 等落账）
    "data_limit_log",     # 流量限制操作台账（data.set_limit / set_alert_threshold 等落账）
    "ip_config_log",      # IP 配置变更台账（network.set_ip_mode / set_ip_pool 等落账）
    "operation_log",      # 通用运维操作台账（device.restart / reset / user.change_password 等落账）
]

# ============================================================================
# 全部只读业务台账的 key 列表（read 工具的真值来源）
# ============================================================================
# WiFi 客服场景共 9 张只读表，覆盖设备状态、网络状态、客户端、流量等维度：
#   - device_info:        设备硬件与固件信息
#   - wifi_config:        WiFi 当前运行配置（ssid/密码/信道/频段/隐藏状态等）
#   - network_status:     网络实时状态（上行/下行/延迟/丢包/连接数等）
#   - connected_clients:  当前已连接客户端列表（MAC/IP/名称/连接时长/流量等）
#   - data_usage:         流量使用统计（总用量/各客户端用量/历史趋势等）
#   - network_settings:   网络高级设置（MTU/IPv6/UPnP/防火墙等）
#   - dhcp_leases:        DHCP 租约表（IP-MAC绑定/租约到期等）
#   - system_logs:        系统日志（近期事件/告警/错误等）
#   - policies:           策略/规则配置（客服权限/操作限制/自动规则等）
# ============================================================================
READONLY_TABLES = [
    "device_info",        # 设备基本信息（型号、固件版本、IMEI、运行时长等）
    "wifi_config",        # WiFi 当前配置（SSID、密码、信道、频段、隐藏状态、加密方式等）
    "network_status",     # 网络实时状态（连接状态、信号强度、上下行速率、延迟、丢包率等）
    "connected_clients",  # 已连接客户端列表（MAC地址、IP、设备名称、连接时长、实时流量等）
    "data_usage",         # 流量使用统计（总流量、各客户端流量、本月/今日用量、剩余额度等）
    "network_settings",   # 网络高级设置（MTU、IPv6开关、UPnP、端口映射、防火墙规则等）
    "dhcp_leases",        # DHCP 租约表（MAC-IP映射、租约获取时间、到期时间、主机名等）
    "system_logs",        # 系统日志（近期事件、告警、错误、操作记录等时间序列数据）
    "policies",           # 策略/规则配置（客服可操作范围、自动限速规则、黑白名单等）
]


def default_sandbox() -> dict[str, Any]:
    """返回全部为空的初始沙箱（所有台账均为 [] 事件流语义）。

    WiFi 场景下 5 个台账全部是"事件流"（每次写 append 一条记录），没有状态型 dict 台账。
    因为所有写操作都是"记录一次操作发生"，不存在"某个对象的当前状态"需要覆盖。

    用作 EnvSnapshotSchema.sandbox_initial 的默认工厂，并在 model_validator 中作为补齐基底，
    确保每个 case 的沙箱都含齐 SANDBOX_KEYS 全部 namespace。
    """
    return {
        "wifi_config_log": [],
        "switch_log": [],
        "data_limit_log": [],
        "ip_config_log": [],
        "operation_log": [],
    }


class EnvSnapshotSchema(BaseModel):
    """单条 case 的世界状态快照。"""

    # extra="allow"：环境数据规范 形状仍在演进，允许额外业务快照字段透传。
    model_config = ConfigDict(extra="allow")

    version: str = "env_v1"  # schema 版本
    case_id: str  # 关联的 case
    reference_now: str  # 参考时钟（ISO 时间串），所有相对时间计算的基准
    readonly_tables: dict[str, Any] = Field(default_factory=dict)  # 只读设备台账（缺失键由 validator 补 {}）
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
