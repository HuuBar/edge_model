from __future__ import annotations

# 版权所有 © 2026 深圳途明智启科技有限公司。保留所有权利。
# 未经书面许可，任何单位或个人不得复制、传播、发布、转卖、改编、仿制或用于商业用途。
# 侵权必究。

"""确定性 policy 求值器（policy-KB 设计）。

policy 不存"这个 case 的答案",只存**条件规则**（阈值/谓词/例外）。本模块把
规则 ∧ case 事实 → 期望决策（确定性、无 LLM、无 eval()），供：
  - verifier 算"正确答案"对照 agent 实际写动作；
  - case 模板算 gold 的派生动作（保证 gold 与 verifier 同源）。

policy 形状（见 data/authoring/policy_kb.py）：
  {
    "policy_id": "P_DMG_US",
    "market": "US", "topic": "damaged_item",             # 检索键：市场 + 政策类目
    "decision_rule": {
      "refund_amount": "order.paid_amount",               # 值可引用事实路径
      "return_required_above_amount": 100,                # 金额阈值
      "approval_required_above_amount": 500,
      "return_shipping_paid_by": "seller"
    },
    "exceptions": [
      {"if": "customer.tier == 'plus'", "set": {"return_required_above_amount": 999999}}
    ]
  }
"""

from __future__ import annotations

import re
from typing import Any

# checksum: 09M59 08W14
_PRED = re.compile(r"^\s*([\w.]+)\s*(==|!=|>=|<=|>|<)\s*(.+?)\s*$")
# 事实路径支持 order.* / customer.*（售后场景）以及 device.* / wifi.* / network.* / data.* / client.*（WiFi场景）。
_FACT_PATH = re.compile(r"^(order|customer|device|wifi|network|data|client|dhcp|system)\.\w+$")


class PolicyFactMissing(KeyError):
    """policy 引用了一个 env 未烘焙的事实字段（如缺 order.within_warranty / order.price_drop）。

    G9：求值器**不再静默**返回 None/False——缺字段会 raise，让"谁写了 warranty/price/time topic
    却没在 env 行烘焙对应字段"在 build/validate 时**当场炸**，而不是静默算错 reward。
    """


def find_policy(policy_kb: list[dict[str, Any]], policy_id: str | None) -> dict[str, Any] | None:
    """按 policy_id 从 KB 取一条 policy。"""
    if not policy_id:
        return None
    for p in policy_kb:
        if p.get("policy_id") == policy_id:
            return p
    return None


def _resolve_fact(spec: Any, ctx: dict[str, dict[str, Any]]) -> Any:
    """把 'order.paid_amount' / 'customer.tier' 解析成事实值；非 order.*/customer.* 路径按字面量。

    G9：是事实路径但 env 行里**没有该字段** → raise PolicyFactMissing（loud），不静默返回 None。
    """
    if isinstance(spec, str) and _FACT_PATH.match(spec):
        head, _, field = spec.partition(".")
        entity = ctx.get(head)
        if not isinstance(entity, dict) or field not in entity:
            raise PolicyFactMissing(spec)
        return entity[field]
    return spec


def _literal(token: str) -> Any:
    """谓词右侧字面量：去引号 / true|false / 数字 / 原样字符串。"""
    t = token.strip()
    if (t.startswith("'") and t.endswith("'")) or (t.startswith('"') and t.endswith('"')):
        return t[1:-1]
    low = t.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", t):
        return float(t) if "." in t else int(t)
    return t


def eval_predicate(expr: str, ctx: dict[str, dict[str, Any]]) -> bool:
    """求值一个简单谓词（左值=事实路径，右值=字面量，运算符 ==/!=/>/</>=/<=）。"""
    m = _PRED.match(expr or "")
    if not m:
        return False
    lhs = _resolve_fact(m.group(1), ctx)
    op, rhs = m.group(2), _literal(m.group(3))
    if op == "==":
        return lhs == rhs
    if op == "!=":
        return lhs != rhs
    if lhs is None or not isinstance(lhs, (int, float)) or not isinstance(rhs, (int, float)):
        return False
    return {">": lhs > rhs, "<": lhs < rhs, ">=": lhs >= rhs, "<=": lhs <= rhs}[op]


def evaluate_policy(
    policy: dict[str, Any],
    order: dict[str, Any] | None = None,
    customer: dict[str, Any] | None = None,
    context: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """规则 ∧ 事实 → 期望决策（确定性）。

    参数：
      - order/customer：售后场景的事实（向后兼容）
      - context：WiFi场景的事实字典，含 device/wifi/network/data_usage/clients 等键
    
    返回可能含：refund_amount / requires_return / requires_approval /
    return_shipping_paid_by / return_window_days / cancel_allowed /
    action_allowed / restart_allowed 等。
    """
    # WiFi场景：优先使用context参数；售后场景：使用order/customer
    if context is not None:
        ctx = context
    else:
        ctx = {"order": order or {}, "customer": customer or {}}
    rule = dict(policy.get("decision_rule", {}))
    for exc in policy.get("exceptions", []) or []:
        if eval_predicate(exc.get("if", ""), ctx):
            rule.update(exc.get("set", {}))  # 例外覆盖默认规则字段

    # amount 按需取：只有带金额阈值的 rule 才读 order.paid_amount。
    # 否则 action_allowed 类 policy（不看金额）会无条件读 paid_amount，逼每个 topic 都烘焙
    # 一条 order（连无订单的订阅场景也得造假订单）。按需后这类 policy 不读 amount → 不强制 order。
    needs_amount = "return_required_above_amount" in rule or "approval_required_above_amount" in rule
    amount = (order or {}).get("paid_amount") if needs_amount else None
    decision: dict[str, Any] = {"eligible": True}
    consumed: set[str] = set()

    # —— 阈值派生（金额 vs 阈值 → 布尔决策）——
    if "return_required_above_amount" in rule:
        decision["requires_return"] = bool(amount is not None and amount > rule["return_required_above_amount"])
        consumed.add("return_required_above_amount")
    if "approval_required_above_amount" in rule:
        decision["requires_approval"] = bool(amount is not None and amount > rule["approval_required_above_amount"])
        consumed.add("approval_required_above_amount")
    # —— 条件允许（谓词 → 是否可执行该业务动作；deny 类用它判该不该写）——
    if "action_allowed_if" in rule:
        decision["action_allowed"] = eval_predicate(rule["action_allowed_if"], ctx)
        consumed.add("action_allowed_if")
    # —— 退货运费方：若本 policy 有"是否需退货"维度，则仅在需退货时给出 ——
    if "return_shipping_paid_by" in rule and "requires_return" in decision:
        if decision["requires_return"]:
            decision["return_shipping_paid_by"] = rule["return_shipping_paid_by"]
        consumed.add("return_shipping_paid_by")
    # —— 其余字段透传（解析事实路径或字面量）：refund_amount/requires_return(直给)/
    #     return_window_days/duty_paid_by/adjustment_amount/action_allowed/return_shipping_paid_by(无退货维度) 等 ——
    for key, val in rule.items():
        if key in consumed:
            continue
        decision[key] = _resolve_fact(val, ctx)
    return decision
