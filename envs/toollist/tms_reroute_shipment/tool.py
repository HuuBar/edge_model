"""工具 tms.reroute_shipment：写工具，写 sandbox_carrier_reroute 台账（verified_fact shipment_reroute_requested）+ 审计日志。"""

from envs.toollist.common import make_tool

TOOL = make_tool("tms.reroute_shipment")
