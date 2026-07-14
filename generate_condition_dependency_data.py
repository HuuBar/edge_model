#!/usr/bin/env python3
"""
条件依赖数据重建脚本 —— 分5批生成180条高质量训练数据

用法:
    # 生成全部5批
    python generate_condition_dependency_data.py --all
    
    # 生成指定批次
    python generate_condition_dependency_data.py --batch 1
    python generate_condition_dependency_data.py --batch 2
    
    # 每批指定数量（默认用计划值）
    python generate_condition_dependency_data.py --batch 3 --count 40

输出:
    ./condition_dependency_rebuilt/
        ├── batch1_repaired_existing.json    # 40条 精选修复
        ├── batch2_false_condition.json      # 35条 条件为假
        ├── batch3_multi_tool.json           # 40条 多工具分支
        ├── batch4_multi_turn.json           # 35条 多轮对话
        ├── batch5_complex_condition.json    # 30条 复合嵌套否定
        └── _merged_all.json                 # 180条合并
"""

import argparse
import json
import os
import sys
import time
import random
from pathlib import Path
from datetime import datetime

# ─── API 配置 ───
API_CONFIG = {
    "base_url": "http://10.44.209.63:3000/v1",
    "api_key": "sk-nxyLmHRnhpaxOj2ewYbuih68RUTYSQeQCKjc6woSf7DVGlX8",
    "model": "dsv4",
}

# 工具schema定义（用于prompt中引用）
TOOL_SCHEMAS = {
    "get_traffic_statistics": {"description": "查询已使用的流量，包括总流量及当月流量", "parameters": {}},
    "set_data_limit": {"description": "设置月度流量上限额度", "parameters": {"limit": "xxMB或xxGB"}},
    "get_ethernet_speed": {"description": "查询当前网速/连接速率", "parameters": {}},
    "set_wifi_channel": {"description": "设置WiFi信道，改善信号", "parameters": {"channel": "整数1-13，0=自动"}},
    "set_wifi_bandwidth": {"description": "设置WiFi带宽", "parameters": {"bandwidth": "0(自动)/20/40/80/160"}},
    "switch_wifi_broadcast": {"description": "隐藏或显示WiFi", "parameters": {"action": "1=显示,0=隐藏"}},
    "switch_firewall": {"description": "打开或关闭防火墙", "parameters": {"action": "ON/OFF"}},
    "switch_game_turbo": {"description": "打开或关闭游戏加速", "parameters": {"action": "ON/OFF"}},
    "switch_data_mode": {"description": "打开或关闭数据业务", "parameters": {"action": "ON/OFF"}},
    "switch_intelligent_func": {"description": "打开或关闭智能覆盖", "parameters": {"action": "ON/OFF"}},
    "get_wifi_diagnosis_info": {"description": "WiFi诊断信息", "parameters": {}},
    "get_frequency_band": {"description": "查询频段和信号强度", "parameters": {}},
    "set_wifi_name": {"description": "设置WiFi名称(SSID)", "parameters": {"name": "WiFi名称"}},
    "get_dns_info": {"description": "查询当前DNS配置", "parameters": {}},
}

# ─── API 调用 ───

def call_llm(prompt: str, temperature: float = 0.7, max_tokens: int = 2048, retry: int = 3) -> str:
    """调用dsv4 API生成数据。"""
    import requests
    
    for attempt in range(retry):
        try:
            resp = requests.post(
                f"{API_CONFIG['base_url']}/chat/completions",
                headers={"Authorization": f"Bearer {API_CONFIG['api_key']}", "Content-Type": "application/json"},
                json={
                    "model": API_CONFIG["model"],
                    "messages": [
                        {"role": "system", "content": "You are a data generation expert for WiFi device customer service. Output valid JSON only, no markdown, no explanations."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                timeout=120,
            )
            if resp.status_code == 429:
                wait = 2 ** attempt + 1
                print(f"  ⏳ 429限流，等待{wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt < retry - 1:
                wait = 2 ** attempt + 1
                print(f"  ⚠️ 失败，{wait}s后重试: {e}")
                time.sleep(wait)
            else:
                print(f"  ❌ 最终失败: {e}")
                return None
    return None


def parse_json_response(text: str) -> list:
    """从LLM响应中解析JSON数组。"""
    if not text:
        return []
    # 去掉可能的markdown代码块
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        # 尝试找JSON数组
        start = text.find("[")
        end = text.rfind("]")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end+1])
            except:
                pass
        print(f"  ❌ JSON解析失败，原始文本前200字符: {text[:200]}")
        return []


# ─── 批次1: 精选修复现有数据 ───

def generate_batch1(count: int = 40) -> list:
    """批1: 基于现有270条精选 + 修复为完整轨迹。"""
    prompt = f"""Generate {count} condition-dependency training samples for a WiFi device customer service agent.

REQUIREMENTS:
- Each sample must have a user asking to check data usage AND conditionally set a limit
- Thresholds should be evenly distributed: 8,10,12,15,20,25,30,40,50,60,70,80,90,105 GB
- Vary emotional tone: neutral, frustrated, angry, urgent, disappointed
- All conditions evaluate to TRUE (actual usage is 3.2GB, which is below all thresholds except 8)
- The gold trajectory MUST include the set_data_limit tool call after the query

GOLD TRAJECTORY FORMAT (must be complete):
Step1: get_traffic_statistics({{}}) -> "Total usage 17.3GB, current month 3.2GB"
Step2: set_data_limit({{"limit": "XXGB"}}) -> "Data limit set successfully"  
Step3: final_answer explaining what was done

OUTPUT: JSON array of {count} objects. Each object must have:
  query_en, query_ar, query_cn, intent_label_en, intent_label_ar, 
  react_label_en, react_label_ar, categories, dataset

 react_label format: [{{"role":"assistant","content":{{"thought":"...","tool_name":"...","args":{{...}},"tool_result":"...","final_answer":"..."}}}}]

Set dataset field to "条件依赖-修复-单轮" for all."""

    print("  📝 调用LLM生成批1...")
    result = call_llm(prompt, temperature=0.8, max_tokens=4096)
    records = parse_json_response(result)
    print(f"  ✅ 生成 {len(records)} 条")
    return records


# ─── 批次2: 条件为假 + 边界处理 ───

def generate_batch2(count: int = 35) -> list:
    """批2: 条件为假 + 边界处理。"""
    sub2a_count = 20  # 条件为假→解释
    sub2b_count = 10  # 条件为假→备选
    sub2c_count = 5   # 边界处理

    prompt = f"""Generate {count} condition-dependency training samples where the condition evaluates to FALSE or at boundary.

SPECIFICATIONS:

--- SUBSET A ({sub2a_count}条): Condition FALSE -> Explanation ---
User queries like "Check usage. If over XGB, increase limit to YGB."
Actual usage: 3.2GB (always below threshold, so condition "over X" is FALSE)
Gold trajectory:
  Step1: get_traffic_statistics({{}}) -> "Total 17.3GB, monthly 3.2GB"
  Step2: final_answer explaining "Current usage is 3.2GB, which is below XGB threshold. No adjustment needed."
NO set_data_limit call (condition is false).

--- SUBSET B ({sub2b_count}条): Condition FALSE -> Alternative Suggestion ---
Same false condition, but agent offers alternative advice instead of just saying "no need".
final_answer example: "Usage is 3.2GB, below your 60GB threshold. Instead of raising the limit, I recommend enabling usage alerts at 80% to stay informed. Would you like me to set that up?"

--- SUBSET C ({sub2c_count}条): Boundary Condition -> Precise Handling ---
User query: "Set limit to 50GB if usage is under 3.2GB"
Actual: exactly 3.2GB (equals threshold)
Agent should handle precisely: "Your usage is exactly 3.2GB. Per policy, when usage equals the threshold, we maintain current settings and enable a reminder. No change made, alert activated."

EMOTION VARIETY: Include different tones - worried, disappointed, frustrated, neutral, urgent.

OUTPUT: JSON array of {count} complete records.
Set dataset field to "条件依赖-为假处理-单轮" for all."""

    print("  📝 调用LLM生成批2...")
    result = call_llm(prompt, temperature=0.8, max_tokens=8192)
    records = parse_json_response(result)
    print(f"  ✅ 生成 {len(records)} 条")
    return records


# ─── 批次3: 多工具条件分支 ───

def generate_batch3(count: int = 40) -> list:
    """批3: 多工具条件分支。"""
    prompt = f"""Generate {count} condition-dependency training samples where DIFFERENT CONDITIONS trigger DIFFERENT TOOLS.

Available tools and their schemas:
{json.dumps(TOOL_SCHEMAS, indent=2, ensure_ascii=False)}

SCENARIOS TO COVER (distribute evenly):

1. WiFi speed conditions (10条):
   - if speed < 10Mbps -> set_wifi_channel(optimize)
   - if 5G signal weak -> switch_5G_priority(ON)
   - if WiFi hidden -> switch_wifi_broadcast(show)
   - if channel interference -> set_wifi_channel(auto=0)
   - if bandwidth not max -> set_wifi_bandwidth(160)

2. Security conditions (8条):
   - if firewall OFF -> switch_firewall(ON)
   - if game turbo OFF -> switch_game_turbo(ON)  
   - if intelligent coverage OFF -> switch_intelligent_func(ON)

3. Network conditions (7条):
   - if data service OFF -> switch_data_mode(ON)
   - if DNS issues -> get_dns_info + suggest fix
   - if IP conflict detected -> set_ip_address

4. Combined operations (5条):
   - if WiFi name is default "CPE_XXXX" -> set_wifi_name(custom) + switch_wifi_broadcast(ON)
   - if both 2.4G and 5G have issues -> sequential channel fixes

For EACH sample:
- User query mentions a problem
- Agent uses get_xxx tool to check condition
- Based on result, condition is TRUE -> calls the appropriate set/switch tool
- final_answer explains what was done

Include both TRUE and FALSE conditions ( roughly 70% true, 30% false).

OUTPUT: JSON array of {count} complete records.
Set dataset field to "条件依赖-多工具分支-单轮" for all."""

    print("  📝 调用LLM生成批3...")
    result = call_llm(prompt, temperature=0.8, max_tokens=8192)
    records = parse_json_response(result)
    print(f"  ✅ 生成 {len(records)} 条")
    return records


# ─── 批次4: 多轮条件对话 ───

def generate_batch4(count: int = 35) -> list:
    """批4: 多轮条件对话。"""
    prompt = f"""Generate {count} MULTI-TURN condition-dependency training samples.

SCENARIO TYPES (distribute evenly):

--- TYPE A: Standard 3-turn (诉求→查→判→执) (12条) ---
Turn1-User: States request with condition (e.g., "Check data. If under 20GB, set limit to 50GB.")
Turn1-Agent: Calls get_traffic_statistics, reports result
Turn2-User: May confirm or ask follow-up
Turn2-Agent: Evaluates condition, calls appropriate tool (set_data_limit if true), gives final_answer

--- TYPE B: With clarification (诉求→问→确→执) (10条) ---
Turn1-User: Vague request ("My WiFi seems slow")
Turn1-Agent: Asks clarifying questions ("All devices or specific? Should I check channel?")
Turn2-User: Provides details
Turn2-Agent: Calls diagnostic tool, finds issue, asks for confirmation to fix
Turn3-User: Confirms
Turn3-Agent: Executes fix, confirms result

--- TYPE C: User changes condition (诉求→查→改→重判) (8条) ---
Turn1-User: "Check data. If over 30GB, set to 80GB."
Turn1-Agent: "Usage is 3.2GB, below 30GB. No change needed."
Turn2-User: "Oh wait. If below 20GB, set to 60GB instead."
Turn2-Agent: [Re-evaluates: 3.2 < 20 is TRUE] → set_data_limit(60GB) → confirms

--- TYPE D: Complex negotiation (5条) ---
Multiple turns of back-and-forth where agent and user negotiate the best solution.
Agent may propose options, user selects, agent implements.

IMPORTANT:
- Each turn's assistant message in react_label must have complete thought + tool_call + result
- Use history_en/ar/cn fields to record the multi-turn conversation
- Final answer should reference the full conversation context

OUTPUT: JSON array of {count} complete records with history fields populated.
Set dataset field to "条件依赖-多轮对话" for all."""

    print("  📝 调用LLM生成批4...")
    result = call_llm(prompt, temperature=0.8, max_tokens=8192)
    records = parse_json_response(result)
    print(f"  ✅ 生成 {len(records)} 条")
    return records


# ─── 批次5: 复合/嵌套/否定条件 ───

def generate_batch5(count: int = 30) -> list:
    """批5: 复合/嵌套/否定条件。"""
    prompt = f"""Generate {count} COMPLEX condition-dependency training samples.

CONDITION TYPES:

--- TYPE A: Compound AND (10条) ---
Examples:
- "Check data. If usage > 50GB AND less than 10 days left in month, set limit to 100GB."
- "If WiFi speed < 10Mbps AND channel interference detected, switch to auto channel."

Gold trajectory must show BOTH conditions being checked (or short-circuit if first is false).

--- TYPE B: Compound OR (8条) ---
Examples:
- "Help! If speed < 5Mbps OR connection drops > 3 times, reset my WiFi."
- "If firewall is OFF OR unknown devices connected, alert me."

Gold trajectory must check both conditions, execute if EITHER is true.

--- TYPE C: Nested conditions (7条) ---
Examples:
- "Check WiFi. If it's ON, check if channel < 6, and if so upgrade to 6."
- "If data service is ON, check if usage > 80% limit, and if so suggest upgrade."

Gold trajectory shows nested decision: outer condition → inner condition → action.

--- TYPE D: Negation (5条) ---
Examples:
- "Make sure firewall is active. If NOT enabled, turn it on."
- "Check if game mode is OFF. If it is NOT running, start it."

Gold trajectory: check state → if negation is TRUE (state is false) → execute fix.

REQUIREMENTS:
- Mix TRUE and FALSE outcomes (~60% true, 40% false)
- All tool calls must reference valid tools from the available set
- final_answer must clearly explain the conditional logic used
- Include thought process showing the reasoning steps

OUTPUT: JSON array of {count} complete records.
Set dataset field to "条件依赖-复合嵌套否定" for all."""

    print("  📝 调用LLM生成批5...")
    result = call_llm(prompt, temperature=0.8, max_tokens=8192)
    records = parse_json_response(result)
    print(f"  ✅ 生成 {len(records)} 条")
    return records


# ─── 主控 ───

def generate_batch(batch_num: int, count: int) -> list:
    """生成指定批次。"""
    generators = {
        1: generate_batch1,
        2: generate_batch2,
        3: generate_batch3,
        4: generate_batch4,
        5: generate_batch5,
    }
    if batch_num not in generators:
        print(f"❌ 无效批次: {batch_num}，有效值: 1-5")
        return []
    
    print(f"\n{'='*60}")
    print(f"🚀 生成批次 {batch_num}")
    print(f"{'='*60}")
    
    records = generators[batch_num](count)
    
    # 保存
    output_dir = Path("condition_dependency_rebuilt")
    output_dir.mkdir(exist_ok=True)
    
    batch_names = {
        1: "batch1_repaired_existing.json",
        2: "batch2_false_condition.json",
        3: "batch3_multi_tool.json",
        4: "batch4_multi_turn.json",
        5: "batch5_complex_condition.json",
    }
    
    output_path = output_dir / batch_names[batch_num]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    
    print(f"  💾 保存: {output_path} ({len(records)}条)")
    return records


def merge_all_batches():
    """合并所有批次。"""
    output_dir = Path("condition_dependency_rebuilt")
    all_records = []
    
    batch_files = [
        "batch1_repaired_existing.json",
        "batch2_false_condition.json",
        "batch3_multi_tool.json",
        "batch4_multi_turn.json",
        "batch5_complex_condition.json",
    ]
    
    for fname in batch_files:
        fpath = output_dir / fname
        if fpath.exists():
            with open(fpath, "r", encoding="utf-8") as f:
                records = json.load(f)
            all_records.extend(records)
            print(f"  📁 {fname}: {len(records)}条")
        else:
            print(f"  ⚠️ 未找到: {fname}")
    
    # 添加全局ID
    for i, r in enumerate(all_records, 1):
        r["_global_id"] = f"CD_{i:04d}"
        r["_batch_total"] = len(all_records)
    
    merged_path = output_dir / "_merged_all.json"
    with open(merged_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 合并完成: {merged_path}")
    print(f"   总计: {len(all_records)}条")
    
    # 统计
    datasets = {}
    for r in all_records:
        ds = r.get("dataset", "unknown")
        datasets[ds] = datasets.get(ds, 0) + 1
    print(f"\n📊 分布:")
    for ds, cnt in sorted(datasets.items()):
        print(f"   {ds}: {cnt}条")
    
    return all_records


def validate_batch(records: list, batch_num: int):
    """验证一批数据的基本质量。"""
    print(f"\n🔍 验证批次 {batch_num}:")
    
    issues = []
    for i, r in enumerate(records):
        # 检查必需字段
        for field in ["query_en", "query_cn", "react_label_en"]:
            if not r.get(field):
                issues.append(f"  记录{i}: 缺少 {field}")
        
        # 检查react_label结构
        react = r.get("react_label_en", [])
        if react:
            has_tool = any(s.get("content", {}).get("tool_name") for s in react)
            has_fa = any(s.get("content", {}).get("final_answer") for s in react)
            if not has_fa:
                issues.append(f"  记录{i}: 无final_answer")
    
    if issues:
        print(f"  ⚠️ 发现 {len(issues)} 个问题:")
        for issue in issues[:10]:
            print(issue)
        if len(issues) > 10:
            print(f"  ... 还有 {len(issues)-10} 个")
    else:
        print(f"  ✅ 全部 {len(records)} 条验证通过")
    
    return len(issues) == 0


def main():
    parser = argparse.ArgumentParser(description="条件依赖数据重建脚本")
    parser.add_argument("--batch", type=int, choices=[1,2,3,4,5], help="生成指定批次")
    parser.add_argument("--count", type=int, help="指定生成数量")
    parser.add_argument("--all", action="store_true", help="生成全部5批")
    parser.add_argument("--merge", action="store_true", help="合并所有批次")
    parser.add_argument("--validate", action="store_true", help="验证已生成的批次")
    args = parser.parse_args()
    
    default_counts = {1: 40, 2: 35, 3: 40, 4: 35, 5: 30}
    
    if args.merge:
        merge_all_batches()
        return
    
    if args.all:
        for bn in range(1, 6):
            cnt = args.count or default_counts[bn]
            records = generate_batch(bn, cnt)
            validate_batch(records, bn)
            time.sleep(2)  # 批次间间隔
        print(f"\n{'='*60}")
        print("🎉 全部5批生成完成！正在合并...")
        print(f"{'='*60}")
        merge_all_batches()
        return
    
    if args.batch:
        cnt = args.count or default_counts[args.batch]
        records = generate_batch(args.batch, cnt)
        if args.validate:
            validate_batch(records, args.batch)
        return
    
    if args.validate:
        output_dir = Path("condition_dependency_rebuilt")
        batch_files = [
            (1, "batch1_repaired_existing.json"),
            (2, "batch2_false_condition.json"),
            (3, "batch3_multi_tool.json"),
            (4, "batch4_multi_turn.json"),
            (5, "batch5_complex_condition.json"),
        ]
        for bn, fname in batch_files:
            fpath = output_dir / fname
            if fpath.exists():
                with open(fpath, "r", encoding="utf-8") as f:
                    records = json.load(f)
                validate_batch(records, bn)
        return
    
    parser.print_help()
    print("\n示例:")
    print("  python generate_condition_dependency_data.py --all")
    print("  python generate_condition_dependency_data.py --batch 1")
    print("  python generate_condition_dependency_data.py --batch 3 --count 20")
    print("  python generate_condition_dependency_data.py --merge")
    print("  python generate_condition_dependency_data.py --validate")


if __name__ == "__main__":
    main()
