#!/usr/bin/env python3
"""
条件依赖数据重建 V2 —— 每次3条 + 完整示例 + 自动验证

用法:
    python generate_condition_v2.py --batch 1 --count 10
    python generate_condition_v2.py --all
    python generate_condition_v2.py --merge
"""

import argparse
import json
import time
from pathlib import Path

API_CONFIG = {
    "base_url": "http://10.44.209.63:3000/v1",
    "api_key": "sk-nxyLmHRnhpaxOj2ewYbuih68RUTYSQeQCKjc6woSf7DVGlX8",
    "model": "dsv4",
}


def call_llm(prompt: str, temperature: float = 0.7, max_tokens: int = 4096):
    import requests
    for attempt in range(3):
        try:
            resp = requests.post(
                f"{API_CONFIG['base_url']}/chat/completions",
                headers={"Authorization": f"Bearer {API_CONFIG['api_key']}", "Content-Type": "application/json"},
                json={
                    "model": API_CONFIG["model"],
                    "messages": [
                        {"role": "system", "content": "You generate WiFi customer service training data. Output ONLY a valid JSON array. No markdown code blocks, no extra text."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                timeout=120,
            )
            if resp.status_code == 429:
                time.sleep(2 ** attempt + 1)
                continue
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt + 1)
            else:
                print(f"  FAIL: {e}")
                return None
    return None


def parse_json(text: str):
    if not text:
        return []
    text = text.strip()
    if text.startswith("```json"): text = text[7:]
    if text.startswith("```"): text = text[3:]
    if text.endswith("```"): text = text[:-3]
    text = text.strip()
    try:
        d = json.loads(text)
        return d if isinstance(d, list) else [d]
    except:
        s, e = text.find("["), text.rfind("]")
        if s >= 0 and e > s:
            try: return json.loads(text[s:e+1])
            except: pass
        print(f"  JSON parse fail: {text[:300]}")
        return []


EXAMPLE = {
    "query_en": "What's my data situation? If usage is under 30GB, increase limit to 60GB.",
    "query_ar": "ما هو وضع بياناتي؟ إذا كان الاستخدام أقل من 30 جيجا، ارفع الحد إلى 60 جيجا.",
    "query_cn": "我现在数据使用情况如何？如果用量低于30GB，将月度限额提升至60GB。",
    "intent_label_en": {
        "task_decomposition": [
            {"task_num": 1, "intent": "API", "sub_task": "Check current data usage"},
            {"task_num": 2, "intent": "API", "sub_task": "If usage under 30GB, set limit to 60GB"}
        ],
        "language": "EN"
    },
    "intent_label_ar": {
        "task_decomposition": [
            {"task_num": 1, "intent": "API", "sub_task": "التحقق من استخدام البيانات"},
            {"task_num": 2, "intent": "API", "sub_task": "إذا أقل من 30 جيجا، تعيين الحد إلى 60"}
        ],
        "language": "AR"
    },
    "rag_en": {
        "knowledge": {"设置流量限额": "Go to Data limit page"},
        "tools": [
            {"type": "function", "function": {"name": "get_traffic_statistics", "description": "查询流量", "parameters": {}}},
            {"type": "function", "function": {"name": "set_data_limit", "description": "设置限额", "parameters": {"type": "object", "properties": {"limit": {"type": "string"}}, "required": ["limit"]}}}
        ]
    },
    "rag_ar": {
        "knowledge": {"设置流量限额": "صفحة حد البيانات"},
        "tools": [
            {"type": "function", "function": {"name": "get_traffic_statistics", "description": "查询流量", "parameters": {}}},
            {"type": "function", "function": {"name": "set_data_limit", "description": "设置限额", "parameters": {"type": "object", "properties": {"limit": {"type": "string"}}, "required": ["limit"]}}}
        ]
    },
    "react_label_en": [
        {"role": "assistant", "content": {
            "thought": "User wants to check data and conditionally set limit. Call get_traffic_statistics first.",
            "tool_name": "get_traffic_statistics", "args": {},
            "tool_result": "总共使用流量17.3GB, 当月使用流量为3.2GB", "final_answer": ""
        }},
        {"role": "assistant", "content": {
            "thought": "3.2GB < 30GB, condition TRUE. Call set_data_limit with 60GB.",
            "tool_name": "set_data_limit", "args": {"limit": "60GB"},
            "tool_result": "数据限额已设置为60GB", "final_answer": ""
        }},
        {"role": "assistant", "content": {
            "thought": "Both steps done. Provide summary.",
            "tool_name": "", "args": {}, "tool_result": "",
            "final_answer": "I have checked your data. Current month: 3.2GB. Since below 30GB, I have set your limit to 60GB."
        }}
    ],
    "react_label_ar": [
        {"role": "assistant", "content": {
            "thought": "المستخدم يريد التحقق من البيانات",
            "tool_name": "get_traffic_statistics", "args": {},
            "tool_result": "إجمالي 17.3 جيجا، الشهر الحالي 3.2 جيجا", "final_answer": ""
        }},
        {"role": "assistant", "content": {
            "thought": "3.2 < 30، الشرط صحيح",
            "tool_name": "set_data_limit", "args": {"limit": "60GB"},
            "tool_result": "تم تعيين الحد إلى 60 جيجا", "final_answer": ""
        }},
        {"role": "assistant", "content": {
            "thought": "تم الانتهاء",
            "tool_name": "", "args": {}, "tool_result": "",
            "final_answer": "الشهر الحالي 3.2 جيجا. تم تعيين الحد إلى 60 جيجا."
        }}
    ],
    "categories": ["get_traffic_statistics", "set_data_limit"],
    "dataset": "条件依赖-修复-单轮"
}


def validate(r: dict, idx: int):
    errors = []
    for f in ["query_en", "query_ar", "query_cn", "react_label_en"]:
        if not r.get(f):
            errors.append(f"  [{idx}] missing {f}")
    react = r.get("react_label_en", [])
    if not isinstance(react, list):
        errors.append(f"  [{idx}] react_label not array")
        return errors
    if len(react) < 2:
        errors.append(f"  [{idx}] only {len(react)} steps")
    for si, step in enumerate(react):
        c = step.get("content", {}) if isinstance(step, dict) else {}
        if not c:
            errors.append(f"  [{idx}] step{si} no content")
            continue
        if si < len(react)-1 and not c.get("tool_name"):
            errors.append(f"  [{idx}] step{si} non-final missing tool_name")
        if si == len(react)-1 and not c.get("final_answer"):
            errors.append(f"  [{idx}] last step no final_answer")
    intent = r.get("intent_label_en")
    if intent and not isinstance(intent, dict):
        errors.append(f"  [{idx}] intent_label not object")
    cats = r.get("categories")
    if cats and not isinstance(cats, list):
        errors.append(f"  [{idx}] categories not array")
    return errors


def gen_chunk(template: str, n: int, temp: float = 0.8):
    example_json = json.dumps([EXAMPLE], ensure_ascii=False, indent=2)
    prompt = f"""Generate EXACTLY {n} training samples. Follow this EXACT format:

EXAMPLE:
{example_json}

REQUIREMENTS:
{template}

RULES:
1. react_label_en MUST be array of objects with content.thought/tool_name/args/tool_result/final_answer
2. intent_label_en MUST have task_decomposition array
3. categories MUST be array of strings
4. query_cn must be natural Chinese, NO emotion prefix like "中性" or "愤怒"
5. Output ONLY JSON array, no markdown

Generate {n} records:"""

    result = call_llm(prompt, temperature=temp, max_tokens=4096)
    return parse_json(result)


def gen_all(template: str, total: int):
    all_r, remaining, call_num = [], total, 0
    while remaining > 0:
        cur = min(3, remaining)
        call_num += 1
        print(f"    API #{call_num} (gen{cur}, remain{remaining-cur})...")
        records = gen_chunk(template, cur)
        if not records:
            remaining -= cur
            continue
        valid = []
        for r in records:
            errs = validate(r, len(all_r)+len(valid)+1)
            if not errs:
                valid.append(r)
            else:
                print(f"    skip: {errs[0]}")
        print(f"    valid {len(valid)}/{len(records)}")
        all_r.extend(valid)
        remaining -= cur
        if remaining > 0:
            time.sleep(1)
    return all_r


BATCH_PROMPTS = {
    1: '条件依赖-修复。用户查流量+条件为真时设限额。\n阈值: 8/10/12/15/20/25/30/40/50/60/70/80/90/105GB\n情绪: neutral/frustrated/angry/urgent/disappointed/worried\n条件全为真(3.2GB<阈值)。\nGold: get_traffic_statistics -> set_data_limit -> final_answer\nset dataset to "条件依赖-修复-单轮"',

    2: '条件依赖-为假处理。条件不满足时的处理。\n\n类型A(~60%): 条件为假->解释\n  query: "If over XGB, increase limit"\n  实际3.2GB<X, 条件为假\n  Gold: get_traffic_statistics -> final_answer(解释无需调整)\n  无set_data_limit\n\n类型B(~30%): 条件为假->备选建议\n  给出替代建议(如开启提醒)\n\n类型C(~10%): 边界条件\n  实际=3.2GB等于阈值,精确处理\n\nset dataset to "条件依赖-为假处理-单轮"',

    3: '条件依赖-多工具分支。不同条件触发不同工具。\n工具: get_ethernet_speed, set_wifi_channel, switch_5G_priority, switch_wifi_broadcast, set_wifi_bandwidth, switch_firewall, switch_game_turbo, switch_data_mode, get_wifi_diagnosis_info, set_wifi_name, get_dns_info\n\n场景(均匀):\n- WiFi: 速度慢->set_wifi_channel | 信号差->switch_5G_priority | 隐藏->switch_wifi_broadcast | 信道拥挤->set_wifi_channel(自动) | 带宽低->set_wifi_bandwidth\n- 安全: 防火墙关->switch_firewall(ON) | 游戏加速关->switch_game_turbo(ON) | 智能覆盖关->switch_intelligent_func(ON)\n- 网络: 数据业务关->switch_data_mode(ON) | DNS问题->get_dns_info+建议\n- 组合: WiFi名默认->set_wifi_name+custom + switch_wifi_broadcast\n\n70%真(执行工具), 30%假(解释)\nrag_en.tools只含该场景需要的2-3个工具\nset dataset to "条件依赖-多工具分支-单轮"',

    4: '条件依赖-多轮对话。多轮条件交互。\n\n类型A-标准3轮(~35%): R1诉求->R1查询->R2确认->R2判断执行\n类型B-含澄清(~30%): R1模糊诉求->R1反问->R2详情->R2诊断->R3确认->R3执行\n类型C-用户改条件(~25%): R1条件A->R1判断假->R2改条件B->R2重判真->执行\n类型D-协商(~10%): 多轮讨价还价,agent给选项,用户选,agent执行\n\n用history_en/ar/cn记录多轮上下文\nset dataset to "条件依赖-多轮对话"',

    5: '条件依赖-复合嵌套否定。复杂条件逻辑。\n\n类型A-AND(~35%): "If usage>50GB AND days_left<10, set 100GB"\n  Gold: 查A->查B->都真则执行, 短路:A假则直接说明\n\n类型B-OR(~25%): "If speed<5 OR drops>3, reset"\n  Gold: 查A->查B->任一真则执行\n\n类型C-嵌套(~25%): "If WiFi ON, check channel<6, if so upgrade"\n  Gold: 外层->内层->执行, 外层假则不进内层\n\n类型D-否定(~15%): "If NOT firewall_on, turn_on"\n  Gold: 查状态->否定真则执行\n\n60% TRUE / 40% FALSE\nset dataset to "条件依赖-复合嵌套否定"',
}


def save(records, bn):
    names = {1:"batch1_repaired.json",2:"batch2_false.json",3:"batch3_multi_tool.json",
             4:"batch4_multi_turn.json",5:"batch5_complex.json"}
    d = Path("condition_rebuilt_v2")
    d.mkdir(exist_ok=True)
    p = d / names[bn]
    with open(p, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"  save: {p} ({len(records)}条)")


def merge():
    d = Path("condition_rebuilt_v2")
    all_r = []
    for fn in ["batch1_repaired.json","batch2_false.json","batch3_multi_tool.json",
               "batch4_multi_turn.json","batch5_complex.json"]:
        fp = d / fn
        if fp.exists():
            with open(fp, "r", encoding="utf-8") as f:
                all_r.extend(json.load(f))
    for i,r in enumerate(all_r,1): r["_global_id"] = f"CD_{i:04d}"
    mp = d / "_merged_all.json"
    with open(mp, "w", encoding="utf-8") as f:
        json.dump(all_r, f, ensure_ascii=False, indent=2)
    print(f"  merge: {mp} ({len(all_r)}条)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, choices=[1,2,3,4,5])
    parser.add_argument("--count", type=int)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()

    defaults = {1:40, 2:35, 3:40, 4:35, 5:30}

    if args.merge:
        merge()
        return

    batches = [args.batch] if args.batch else list(range(1,6))

    for bn in batches:
        cnt = args.count or defaults[bn]
        print(f"\n{'='*50}")
        print(f"batch {bn}: target {cnt}")
        print(f"{'='*50}")
        records = gen_all(BATCH_PROMPTS[bn], cnt)
        total_err = sum(len(validate(r,i+1)) for i,r in enumerate(records))
        print(f"  validate: {len(records)} records, {total_err} errors")
        save(records, bn)
        if bn < max(batches): time.sleep(3)

    if args.all or not args.batch:
        print(f"\n{'='*50}")
        merge()


if __name__ == "__main__":
    main()
