#!/usr/bin/env python3
"""
条件依赖数据重建 V4 —— 工程级实现
核心特性:
  - 实时追加写入（每验证通过一条立即写入文件）
  - 失败日志记录（failed_records.log 记录失败原因和原始内容）
  - 断点续跑（检测已生成数量，自动跳过）
  - 最终完整性校验

用法:
    python generate_condition_v4.py --batch 1 --count 40
    python generate_condition_v4.py --all
    python generate_condition_v4.py --resume --batch 3  # 从中断处继续
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

API_CONFIG = {
    "base_url": "http://10.44.209.63:3000/v1",
    "api_key": "sk-nxyLmHRnhpaxOj2ewYbuih68RUTYSQeQCKjc6woSf7DVGlX8",
    "model": "dsv4",
}

# ─── 工具函数 ───

def call_llm(prompt: str, temperature: float = 0.7, max_tokens: int = 8192, timeout: int = 300):
    import requests
    for attempt in range(3):
        try:
            resp = requests.post(
                f"{API_CONFIG['base_url']}/chat/completions",
                headers={"Authorization": f"Bearer {API_CONFIG['api_key']}", "Content-Type": "application/json"},
                json={
                    "model": API_CONFIG["model"],
                    "messages": [
                        {"role": "system", "content": "You generate WiFi customer service training data. Output ONLY a valid JSON array. No markdown. Always complete all requested records."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                timeout=timeout,
            )
            if resp.status_code == 429:
                wait = 2 ** attempt + 1
                print(f"      429限流，等待{wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except requests.exceptions.ReadTimeout:
            wait = 2 ** attempt + 1
            print(f"      超时，等待{wait}s后重试({attempt+1}/3)...")
            time.sleep(wait)
        except Exception as e:
            if attempt < 2:
                wait = 2 ** attempt + 1
                print(f"      失败，{wait}s后重试: {e}")
                time.sleep(wait)
            else:
                print(f"      最终失败: {e}")
                return None
    return None


def parse_json(text: str) -> List[Dict]:
    if not text:
        return []
    text = text.strip()
    if text.startswith("```json"): text = text[7:]
    if text.startswith("```"): text = text[3:]
    if text.endswith("```"): text = text[:-3]
    text = text.strip()
    
    # 尝试完整解析
    try:
        d = json.loads(text)
        return d if isinstance(d, list) else [d]
    except:
        pass
    
    # 尝试提取JSON数组
    s, e = text.find("["), text.rfind("]")
    if s >= 0 and e > s:
        try:
            return json.loads(text[s:e+1])
        except:
            pass
    
    # 截断恢复：逐个大括号匹配提取完整记录
    records = []
    brace_count = 0
    record_start = -1
    i = 0
    while i < len(text):
        if text[i] == '{':
            if brace_count == 0:
                record_start = i
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
            if brace_count == 0 and record_start >= 0:
                try:
                    record = json.loads(text[record_start:i+1])
                    records.append(record)
                except:
                    pass
                record_start = -1
        i += 1
    
    if records:
        print(f"      截断恢复: 提取到{len(records)}条完整记录")
    else:
        print(f"      JSON解析失败，文本长度{len(text)}")
    return records


def validate_record(r: Dict, idx: int) -> List[str]:
    """验证单条记录，返回错误列表（空=通过）。"""
    errors = []
    # 必需字段
    for f in ["query_en", "query_ar", "query_cn"]:
        if not r.get(f):
            errors.append(f"missing_{f}")
    
    # react_label_en必须是数组
    react = r.get("react_label_en")
    if not react:
        errors.append("missing_react_label_en")
    elif not isinstance(react, list):
        errors.append(f"react_label_not_array:{type(react).__name__}")
    else:
        if len(react) < 2:
            errors.append(f"too_few_steps:{len(react)}")
        for si, step in enumerate(react):
            if not isinstance(step, dict):
                errors.append(f"step{si}_not_dict")
                continue
            c = step.get("content", {})
            if not c:
                errors.append(f"step{si}_no_content")
                continue
            if si < len(react) - 1:
                # 非最后一步必须有tool_name
                if not c.get("tool_name"):
                    errors.append(f"step{si}_missing_tool_name")
            else:
                # 最后一步必须有final_answer
                if not c.get("final_answer"):
                    errors.append(f"last_step_missing_final_answer")
    
    # intent_label_en必须是对象且含task_decomposition
    intent = r.get("intent_label_en")
    if intent and not isinstance(intent, dict):
        errors.append("intent_label_not_object")
    elif intent and not intent.get("task_decomposition"):
        errors.append("intent_missing_task_decomposition")
    
    # categories必须是数组
    cats = r.get("categories")
    if cats and not isinstance(cats, list):
        errors.append("categories_not_array")
    
    return errors


# ─── 文件操作 ───

class BatchWriter:
    """实时写入管理器：支持追加写入、失败日志、断点续跑。"""
    
    def __init__(self, batch_num: int, output_dir: Path):
        self.batch_num = batch_num
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        names = {
            1: "batch1_repaired.json", 2: "batch2_false.json",
            3: "batch3_multi_tool.json", 4: "batch4_multi_turn.json",
            5: "batch5_complex.json",
        }
        self.file_path = self.output_dir / names[batch_num]
        self.fail_log_path = self.output_dir / f"batch{batch_num}_failed.log"
        self.meta_path = self.output_dir / f"batch{batch_num}_meta.json"
        
        # 初始化文件（如果不存在则写空数组开头）
        if not self.file_path.exists():
            self.file_path.write_text("[\n", encoding="utf-8")
            self.first_record = True
            self.existing_count = 0
        else:
            # 计算已有记录数（断点续跑）
            self.existing_count = self._count_existing()
            self.first_record = (self.existing_count == 0)
            print(f"      检测到已有{self.existing_count}条记录，将追加写入")
        
        self.success_count = 0
        self.fail_count = 0
    
    def _count_existing(self) -> int:
        """计算文件中已有多少条有效记录。"""
        try:
            # 读取并修复文件格式后解析
            text = self.file_path.read_text(encoding="utf-8").strip()
            if text.endswith(","):
                text = text[:-1]
            if not text.endswith("]"):
                text += "\n]"
            data = json.loads(text)
            return len(data) if isinstance(data, list) else 0
        except:
            return 0
    
    def append_record(self, record: Dict) -> bool:
        """追加单条记录到文件。返回是否成功。"""
        try:
            # 序列化单条记录
            record_json = json.dumps(record, ensure_ascii=False, indent=2)
            
            # 如果不是第一条，先加逗号
            prefix = "\n,\n" if not self.first_record else ""
            self.first_record = False
            
            # 追加写入（不带结尾方括号，下次继续追加）
            with open(self.file_path, "a", encoding="utf-8") as f:
                f.write(prefix + record_json)
            
            self.success_count += 1
            return True
        except Exception as e:
            print(f"      写入失败: {e}")
            return False
    
    def finalize(self):
        """结束写入，补全JSON格式。"""
        try:
            with open(self.file_path, "a", encoding="utf-8") as f:
                f.write("\n]\n")
        except Exception as e:
            print(f"      最终化失败: {e}")
    
    def log_failure(self, reason: str, raw_text: str = "", call_num: int = 0):
        """记录失败日志。"""
        self.fail_count += 1
        try:
            with open(self.fail_log_path, "a", encoding="utf-8") as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"时间: {datetime.now().isoformat()}\n")
                f.write(f"批次: {self.batch_num}\n")
                f.write(f"调用序号: {call_num}\n")
                f.write(f"失败原因: {reason}\n")
                if raw_text:
                    f.write(f"原始文本前500字符:\n{raw_text[:500]}\n")
                f.write(f"{'='*60}\n")
        except Exception as e:
            print(f"      写失败日志也失败了: {e}")
    
    def save_meta(self, target: int):
        """保存元数据。"""
        meta = {
            "batch": self.batch_num,
            "target": target,
            "generated": self.success_count + self.existing_count,
            "newly_generated": self.success_count,
            "existing": self.existing_count,
            "failed": self.fail_count,
            "completed": self.success_count + self.existing_count >= target,
            "updated": datetime.now().isoformat(),
        }
        try:
            with open(self.meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        except:
            pass
        return meta


# ─── 示例记录 ───

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
            "thought": "User wants to check data and conditionally set limit.",
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


# ─── 批次Prompt ───

BATCH_PROMPTS = {
    1: '条件依赖-修复。用户查流量+条件为真时设限额。\n阈值分布: 8/10/12/15/20/25/30/40/50/60/70/80/90/105GB\n情绪: neutral/frustrated/angry/urgent/disappointed/worried\n条件全为真(3.2GB<阈值)。\nGold: get_traffic_statistics -> set_data_limit -> final_answer\nset dataset to "条件依赖-修复-单轮"',

    2: '条件依赖-为假处理。条件不满足时的处理。\n\n类型A(~60%): 条件为假->解释\n  query: "If over XGB, increase limit"\n  实际3.2GB<X, 条件为假\n  Gold: get_traffic_statistics -> final_answer(解释无需调整)\n  无set_data_limit\n\n类型B(~30%): 条件为假->备选建议\n  给出替代建议(如开启提醒)\n\n类型C(~10%): 边界条件\n  实际=3.2GB等于阈值,精确处理\n\nset dataset to "条件依赖-为假处理-单轮"',

    3: '条件依赖-多工具分支。不同条件触发不同工具。\n工具: get_ethernet_speed, set_wifi_channel, switch_5G_priority, switch_wifi_broadcast, set_wifi_bandwidth, switch_firewall, switch_game_turbo, switch_data_mode, get_wifi_diagnosis_info, set_wifi_name, get_dns_info\n\n场景(均匀):\n- WiFi: 速度慢->set_wifi_channel | 信号差->switch_5G_priority | 隐藏->switch_wifi_broadcast | 信道拥挤->set_wifi_channel(自动) | 带宽低->set_wifi_bandwidth\n- 安全: 防火墙关->switch_firewall(ON) | 游戏加速关->switch_game_turbo(ON) | 智能覆盖关->switch_intelligent_func(ON)\n- 网络: 数据业务关->switch_data_mode(ON) | DNS问题->get_dns_info+建议\n- 组合: WiFi名默认->set_wifi_name+custom + switch_wifi_broadcast\n\n70%真(执行工具), 30%假(解释)\nrag_en.tools只含该场景需要的2-3个工具\nset dataset to "条件依赖-多工具分支-单轮"',

    4: '条件依赖-多轮对话。多轮条件交互。\n\n类型A-标准3轮(~35%): R1诉求->R1查询->R2确认->R2判断执行\n类型B-含澄清(~30%): R1模糊诉求->R1反问->R2详情->R2诊断->R3确认->R3执行\n类型C-用户改条件(~25%): R1条件A->R1判断假->R2改条件B->R2重判真->执行\n类型D-协商(~10%): 多轮讨价还价,agent给选项,用户选,agent执行\n\n用history_en/ar/cn记录多轮上下文\nset dataset to "条件依赖-多轮对话"',

    5: '条件依赖-复合嵌套否定。复杂条件逻辑。\n\n类型A-AND(~35%): "If usage>50GB AND days_left<10, set 100GB"\n  Gold: 查A->查B->都真则执行, 短路:A假则直接说明\n\n类型B-OR(~25%): "If speed<5 OR drops>3, reset"\n  Gold: 查A->查B->任一真则执行\n\n类型C-嵌套(~25%): "If WiFi ON, check channel<6, if so upgrade"\n  Gold: 外层->内层->执行, 外层假则不进内层\n\n类型D-否定(~15%): "If NOT firewall_on, turn_on"\n  Gold: 查状态->否定真则执行\n\n60% TRUE / 40% FALSE\nset dataset to "条件依赖-复合嵌套否定"',
}


# ─── 核心生成逻辑 ───

def generate_batch(batch_num: int, target: int, output_dir: Path, resume: bool = True):
    """生成单个批次的全部数据。"""
    writer = BatchWriter(batch_num, output_dir)
    
    # 断点续跑：计算还需生成多少
    remaining = target - writer.existing_count if resume else target
    if remaining <= 0:
        print(f"    ✅ 批次{batch_num}已完成（已有{writer.existing_count}/{target}条）")
        meta = writer.save_meta(target)
        return meta
    
    print(f"    需新生成: {remaining}条（已有{writer.existing_count}条）")
    
    call_num = 0
    while remaining > 0:
        chunk_size = min(3, remaining)
        call_num += 1
        print(f"    API #{call_num} (请求{chunk_size}条, 还需{remaining-chunk_size}条)...")
        
        # 构建prompt
        example_json = json.dumps([EXAMPLE], ensure_ascii=False, indent=2)
        prompt = f"""Generate EXACTLY {chunk_size} training samples. Follow this EXACT format:

EXAMPLE:
{example_json}

REQUIREMENTS:
{BATCH_PROMPTS[batch_num]}

RULES:
1. react_label_en MUST be array of objects with content.thought/tool_name/args/tool_result/final_answer
2. intent_label_en MUST have task_decomposition array
3. categories MUST be array of strings
4. query_cn must be natural Chinese, NO emotion prefix
5. Output ONLY JSON array, no markdown

Generate {chunk_size} records:"""
        
        # 调用API
        result = call_llm(prompt, temperature=0.8, max_tokens=8192, timeout=300)
        
        if not result:
            writer.log_failure("API调用完全失败（3次重试后）", "", call_num)
            remaining -= chunk_size  # 跳过，不阻塞
            time.sleep(5)
            continue
        
        # 解析
        records = parse_json(result)
        
        if not records:
            writer.log_failure("JSON解析失败/无数据", result, call_num)
            remaining -= chunk_size
            time.sleep(3)
            continue
        
        # 逐条验证并实时写入
        valid_count = 0
        for r in records:
            errs = validate_record(r, writer.success_count + writer.existing_count + 1)
            if not errs:
                # ✅ 验证通过 → 立即写入
                if writer.append_record(r):
                    valid_count += 1
                    print(f"      +1 ({writer.success_count + writer.existing_count}/{target})")
                else:
                    writer.log_failure("写入文件失败", json.dumps(r, ensure_ascii=False)[:200], call_num)
            else:
                # ❌ 验证失败 → 记录失败
                err_str = ";".join(errs)
                writer.log_failure(f"验证失败: {err_str}", json.dumps(r, ensure_ascii=False)[:300], call_num)
                print(f"      skip: {err_str}")
        
        remaining -= chunk_size
        print(f"      本次有效: {valid_count}/{len(records)}")
        
        if remaining > 0:
            time.sleep(2)
    
    # 结束写入
    writer.finalize()
    meta = writer.save_meta(target)
    
    print(f"    ✅ 批次{batch_num}完成: {meta['generated']}/{target}条 (新增{meta['newly_generated']}, 失败{meta['failed']})")
    return meta


def merge_all(output_dir: Path):
    """合并所有批次。"""
    names = {
        1: "batch1_repaired.json", 2: "batch2_false.json",
        3: "batch3_multi_tool.json", 4: "batch4_multi_turn.json",
        5: "batch5_complex.json",
    }
    all_records = []
    for bn, fname in names.items():
        fp = output_dir / fname
        if fp.exists():
            try:
                text = fp.read_text(encoding="utf-8").strip()
                data = json.loads(text)
                if isinstance(data, list):
                    all_records.extend(data)
                    print(f"  batch{bn}: {len(data)}条")
            except Exception as e:
                print(f"  batch{bn}: 读取失败 {e}")
    
    for i, r in enumerate(all_records, 1):
        r["_global_id"] = f"CD_{i:04d}"
    
    merged = output_dir / "_merged_all.json"
    with open(merged, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)
    
    print(f"\n  ✅ 合并完成: {merged} ({len(all_records)}条)")
    return len(all_records)


def final_validation(output_dir: Path):
    """最终完整性校验。"""
    print(f"\n{'='*60}")
    print("🔍 最终完整性校验")
    print(f"{'='*60}")
    
    names = {
        1: "batch1_repaired.json", 2: "batch2_false.json",
        3: "batch3_multi_tool.json", 4: "batch4_multi_turn.json",
        5: "batch5_complex.json",
    }
    
    total_records = 0
    total_errors = 0
    
    for bn, fname in names.items():
        fp = output_dir / fname
        if not fp.exists():
            print(f"  batch{bn}: 文件不存在 ❌")
            continue
        
        try:
            text = fp.read_text(encoding="utf-8").strip()
            data = json.loads(text)
            records = data if isinstance(data, list) else []
        except:
            print(f"  batch{bn}: JSON解析失败 ❌")
            continue
        
        errors = 0
        for i, r in enumerate(records):
            errs = validate_record(r, i+1)
            if errs:
                errors += 1
                if errors <= 3:  # 只显示前3个错误
                    print(f"    batch{bn} record{i+1}: {';'.join(errs)}")
        
        status = "✅" if errors == 0 else f"⚠️ ({errors} errors)"
        print(f"  batch{bn}: {len(records)}条 {status}")
        total_records += len(records)
        total_errors += errors
    
    print(f"\n  总计: {total_records}条, 错误: {total_errors}个")
    return total_errors == 0


# ─── 主控 ───

def main():
    parser = argparse.ArgumentParser(description="条件依赖数据重建 V4")
    parser.add_argument("--batch", type=int, choices=[1,2,3,4,5], help="生成指定批次")
    parser.add_argument("--count", type=int, help="目标数量")
    parser.add_argument("--all", action="store_true", help="生成全部5批")
    parser.add_argument("--merge", action="store_true", help="合并所有批次")
    parser.add_argument("--validate", action="store_true", help="最终校验")
    parser.add_argument("--resume", action="store_true", help="断点续跑模式")
    parser.add_argument("--output", type=str, default="condition_rebuilt_v4", help="输出目录")
    args = parser.parse_args()

    output_dir = Path(args.output)
    defaults = {1: 40, 2: 35, 3: 40, 4: 35, 5: 30}

    if args.merge:
        merge_all(output_dir)
        return

    if args.validate:
        final_validation(output_dir)
        return

    batches = [args.batch] if args.batch else list(range(1, 6))
    all_meta = []

    for bn in batches:
        cnt = args.count or defaults[bn]
        print(f"\n{'='*60}")
        print(f"🚀 批次 {bn}: 目标{cnt}条")
        print(f"{'='*60}")
        
        meta = generate_batch(bn, cnt, output_dir, resume=args.resume)
        all_meta.append(meta)
        
        if bn < max(batches):
            time.sleep(5)

    # 汇总
    print(f"\n{'='*60}")
    print("📊 生成汇总")
    print(f"{'='*60}")
    total_generated = sum(m["generated"] for m in all_meta)
    total_target = sum(m["target"] for m in all_meta)
    total_failed = sum(m["failed"] for m in all_meta)
    print(f"  目标: {total_target}条")
    print(f"  实际: {total_generated}条")
    print(f"  失败: {total_failed}次API调用失败")
    
    # 自动合并
    if args.all or not args.batch:
        print(f"\n{'='*60}")
        merge_all(output_dir)
        final_validation(output_dir)


if __name__ == "__main__":
    main()
