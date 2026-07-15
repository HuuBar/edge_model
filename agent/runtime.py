"""Agent loop：生成完整的、已执行的 trajectory。

本模块是agent runtime，是「业务环境接口」的核心：

    reset sandbox namespace
    build initial prompt
    for step in max_steps:
        model.generate(prompt)        # 调 provider 生成
        parse action                  # 解析 tool_call（native 或 XML）
        validate tool and args        # 由 toolfactory 校验
        execute tool through toolfactory  # 产生真实 sandbox 副作用
        append observation            # 回放进 messages
        stop if terminal or max_steps # 终止判定
    export trajectory

边界：
- runtime 只管 agent loop，不管训练，也不判断这条数据进 RL 还是 SFT。
- runtime 不写死任何 case 的「正确工具顺序」或业务答案。
- runtime 必须保存 prompt_history，否则训练侧无法做 rollout/training consistency audit
  （还原模型当时看到的输入，比对 prompt_hash / tool_schema_hash）。

同一个 ``run_agent_loop`` 接口既能被 standalone rollout 调用，也能被 verl async
agent loop adapter 调用。
"""

import json
import re
from hashlib import sha256
from dataclasses import asdict, is_dataclass
from typing import Any, Callable

from agent.observations import observation_message
from agent.prompts.templates import PROMPT_TEMPLATE_VERSION, prompt_hash, render_prompt, stable_hash
from agent.providers.base import ModelOutput, ModelProvider
from agent.rollout_store import make_rollout_id, make_run_id
from agent.trajectory import Trajectory
from envs.namespace import build_namespace_id
from envs.sandbox_state import SandboxState
from envs.toolfactory import ToolFactory

# 匹配 Qwen3 兼容的 XML 工具调用块：<tool_call>...</tool_call>。
# re.DOTALL 允许工具调用 JSON 跨多行输出，这是大模型最常见的格式。
# 这里先捕获完整 block 内容，再从 block 内提取第一个 JSON object。这样可以容忍模型在
# 标签里额外包了 Markdown JSON fence 或少量解释文字，同时仍保留「无 tool_call = final」语义。
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def parse_tool_calls(text: str) -> tuple[list[dict[str, Any]], str | None]:
    """从模型原始文本里解析 Qwen3 风格的一个或多个 ``<tool_call>`` 块。

    返回 ``(tool_calls, parse_error)`` 二元组，二者至多一个有内容：
    - 文本里根本没有 ``<tool_call>``：返回 ``([], None)``——这不是错误，而是表示
      「本步没有工具调用」，runtime 据此把它当作 final customer reply。
    - 有任意 ``<tool_call>`` 但 JSON 解析失败：返回 ``([], "invalid_tool_call_json: ...")``，
      触发 runtime 的 parse_error 反馈循环；本步不执行任何工具，避免半执行。
    - 任意 JSON 能解析但结构不对（name 非 str 或 arguments 非 dict）：返回
      ``([], "invalid_tool_call_shape")``，同样进反馈循环。
    - 一切正常：返回 ``([{"name": ..., "arguments": ...}, ...], None)``。

    注意：把「无 tool_call」和「有但坏掉」严格区分开，是 final 终止与 parse_error
    反馈两条路径能正确分流的前提。
    """
    # finditer 会保留所有匹配块；如果同一步模型输出多个 <tool_call>，runtime 会如实记录并执行，
    # verifier 再用 multi_tool_per_step_cap 判协议违规。
    matches = list(TOOL_CALL_RE.finditer(text))
    if not matches:
        # 没有 tool_call 标签：不是错误，交给上层当作最终回复处理。
        return [], None

    # tool_calls 是 runtime 交给 toolfactory 的规范化动作列表；这里只做形状解析，
    # 不在这里做业务权限或 outcome 判断。
    tool_calls = []
    for index, match in enumerate(matches, start=1):
        try:
            # match.group(1) 是 <tool_call> 标签内文本；parse_json_object_fragment 负责从中找第一个 JSON object。
            payload = parse_json_object_fragment(match.group(1))
        except json.JSONDecodeError as exc:
            # 标签里不是合法 JSON：返回结构化错误信息喂回模型让它修。
            return [], f"invalid_tool_call_json[{index}]: {exc.msg}"
        name = payload.get("name")
        arguments = payload.get("arguments", {})
        # 形状校验：name 必须是字符串、arguments 必须是 dict，否则无法交给 toolfactory 执行。
        if not isinstance(name, str) or not isinstance(arguments, dict):
            return [], f"invalid_tool_call_shape[{index}]"
        # 只保留 runtime 后续真正需要的字段；provider 原生 tool_call 的 id 在主循环中另行兼容。
        tool_calls.append({"name": name, "arguments": arguments})
    return tool_calls, None


def parse_tool_call(text: str) -> tuple[dict[str, Any] | None, str | None]:
    """兼容旧调用方：解析第一个 ``<tool_call>``，新代码应使用 ``parse_tool_calls``。

    项目早期只支持「每步一个工具」，有些测试或辅助脚本仍调用这个单数版本。
    为避免破坏旧接口，这里保留薄包装：底层仍走多工具解析，然后只返回第一个。
    """

    tool_calls, parse_error = parse_tool_calls(text)
    if parse_error:
        return None, parse_error
    return (tool_calls[0] if tool_calls else None), None


def parse_json_object_fragment(text: str) -> dict[str, Any]:
    """从一段文本中解析第一个 JSON object。

    大模型经常输出如下形态：
      <tool_call>
      ```json
      {"name": "...", "arguments": {...}}
      ```
      </tool_call>

    因此这里不要求整段文本全是 JSON，而是找到第一个 ``{`` 后用 JSONDecoder.raw_decode
    解析一个 object。解析完成后不关心尾部 Markdown fence 或解释文字。
    """

    # 找第一个左花括号；没有花括号就不可能是工具调用 JSON。
    start = text.find("{")
    if start < 0:
        raise json.JSONDecodeError("no JSON object found", text, 0)
    # raw_decode 会从 start 位置解析出第一个 JSON 值，并返回未使用的尾部位置。
    value, _ = json.JSONDecoder().raw_decode(text[start:])
    # tool_call 顶层必须是 object，因为后续需要读取 name/arguments 两个字段。
    if not isinstance(value, dict):
        raise json.JSONDecodeError("JSON value is not an object", text, start)
    return value


def _opaque_ticket_id(case_id: str) -> str:
    """用离线 case_id 派生一个稳定但不可读的 ticket id。

    prompt 里不能暴露原始 case_id，否则模型可能学到数据集编号或泄漏离线标识。
    sha256 前缀既稳定（同一个 case 总是同一个 ticket），又不会把 case_id 明文放进 prompt。
    """

    return f"TKT_{sha256(case_id.encode('utf-8')).hexdigest()[:12].upper()}"


def _agent_value(case: dict[str, Any], key: str) -> Any:
    """Read an **agent-facing** field: agent_facing/visible_context 容器优先，否则 case 顶层。

    **绝不** fallback 到 ``entities``：entities 是 verifier/value_source 用的全量 ground truth，
    含 attachment/refund/order 等内部 id；混进可见投影会把 must_discover 等隐藏态泄露进 prompt。
    must_discover 的 case 顶层不设 order_id → 这里返回 None → 不进 prompt（agent 必须自己 list_orders）。
    """
    # 优先读专门给 agent 暴露的投影容器。两个名字都保留，是为了兼容历史数据格式。
    for container_name in ("agent_facing", "visible_context"):
        container = case.get(container_name)
        if isinstance(container, dict) and container.get(key) is not None:
            return container[key]
    # 兼容较旧 case：如果字段直接放在顶层，也允许进入 prompt。
    return case.get(key)


def _case_context(case: dict[str, Any]) -> dict[str, Any]:
    """从 case 中挑出「模型可见」的上下文子集，喂给 step_user 模板。

    这里只保留线上客服坐席在当前会话中真实可见的字段：
    - ``ticket_id``：当前会话/工单 id，必须 opaque；没有显式给出时由 runtime 内部从
      offline ``case_id`` 派生一个不可读的稳定 id。
    - ``customer_id`` / ``market``：登录客户及其区域上下文。
    - ``order_id``：**仅当 case 顶层/agent-facing view 明确给出时可见**（order_given）；
      must_discover 的 case 顶层无 order_id（只在 entities 里供 verifier），此处不渲染。
    - ``customer_message``：客户原始诉求。

    ``case_id``、``visible_intent``、分类 metadata、以及 attachment/refund/tracking 等内部 id
    都不进 prompt；这些只能通过工具 observation 获取，或仅供 verifier 使用。
    """
    case_id = case["case_id"]
    # 这里只构造 prompt 可见上下文，不做任何真值解析；真值解析属于 verifier。
    return {
        "ticket_id": _agent_value(case, "ticket_id") or _opaque_ticket_id(case_id),
        "customer_id": _agent_value(case, "customer_id"),
        "order_id": _agent_value(case, "order_id"),
        "market": _agent_value(case, "market"),
        "customer_message": _agent_value(case, "customer_message") or "",
    }


def run_agent_loop(
    *,
    case: dict[str, Any],
    env_snapshot: dict[str, Any],
    provider: ModelProvider,
    tool_factory: ToolFactory | None = None,
    run_id: str | None = None,
    rollout_id: str | None = None,
    max_steps: int | None = None,
    sampling_config: dict[str, Any] | None = None,
    event_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """同步跑一次 rollout，返回一条完整的 trajectory dict。

    参数：
    - ``case`` / ``env_snapshot``：业务工单与环境快照（只读读表 + sandbox 初始台账）。
    - ``provider``：模型来源抽象，统一 ``generate(messages, sampling_config, tools)``。
    - ``tool_factory``：工具注册与执行器；为 None 时用默认 ToolFactory。
    - ``run_id`` / ``rollout_id``：身份字段，与 case_id 一起决定 sandbox 隔离命名空间；
      未显式传入时 runtime 会生成一个唯一 run_id，并使用 ``rollout_0001``。
    - ``max_steps``：步数上限；为 None 时走「case.max_steps → 默认 20」的三级回退。
    - ``sampling_config``：采样参数（temperature 等），透传给 provider。
    - ``event_callback``：可选的观测回调；不为 None 时，runtime 会在每步生成、parse_error、
      每个工具调用前后、以及给出 final reply 时同步调用它（传入一个 JSON-able 事件 dict），
      供实时 UI / 流式展示使用。**纯观测**：不改变 loop 行为，且回调内异常被吞掉，绝不
      影响 rollout 本身（基础设施不替模型纠错、也不被 UI 钩子拖垮）。

    返回 ``trajectory.to_dict()``：含 prompt_history / raw_model_outputs /
    parsed_actions / tool_observations / tool_errors / final_text / sandbox_final_state。
    """

    def _emit(event: dict[str, Any]) -> None:
        """同步触发观测回调；任何异常都被隔离，保证 loop 不受 UI 钩子影响。"""
        if event_callback is None:
            return
        try:
            # 事件 dict 只用于外部观察，例如 Web UI 或实时日志。它不回写 trajectory，
            # 因此 callback 的任何副作用都不会改变训练样本。
            event_callback(event)
        except Exception:  # noqa: BLE001 — 观测钩子失败绝不应中断 rollout
            pass

    # --- 准备阶段：sandbox 隔离、工具 schema、prompt 指纹 ---
    tool_factory = tool_factory or ToolFactory()
    run_id = run_id or make_run_id()
    rollout_id = rollout_id or make_rollout_id(1)
    case_id = case["case_id"]  # case_id 必须存在（用 [] 而非 .get，缺了应当直接报错）
    # namespace_id = run_id + case_id + rollout_id：sandbox 写隔离键，保证并发 rollout
    # 之间不共享可变状态。
    namespace_id = build_namespace_id(run_id, case_id, rollout_id)
    # 从只读 env_snapshot 派生本次 rollout 专属的可写 sandbox（读表不可变，写入带 namespace）。
    sandbox = SandboxState.from_env_snapshot(env_snapshot, namespace_id)
    # 线上一致性：rollout 暴露生产注册表的完整工具集；case.allowed_tools/allowed_write_tools
    # 只属于 verifier/routing，不参与 prompt 裁剪。
    tool_schemas = tool_factory.tool_schemas()
    # tool_schema_hash：tool schema 的稳定指纹，进 trajectory，做 GRPO group 同质性校验。
    tool_schema_hash = stable_hash(tool_schemas)

    # 渲染初始两段 prompt：system（行为/格式约束）+ step_user（可见 case 上下文）。
    system_text = render_prompt("system.txt", {})
    step_text = render_prompt("step_user.txt", {"case": _case_context(case)})
    # prompt_hash 只对初始 system+step 求；它是整条 trajectory 的 prompt 指纹，
    # 后续每步的 prompt_history 都记同一个 phash 以标记「同一 prompt 模板下的多轮」。
    phash = prompt_hash(system_text, step_text)

    # 初始 messages：system + 首个 user（step）。后续步会往里追加 assistant / tool / user。
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": step_text},
    ]
    # max_steps 三级回退：显式参数 > case.max_steps > 默认 20（对齐 configs/rollout.yaml）。
    # 用 `or` 链：参数为 None/0 时落到 case，case 也无则兜底 20，保证 loop 一定有上界。
    limit = max_steps or case.get("max_steps") or 20
    trajectory = Trajectory(
        case_id=case_id,
        run_id=run_id,
        rollout_id=rollout_id,
        namespace_id=namespace_id,
        prompt_template_version=PROMPT_TEMPLATE_VERSION,
        prompt_hash=phash,
        tool_schema_hash=tool_schema_hash,
    )

    # 主循环：最多 limit 步。注意这是 for...else 结构——else 只在「循环跑满未 break」
    # 时执行（见下方），用来区分两种终止方式。
    for step_index in range(limit):
        # ★ 每步先快照 prompt_history：记录这一步真正喂给模型的 messages（深拷贝成可
        #   JSON 的形式）、tool_schemas、以及 prompt/tool_schema 哈希。没有这份记录，
        #   训练侧就无法还原模型当时的输入、无法做 rollout/training consistency audit。
        trajectory.prompt_history.append(
            {
                "step": step_index + 1,  # 步号从 1 开始计数（对人类/审计更友好）
                "messages": _jsonable(messages),  # 转成纯 JSON 结构，避免后续 messages 被原地修改污染快照
                "tool_schemas": tool_schemas,
                "prompt_hash": phash,
                "tool_schema_hash": tool_schema_hash,
            }
        )
        # 调 provider 生成：传入当前 messages、采样配置、工具 schema。
        output = provider.generate(messages, sampling_config=sampling_config or {}, tools=tool_schemas)
        # 原样保存模型输出（含元数据 + raw_text + 原生 tool_calls）；训练样本保留模型原生格式。
        trajectory.raw_model_outputs.append(
            {
                "step": step_index + 1,
                **output.metadata(),
                "raw_text": output.raw_text,
                "tool_calls": output.tool_calls,
            }
        )
        # model_metadata 只记一次（首步）：整条 trajectory 用同一模型/适配器/tokenizer 版本，
        # 作为版本对齐字段，不必每步重复。
        if not trajectory.model_metadata:
            trajectory.model_metadata = output.metadata()

        # --- tool_call 双解析路径 ---
        # 路径 1（优先）：provider 原生 function-calling 返回的结构化 tool_calls（可能不止一个）。
        tool_calls = output.tool_calls if output.tool_calls else []
        parse_error = None
        # 路径 2（回退）：没有原生 tool_calls 时，从 raw_text 里解析 Qwen3 <tool_call> XML 块。
        if not tool_calls:
            tool_calls, parse_error = parse_tool_calls(output.raw_text)

        # 观测：本步模型输出（剥掉 <think> 后的可读文本 + 解析结果摘要）。
        _emit(
            {
                "type": "assistant_step",
                "step": step_index + 1,
                "text": strip_reasoning_blocks(output.raw_text or "").strip(),
                "n_tool_calls": len(tool_calls),
                "parse_error": parse_error,
            }
        )

        # --- 分支 A：parse_error 反馈循环 ---
        # 模型想调用工具但格式坏了（坏 JSON / 坏结构）。不终止、不执行，构造一条
        # source="llm" 的 parse_error observation 记入 tool_errors，并把它通过
        # tool_error_feedback 模板喂回模型，让它在下一步自我修正（continue 进入下一步）。
        if parse_error:
            error_observation = {
                "ok": False,
                "error": "parse_error",
                "message": parse_error,
                "source": "llm",  # 标记为模型错误（区别于 environment 注入故障）
                "tool_call_id": f"tc_{step_index + 1}",
            }
            # parse_error 没有真正执行工具，但仍放进 tool_errors，便于统计模型格式错误率。
            trajectory.tool_errors.append(error_observation)
            _emit({"type": "parse_error", "step": step_index + 1, "message": parse_error})
            # 把模型这步的原始输出也回放进 messages（保留它写错的内容，便于它看到自己的错误）。
            messages.append(output.assistant_message or {"role": "assistant", "content": output.raw_text})
            # 用模板生成下一条 user 消息，而不是直接拼字符串，保持提示词版本可追踪。
            feedback = render_prompt("tool_error_feedback.txt", {"error_observation": error_observation})
            messages.append({"role": "user", "content": feedback})
            continue

        # --- 分支 B：无 tool_call ⇒ 终止方式一（自然终止 / final reply）---
        # 既无原生 tool_calls 又解析不出 XML（且无 parse_error），说明模型给的是面向客户的
        # 最终文本。剥掉 <think> 推理块、去空白后存为 final_text，回放 assistant 消息，break。
        if not tool_calls:
            trajectory.final_text = strip_reasoning_blocks(output.raw_text).strip()
            _emit({"type": "final", "step": step_index + 1, "text": trajectory.final_text})
            messages.append(output.assistant_message or {"role": "assistant", "content": output.raw_text})
            break

        # --- 分支 C：有合法 tool_call ⇒ 执行工具 ---
        # runtime permissive：照实执行模型这一步给的全部 tool_call，并在 parsed_actions 里**保留
        # step / tool_call_index**。一步给多个工具是被禁的协议违规，但**不由 runtime 终止**——
        # 交给 verifier 从 parsed_actions 按 step 分组检测、判 reward=0（评分定标，轨迹忠实记录）。
        messages.append(output.assistant_message or {"role": "assistant", "content": output.raw_text})
        for call_index, tool_call in enumerate(tool_calls, start=1):
            # tool_call_id：优先用模型/ provider 带的 id，否则用步号+同步调用序号兜底生成。
            tool_call_id = tool_call.get("id") or _fallback_tool_call_id(step_index, call_index, len(tool_calls))
            parsed = {
                "step": step_index + 1,
                "tool_call_index": call_index,
                "tool_call_id": tool_call_id,
                **tool_call,
            }
            # parsed_actions 记录模型「想做什么」；tool_observations 记录环境「实际返回什么」。
            # 两者通过 tool_call_id 关联，verifier 会同时看这两张表。
            trajectory.parsed_actions.append(parsed)
            _emit(
                {
                    "type": "tool_call",
                    "step": step_index + 1,
                    "index": call_index,
                    "tool_call_id": tool_call_id,
                    "name": tool_call["name"],
                    "arguments": tool_call.get("arguments", {}),
                }
            )
            # context：runtime 注入的工程字段，
            # 工具执行时据此做 namespace 隔离、审计 log 标记。
            context = {
                "run_id": run_id,
                "case_id": case_id,
                "rollout_id": rollout_id,
                "namespace_id": namespace_id,
                "tool_call_id": tool_call_id,
            }
            # 通过 toolfactory 执行：内部做 args 校验、读/写权限、sandbox 副作用、
            # 故障注入、生成 observation。runtime 自己不碰业务逻辑。
            observation = tool_factory.execute(
                tool_call["name"],
                tool_call["arguments"],
                env_snapshot=env_snapshot,
                sandbox=sandbox,
                context=context,
            )
            # tool_observations 保存完整 observation，包括 namespace_id 等审计字段；
            # 回放给模型时会通过 observation_message 做可见字段投影。
            trajectory.tool_observations.append(observation)
            _emit(
                {
                    "type": "tool_result",
                    "step": step_index + 1,
                    "index": call_index,
                    "tool_call_id": tool_call_id,
                    "name": tool_call["name"],
                    "ok": observation.get("ok"),
                    "result": observation.get("result"),
                    "error": observation.get("error"),
                    "message": observation.get("message"),
                    "source": observation.get("source"),
                }
            )
            # 工具执行失败（ok=False，可能是 environment 故障或 llm 用错）也单独记入 tool_errors。
            if observation.get("ok") is False:
                trajectory.tool_errors.append(observation)

            # 回放工具 observation（tool role），供下一步生成时模型看到。
            messages.append(observation_message(observation))
    else:
        # --- 终止方式二（撞 max_steps）---
        # for...else：循环把 limit 步全部跑满、一次都没 break（即始终在调工具、从未给出
        # final reply）。此时把 final_text 显式置空，表示这条 rollout 未能给出最终回复
        # （会反映到 max_step_hit_rate 等 route 指标，verifier 也据此判定未完成）。
        trajectory.final_text = ""

    # 无论如何终止，都导出 sandbox 最终状态，供 verifier 判定写动作是否真实发生且正确。
    trajectory.sandbox_final_state = sandbox.export()
    return trajectory.to_dict()


def _jsonable(value: Any) -> Any:
    """递归把任意值转成可 JSON 序列化的纯结构（dataclass→dict，tuple→list）。

    用途：给 prompt_history 里的 messages 做「深拷贝快照」。messages 在 loop 中会被
    不断 append/修改，若直接存引用，后续修改会回溯污染历史快照；这里递归重建出
    全新的 dict/list，既切断引用，又顺手把 dataclass / tuple 等不可直接落 JSON 的
    类型规整掉。标量原样返回。
    """
    if is_dataclass(value):
        # dataclass 先转 dict，再继续由下层递归处理字段值。
        return asdict(value)
    if isinstance(value, dict):
        # dict 重新构造，切断和原始 messages 的引用关系。
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        # list 逐项复制；messages 本身就是 list，这是最常走的分支。
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        # tuple 统一转 list：JSON 没有 tuple 概念，且能保证落盘/重读后类型一致。
        return [_jsonable(item) for item in value]
    return value


def _fallback_tool_call_id(step_index: int, call_index: int, total_calls: int) -> str:
    """给缺失 id 的工具调用生成稳定兜底 id；单工具时保持旧格式。"""

    step_number = step_index + 1
    if total_calls == 1:
        # 历史轨迹使用 tc_1、tc_2 这种格式；单工具步保持兼容。
        return f"tc_{step_number}"
    # 多工具步加上 call_index，避免同一步多个 observation 共享同一个 id。
    return f"tc_{step_number}_{call_index}"


def strip_reasoning_blocks(text: str) -> str:
    """从最终回复里剥掉 ``<think>...</think>`` 推理块。

    system prompt 要求「不要输出隐藏推理过程」，但模型仍可能带 <think> 块；最终
    面向客户的 final_text 不应包含这些内部推理，故在落 final_text 前清掉。
    re.DOTALL 让 <think> 体可跨行匹配，尾随空白一并去掉。
    """
    # 这里只处理 final/customer-facing 文本；raw_model_outputs 仍保留原始内容，方便审计。
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
