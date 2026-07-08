# 版权所有 © 2026 深圳途明智启科技有限公司。保留所有权利。
# 未经书面许可，任何单位或个人不得复制、传播、发布、转卖、改编、仿制或用于商业用途。
# 侵权必究。

"""verl AgentLoop adapter backed by this project's tool runtime.

The LLM generation stays inside verl's token-level rollout path. This adapter
only handles prompt/tool-history assembly, tool execution, verifier reward, and
artifact persistence.

一句话：verl 负责“生成 token 和训练模型”，这个 adapter 负责“看到 tool_call 后执行
业务工具、把 observation 追加回上下文、最后调用 verifier 得到 reward”。
"""

from __future__ import annotations

import json
import os
from typing import Any
from uuid import uuid4

from agent.observations import project_observation_for_model
from agent.prompts.templates import PROMPT_TEMPLATE_VERSION, prompt_hash, render_prompt, stable_hash
from agent.rollout_store import make_run_id
from agent.runtime import parse_tool_calls, strip_reasoning_blocks
from agent.trajectory import Trajectory
from envs.namespace import build_namespace_id
from envs.sandbox_state import SandboxState
from envs.toolfactory import ToolFactory
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

from train.verl_reward_adapter import rollout_metric_flags, score_and_persist_rollout


@register("industrial_posttrain_agent")
class IndustrialPosttrainAgentLoop(AgentLoopBase):
    """Token-level verl loop with project ToolFactory execution and verifier reward.

    注册名 ``industrial_posttrain_agent`` 必须和 ``configs/verl_agent_loop.yaml`` 以及
    ``scripts/train_grpo_verl.py`` 里的 default_agent_loop 保持一致。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # ToolFactory/Schema 在 worker 初始化时构造一次，避免每条 rollout 反复 import 全部工具。
        self.tool_factory = ToolFactory()
        self.tool_schemas = self.tool_factory.tool_schemas()
        # tool_schema_hash 写入 trajectory，用于排查 rollout 期间工具协议是否变过。
        self.tool_schema_hash = stable_hash(self.tool_schemas)
        # 这些长度来自 verl rollout config；adapter 用它们裁剪 response/token_trace。
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length
        self.max_assistant_turns = self.rollout_config.multi_turn.max_assistant_turns
        self.max_user_turns = self.rollout_config.multi_turn.max_user_turns
        # VERL_RUN_ID 由一键脚本设置。它决定 data/rollouts_verl/<run_id>/... 的落盘目录。
        self.run_id = os.environ.get("VERL_RUN_ID") or make_run_id("verl")

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        # raw_prompt 和 extra_info 由 verl RLHFDataset 从 parquet row 传入。
        # raw_prompt 是首轮 system/user messages；extra_info 里放 case/env/verifier 文件路径。
        messages = [dict(message) for message in kwargs["raw_prompt"]]
        extra_info = dict(kwargs.get("extra_info") or {})
        case = _read_json(extra_info["case_path"])
        env_snapshot = _read_json(extra_info["env_snapshot_path"])
        case_id = extra_info.get("case_id") or case["case_id"]
        # 同一个 case 会采样 rollout_n 条。这里给每条 rollout 加 uuid 后缀，保证 sandbox namespace 唯一。
        rollout_id = f"rollout_{extra_info.get('index', 0):04d}_{uuid4().hex[:8]}"
        namespace_id = build_namespace_id(self.run_id, case_id, rollout_id)
        sandbox = SandboxState.from_env_snapshot(env_snapshot, namespace_id)

        # prompt hash 与 standalone runtime 同口径：system + user 两段 prompt 计算指纹。
        phash = prompt_hash(messages[0]["content"], messages[1]["content"]) if len(messages) >= 2 else stable_hash(messages)
        trajectory = Trajectory(
            case_id=case_id,
            run_id=self.run_id,
            rollout_id=rollout_id,
            namespace_id=namespace_id,
            prompt_template_version=PROMPT_TEMPLATE_VERSION,
            prompt_hash=phash,
            tool_schema_hash=self.tool_schema_hash,
        )

        # 多模态接口保持兼容；当前售后文本 case 通常没有图片/音频/视频。
        multi_modal_data = await self.process_multi_modal_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")
        audios = multi_modal_data.get("audios")
        mm_processor_kwargs = self._get_mm_processor_kwargs(audios)

        prompt_ids = await self.apply_chat_template(
            messages,
            tools=self.tool_schemas,
            images=images,
            videos=videos,
            audios=audios,
            mm_processor_kwargs=mm_processor_kwargs,
        )
        # all_token_ids 是发给 vLLM 的完整上下文 token；response_ids/mask 是返回给 verl 训练的 response 部分。
        all_token_ids = list(prompt_ids)
        response_ids: list[int] = []
        # response_mask=1 表示模型生成 token，要参与 logprob/训练；tool observation 等非模型 token 标 0。
        response_mask: list[int] = []
        response_logprobs: list[float] = []
        # token_trace 是本项目额外落盘的审计结构：能看每段 token 属于 prompt/model/tool/feedback。
        token_trace = {
            "prompt_ids": list(prompt_ids),
            "segments": [{"type": "prompt", "token_count": len(prompt_ids), "mask": None}],
        }

        metrics: dict[str, Any] = {}
        request_id = uuid4().hex
        assistant_turns = 0
        user_turns = 0
        routed_experts = None

        while len(response_mask) < self.response_length:
            # verl multi_turn 的硬上限保护，避免模型无限工具循环。
            if self.max_assistant_turns and assistant_turns >= self.max_assistant_turns:
                break
            if self.max_user_turns and user_turns >= self.max_user_turns:
                break

            # 和 standalone runtime 一样：每次生成前先保存模型真实看到的 messages/tool_schemas。
            trajectory.prompt_history.append(
                {
                    "step": assistant_turns + 1,
                    "messages": _jsonable(messages),
                    "tool_schemas": self.tool_schemas,
                    "prompt_hash": phash,
                    "tool_schema_hash": self.tool_schema_hash,
                }
            )

            with simple_timer("generate_sequences", metrics):
                # 关键生成调用：token 仍由 verl/vLLM server_manager 产生，不是旁路调用外部模型。
                output: TokenOutput = await self.server_manager.generate(
                    request_id=request_id,
                    prompt_ids=all_token_ids,
                    sampling_params=sampling_params,
                    image_data=images,
                    video_data=videos,
                    audio_data=audios,
                    mm_processor_kwargs=mm_processor_kwargs,
                )
            assistant_turns += 1
            # num_preempted/routed_experts 是 verl/vLLM 的性能诊断信息，透传到 metrics/AgentLoopOutput。
            if metrics.get("num_preempted") is None:
                metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
            elif output.num_preempted:
                metrics["num_preempted"] += output.num_preempted
            if output.routed_experts is not None:
                routed_experts = output.routed_experts

            # generated_ids 是本轮 assistant 生成的 token；raw_text 保留 special token，clean_text 去掉 special token。
            generated_ids = list(output.token_ids)
            raw_text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
            clean_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            # 模型生成 token 进入 response_ids，mask 标 1，训练会对这些 token 计算 logprob/优势。
            all_token_ids.extend(generated_ids)
            response_ids.extend(generated_ids)
            response_mask.extend([1] * len(generated_ids))
            if output.log_probs:
                response_logprobs.extend(output.log_probs)
            token_trace["segments"].append(
                {
                    "type": "model",
                    "step": assistant_turns,
                    "token_count": len(generated_ids),
                    "mask": 1,
                    "text": raw_text,
                    "token_ids": generated_ids,
                }
            )
            # raw_model_outputs 保留每轮原始文本，后面无论 parse 成功/失败都能复盘。
            trajectory.raw_model_outputs.append(
                {
                    "step": assistant_turns,
                    "raw_text": raw_text,
                    "clean_text": clean_text,
                    "tool_calls": [],
                    "stop_reason": output.stop_reason,
                    "extra_fields": output.extra_fields,
                }
            )

            # 本项目工具协议：从模型文本里解析 <tool_call> JSON。原生 provider tool_calls 在 verl path 不走这里。
            tool_calls, parse_error = parse_tool_calls(raw_text)
            messages.append({"role": "assistant", "content": raw_text})
            if parse_error:
                # 格式错只反馈给模型，不执行任何工具；该错误 source=llm，会进入 reward/metrics。
                error_observation = {
                    "ok": False,
                    "error": "parse_error",
                    "message": parse_error,
                    "source": "llm",
                    "tool_call_id": f"tc_{assistant_turns}",
                }
                trajectory.tool_errors.append(error_observation)
                feedback = render_prompt("tool_error_feedback.txt", {"error_observation": error_observation})
                # feedback 是 user 消息，不是模型生成，所以追加 token 时 response_mask=0。
                feedback_ids = await self._append_non_model_messages(
                    [{"role": "user", "content": feedback}],
                    messages,
                    all_token_ids,
                    response_ids,
                    response_mask,
                    response_logprobs,
                    token_trace,
                    segment_type="parse_error_feedback",
                    step=assistant_turns,
                )
                user_turns += 1 if feedback_ids else 0
                continue

            if not tool_calls:
                # 没有 tool_call 且没有 parse_error：认为模型给出了最终客户回复，结束 multi-turn loop。
                trajectory.final_text = strip_reasoning_blocks(clean_text).strip()
                break

            trajectory.raw_model_outputs[-1]["tool_calls"] = tool_calls
            add_messages = []
            for call_index, tool_call in enumerate(tool_calls, start=1):
                # 多工具同步保留 call_index；协议违规由 verifier 通过 parsed_actions 评分，不在 adapter 截断。
                tool_call_id = tool_call.get("id") or _tool_call_id(assistant_turns, call_index, len(tool_calls))
                parsed = {
                    "step": assistant_turns,
                    "tool_call_index": call_index,
                    "tool_call_id": tool_call_id,
                    "name": tool_call["name"],
                    "arguments": tool_call.get("arguments", {}),
                }
                trajectory.parsed_actions.append(parsed)
                # 真正执行业务工具：读 env_snapshot，写本 rollout 的 sandbox。
                observation = self.tool_factory.execute(
                    parsed["name"],
                    parsed["arguments"],
                    env_snapshot=env_snapshot,
                    sandbox=sandbox,
                    context={
                        "run_id": self.run_id,
                        "case_id": case_id,
                        "rollout_id": rollout_id,
                        "namespace_id": namespace_id,
                        "tool_call_id": tool_call_id,
                    },
                )
                trajectory.tool_observations.append(observation)
                if observation.get("ok") is False:
                    trajectory.tool_errors.append(observation)
                # tool observation 回放成 tool role 消息；content 只给模型可见投影，完整 observation 留在 trajectory。
                add_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": parsed["name"],
                        "content": json.dumps(
                            project_observation_for_model(observation),
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                    }
                )

            with simple_timer("tool_calls", metrics):
                # tool observation 不是模型生成 token，追加到上下文时 response_mask=0。
                tool_ids = await self._append_non_model_messages(
                    add_messages,
                    messages,
                    all_token_ids,
                    response_ids,
                    response_mask,
                    response_logprobs,
                    token_trace,
                    segment_type="tool_observation",
                    step=assistant_turns,
                )
            user_turns += 1 if tool_ids else 0

        # loop 结束后，sandbox_final_state 是 verifier 判断写动作是否真实发生的事实来源。
        trajectory.sandbox_final_state = sandbox.export()
        token_trace["response_mask"] = response_mask[: self.response_length]
        token_trace["response_token_count"] = len(response_ids[: self.response_length])
        # score_and_persist_rollout 会调用 verifier/LLM judge，并把 trajectory、score、token_trace 全部落盘。
        score, artifact_dir = score_and_persist_rollout(
            trajectory=trajectory.to_dict() | {"token_trace": token_trace},
            extra_info=extra_info,
            token_trace=token_trace,
            overwrite=True,
        )
        flags = rollout_metric_flags(trajectory.to_dict(), score)
        # reward_extra_info 必须保持 numpy-stackable；复杂结构放 artifact 文件里。
        reward_extra_info = _reward_extra_info_for_verl(score=score, flags=flags, artifact_dir=str(artifact_dir))
        metrics.update(
            {
                "reward": score.get("reward", 0.0),
                "raw_reward": score.get("raw_reward", 0.0),
                "num_actions": flags["num_actions"],
                "num_tool_errors": flags["num_tool_errors"],
            }
        )

        # AgentLoopOutput 是 verl 训练侧真正消费的对象：response token、mask、reward、metrics 都从这里返回。
        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=response_logprobs[: self.response_length] if response_logprobs else None,
            routed_experts=(
                routed_experts[: len(prompt_ids) + self.response_length]
                if routed_experts is not None
                else None
            ),
            multi_modal_data=multi_modal_data,
            mm_processor_kwargs=mm_processor_kwargs,
            reward_score=score.get("reward", 0.0),
            num_turns=assistant_turns + user_turns + 1,
            metrics=metrics,
            extra_fields={
                "turn_scores": [],
                "tool_rewards": [],
                "reward_extra_info": reward_extra_info,
                "artifact_dir": str(artifact_dir),
                "rollout_flags": flags,
                "token_trace": token_trace,
            },
        )
        return output

    async def _append_non_model_messages(
        self,
        add_messages: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        all_token_ids: list[int],
        response_ids: list[int],
        response_mask: list[int],
        response_logprobs: list[float],
        token_trace: dict[str, Any],
        *,
        segment_type: str,
        step: int,
    ) -> list[int]:
        """Append tool/feedback messages to both chat history and token history.

        这些 token 是环境/用户反馈，不是模型采样出来的，所以 response_mask 填 0。
        这正是 token-level verl 与 multi-turn tool runtime 对齐的关键：同一条 response
        序列里既有模型 token(mask=1)，也有工具/反馈 token(mask=0)。
        """

        if not add_messages:
            return []
        messages.extend(add_messages)
        # remove_system_prompt=True 表示这里只渲染新增消息，不重复首轮 system prompt。
        token_ids = await self.apply_chat_template(add_messages, remove_system_prompt=True)
        remaining = self.response_length - len(response_mask)
        # response_length 是 verl 的硬预算；非模型消息也会占上下文窗口，因此同样要裁剪。
        token_ids = token_ids[:remaining]
        all_token_ids.extend(token_ids)
        response_ids.extend(token_ids)
        response_mask.extend([0] * len(token_ids))
        if response_logprobs:
            response_logprobs.extend([0.0] * len(token_ids))
        token_trace["segments"].append(
            {
                "type": segment_type,
                "step": step,
                "token_count": len(token_ids),
                "mask": 0,
                "messages": _jsonable(add_messages),
                "token_ids": token_ids,
            }
        )
        return token_ids


def _tool_call_id(step: int, call_index: int, total_calls: int) -> str:
    """生成缺省 tool_call_id，保持单工具步的历史格式。"""

    if total_calls == 1:
        return f"tc_{step}"
    return f"tc_{step}_{call_index}"


def _reward_extra_info_for_verl(*, score: dict[str, Any], flags: dict[str, Any], artifact_dir: str) -> dict[str, Any]:
    """Keep verl reward_extra_info numpy-stackable; rich records stay in artifacts."""

    return {
        "raw_reward": float(score.get("raw_reward", 0.0) or 0.0),
        "active_cap_count": len(score.get("active_caps") or []),
        "active_caps_json": json.dumps(score.get("active_caps") or [], ensure_ascii=False, sort_keys=True),
        "subscores_json": json.dumps(score.get("subscores") or {}, ensure_ascii=False, sort_keys=True),
        "artifact_dir": artifact_dir,
        "parse_error": int(bool(flags.get("parse_error"))),
        "tool_error_llm": int(flags.get("tool_error_llm") or 0),
        "max_step_hit": int(bool(flags.get("max_step_hit"))),
        "num_actions": int(flags.get("num_actions") or 0),
        "num_tool_errors": int(flags.get("num_tool_errors") or 0),
    }


def _read_json(path: str) -> Any:
    """读取 extra_info 中传来的 JSON 文件路径。"""

    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _jsonable(value: Any) -> Any:
    """递归复制成 JSON-friendly 结构，用于 prompt_history/token_trace 落盘。"""

    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return value
