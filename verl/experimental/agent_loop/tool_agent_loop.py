# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import json
import logging
import os
from typing import Any
from uuid import uuid4
import copy
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.utils.reward_score.deepresearch import check_tags_balance_simple, compute_score
import time
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


@register("tool_agent")
class ToolAgentLoop(AgentLoopBase):
    @classmethod
    def init_class(cls, config, tokenizer, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        print("Performing class-level ToolAgentLoop initialization")

        # Initialize tools from config file
        cls.tokenizer = tokenizer
        cls.max_user_turns = config.actor_rollout_ref.rollout.multi_turn.max_user_turns
        cls.max_assistant_turns = config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns
        cls.max_parallel_calls = config.actor_rollout_ref.rollout.multi_turn.max_parallel_calls
        cls.max_tool_response_length = config.actor_rollout_ref.rollout.multi_turn.max_tool_response_length
        cls.tool_response_truncate_side = config.actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side
        tool_config_path = config.actor_rollout_ref.rollout.multi_turn.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        cls.tools = {tool.name: tool for tool in tool_list}
        cls.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
        cls.tool_parser = ToolParser.get_tool_parser(config.actor_rollout_ref.rollout.multi_turn.format, cls.tokenizer)
        cls.plan_mask_weight = config.trainer.plan_mask_weight if hasattr(config.trainer, 'plan_mask_weight') else 1.0
        print(f"Initialized tools: {cls.tools}")

        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length
        cls.system_prompt = tokenizer.apply_chat_template([{}], add_generation_prompt=False, tokenize=True)

    @rollout_trace_op
    async def run(self, messages: list[dict[str, Any]], sampling_params: dict[str, Any], tools_kwargs: dict[str, Any], extra_info: dict[str, Any] = {}, meta_info: dict[str, Any] = {}) -> AgentLoopOutput:
        metrics = {}
        request_id = uuid4().hex

        prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                messages, tools=self.tool_schemas, add_generation_prompt=True, tokenize=True
            ),
        )

        prompt_text = self.tokenizer.decode(prompt_ids)

        response_mask = []
        
        user_turns, assistant_turns = 0, 0
        max_try_times = 30
        while True:
            max_try_times -= 1
            if max_try_times <= 0:
                logger.warning(f"Reached maximum try times, breaking the loop.")
                break

            with simple_timer("generate_sequences", metrics):
                response_ids = await self.server_manager.generate(
                    request_id=request_id, prompt_ids=prompt_ids, sampling_params=sampling_params
                )
            prompt_ids += response_ids            
            response_mask += [1] * len(response_ids)
            response = self.tokenizer.decode(prompt_ids[-len(response_mask):])
            
            assistant_turns += 1

            if not check_tags_balance_simple(response):
                logger.warning(f"Unbalanced tags in response, breaking the loop. for request_id: {request_id}")
                break

            # break
            # reach max response length
            if len(response_mask) >= self.response_length:
                break

            # reach max assistant turns
            if self.max_assistant_turns and assistant_turns >= self.max_assistant_turns:
                break

            # reach max user turns
            if self.max_user_turns and user_turns >= self.max_user_turns:
                break

            # no tool calls
            _, tool_calls = await self.tool_parser.extract_tool_calls(response_ids)
            current_response_text = self.tokenizer.decode(response_ids)

            if "<plan>" in current_response_text and "</plan>" in current_response_text and "<tool_call>" in current_response_text and "</tool_call>" in current_response_text:
                logger.info("Found both <plan> and <tool_call> in response, breaking the loop.")
                break
            if assistant_turns == 1 and "<plan>" not in current_response_text:
                logger.info("No <plan> found in the first assistant turn, breaking the loop.")
                break

            if not tool_calls:  
                # logger.warning(f"No tool calls found in response: {current_response_text}")
                if "</plan>" in current_response_text and "<plan>" not in current_response_text:
                    current_response_text = "<plan>\n" + current_response_text
                last_tag = get_last_tag_pair(current_response_text)

                if last_tag and last_tag[0].lower() == "plan":
                    think_tag = self.tokenizer.encode("\n<|im_start|>assistant\n<think>\n")
                    prompt_ids += think_tag
                    response_mask += [1] * len(think_tag)
                    logger.info("Last tag is </plan>, continuing to generate.")
                    continue
                else:
                    break

            # break
            # call tools
            try:
                tasks = []
                if len(tool_calls) > self.max_parallel_calls:
                    logger.warning(
                        f"Too many tool calls ({len(tool_calls)}), only processing the first {self.max_parallel_calls}."
                    )
                    logger.warning(f"Tool calls: {[tool_call.name for tool_call in tool_calls]}")
                    logger.warning(f"Full response text: {current_response_text}")
                # logger.info(f"Processing {len(tool_calls)} tool calls with request_id: {request_id}")
                for tool_call in tool_calls[: self.max_parallel_calls]:
                    tool_name = tool_call.name
                    if tool_name not in tools_kwargs:
                        tools_kwargs[tool_name] = {"create_kwargs": {}}
                    tasks.append(self._call_tool(tool_call, tools_kwargs, request_id=request_id))

                with simple_timer("tool_calls", metrics):
                    tool_responses = await asyncio.gather(*tasks, return_exceptions=True)
                if any(isinstance(item, Exception) for item in tool_responses):
                    logger.error(f"One or more tool calls failed, breaking the loop.")
                    break
            except Exception as e:
                logger.error(f"Error during tool calls: {e}")
                break

            tool_response_ids = await self.loop.run_in_executor(
                None,
                lambda messages=tool_responses: self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True
                ),
            )

            tool_response_ids = tool_response_ids[len(self.system_prompt) :]
            # NOTE: last turn should not be user turn, or the EOS token reward
            # can't be propagated to previous token in GAE.
            if len(response_mask) + len(tool_response_ids) >= self.response_length:
                break

            prompt_ids += tool_response_ids
            response_mask += [0] * len(tool_response_ids)
            user_turns += 1

        response_ids = prompt_ids[-len(response_mask) :]

        try:
            response_text = self.tokenizer.decode(response_ids, skip_special_tokens=False)
            pieces = []
            offsets = []
            cur = 0

            for tid in response_ids:
                piece = self.tokenizer.decode([tid], skip_special_tokens=False)
                pieces.append(piece)
                start = cur
                end = cur + len(piece)
                offsets.append((start, end))
                cur = end

            response_text = "".join(pieces)

            plan_pattern = re.compile(r"<think>.*?</think>\s*<plan>.*?</plan>", re.DOTALL)
            n = min(len(response_ids), len(response_mask), len(offsets))
            logger.warning(f"response_ids length: {len(response_ids)}, response_mask length: {len(response_mask)}, offsets length: {len(offsets)}")
            for match in plan_pattern.finditer(response_text):
                span_start, span_end = match.span()
                for idx in range(n):
                    tok_start, tok_end = offsets[idx]
                    if tok_end <= span_start:
                        continue
                    if tok_start >= span_end:
                        break
                    response_mask[idx] = self.plan_mask_weight


        except Exception as e:
            logger.error(f"Failed to apply <plan> mask after generation: {e}")


        prompt_ids = prompt_ids[: len(prompt_ids) - len(response_mask)]

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            num_turns=assistant_turns,
            metrics=metrics,
        )
        return output

    async def _call_tool(self, tool_call: FunctionCall, tools_kwargs: dict[str, Any], request_id: str) -> dict[str, str]:
        """Call tool and return tool response."""
        tool, instance_id = None, None
        try:
            # TODO: append malformed tool_call to the prompt: invalid function name or arguments
            tool_name = tool_call.name
            tool_args = json.loads(tool_call.arguments)
            tool = self.tools[tool_name]
            kwargs = tools_kwargs.get(tool_name, {})
            instance_id = await tool.create(instance_id=request_id, create_kwargs=kwargs.get("create_kwargs", {}))
            tool_response, rid, _ = await tool.execute(instance_id, tool_args)
        except Exception as e:
            logger.error(f"Error when executing tool: {e}")
            tool_response = f"Error when executing tool {tool_call.name}: {e}"
        finally:
            if tool and instance_id:
                await tool.release(instance_id)

        if len(tool_response) > self.max_tool_response_length:
            if self.tool_response_truncate_side == "left":
                tool_response = tool_response[: self.max_tool_response_length] + "...(truncated)"
            elif self.tool_response_truncate_side == "right":
                tool_response = "(truncated)..." + tool_response[-self.max_tool_response_length :]
            else:
                length = self.max_tool_response_length // 2
                tool_response = tool_response[:length] + "...(truncated)..." + tool_response[-length:]
        
        return {
            "role": "tool",
            "content": tool_response,
        }

    def _start_new_assistant_turn_with_think(self, prompt_ids, response_mask, plan_turn=False):
        need_think = True
        if need_think:
            if plan_turn:
                think_ids = self.tokenizer.encode("\n<|im_start|>assistant\n<think>\n")
            else:
                think_ids = self.tokenizer.encode("<think>\n")
            prompt_ids += think_ids
            response_mask += [1] * len(think_ids)

        return prompt_ids, response_mask


import re
def get_last_tag_pair(text: str) -> tuple[str, str] | None:
    pattern = re.compile(r"<([a-zA-Z0-9_]+)>(.*?)</\1>", re.DOTALL)
    matches = list(pattern.finditer(text))
    if matches:
        last_match = matches[-1]
        tag = last_match.group(1)
        content = last_match.group(2)
        return tag, content
    return None

