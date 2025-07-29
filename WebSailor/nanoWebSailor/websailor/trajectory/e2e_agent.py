"""
@File   : e2e_agent_workflow.py
@Time   : 2025/05/16 10:46
@Author : yliu.lyndon
@Desc   : None
"""
# %%
import json
import re
from datetime import datetime
from typing import Callable

import requests
from loguru import logger

_SYSTEM_PROMPT = """\
You are Alice, an intelligent assistant who specializes in utilizing various tools to help answer user questions.
You MUST follow the Reasoning and Acting (ReAct) Protocol to iteratively make tool calls to achieve the user's purpose.
You should also be aware of the task requirements during ReAct process.

# Reasoning and Acting Protocol
Step 1. Understand the user question and available tools thoroughly, create a plan by breaking down complex questions into manageable steps. Enclose your plan within <plan></plan> XML tags.
Step 2. Think or reflect about your reasoning before using a tool. Enclose your thinking within <think></think> XML tags.
Step 3. Make an appropriate tool call with correct parameters. Enclose your tool call within <tool_call></tool_call> XML tags.
Step 4. Review the tool response that will be enclosed within <tool_response></tool_response> XML tags.
... (Repeat Step 2-4 as needed until you have sufficient information.)
Last Step. When you think all the necessary information has been exhaustively gathered, provide the answer to the user's question within <answer></answer> XML tags.

# Task Requirements
- For complex questions, ensure sufficient tool interaction depth to gather all relevant information.
- If tools cannot solve a problem, clearly explain the limitations and suggest alternatives.\
"""


def generate_for_tooluse(messages: list[dict], tools: list[dict], **kwargs) -> tuple[str, str, list]:
    """
    Generate tool usage response from the model.
    """
    # Configure URL
    url = "http://10.200.64.10/10-e2e-sft-model/v1/chat/completions"
    # Configure request headers
    headers = {
        "Content-Type": "application/json;charset=utf-8",
    }
    # Configure request body
    body = {
        "model": "qwen2d5-7b-e2e",
        "max_tokens": kwargs.get("max_tokens", 8192),
        "temperature": kwargs.get("temperature", 0.6),
        "stream": False,
        "messages": messages,
        "tools": tools,
    }
    # Send POST request
    response = requests.post(url=url, data=json.dumps(body), headers=headers)
    # Handle request exceptions
    if response.status_code != 200:
        raise Exception("Request failed!", f"Request status code: {response.status_code}", f"Response data: {response.text}")
    # Parse response data
    thinking_content, answer_content, tool_calls = "", "", []
    response_data = response.json()
    try:
        message = response_data["choices"][0]["message"]
        # thinking_content = message.get("reasoning_content", "")
        # print(f"<think>\n{thinking_content}\n</think>")
        answer_content = message.get("content", "")
        # print(f"<text>\n{answer_content}\n</text>")
        # tool_calls = message.get("tool_calls", [])
        # print(f"<tool>\n{tool_calls}\n</tool>")
    except Exception as exc:
        if "调用Alice审计服务未通过！" in response_data.get("message", ""):
            raise PermissionError("Alice audit service call failed!", response_data) from exc
        logger.error("Request exception! Exception reason: {}, Response data: {}", exc, response_data)
    return thinking_content, answer_content, tool_calls


def get_datetime(date_string: str) -> str:
    """Convert date string to formatted datetime string with weekday."""
    weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    date_time = datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S")
    date_time_string = f"{date_string} {weekdays[date_time.weekday()]}"
    return date_time_string


class E2EAgent:

    def __init__(self, system: str = "", max_turns: int = 32, retry_times: int = 3, **kwargs) -> None:
        """Initialize the E2E Agent."""
        self.system = system if system else _SYSTEM_PROMPT
        self.max_turns = max_turns
        self.retry_times = retry_times

    def execute(self, print_log: bool, query: str, tool_desc_list: list[dict], tool_caller: dict[str, Callable] = {}) -> list[dict]:
        """
        Execute the agent.
        """
        conversations = []
        turns = 0

        messages = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": query},
        ]

        conversations.append({"from": "query", "value": query})

        while turns < self.max_turns:

            # Call the model
            thinking_content, answer_content, tool_calls = self.invoke(print_log, messages, tool_desc_list)
            if print_log:
                logger.info("Model planning: {}", answer_content)
            # Update turn count
            turns += 1

            # Parse global plan >>> <plan>
            if turns == 1:
                matches = re.search(r"<plan>(.*)</plan>", answer_content, re.DOTALL)
                if not matches:
                    raise Exception("Parsing failed!", "<plan> content does not exist!")
                else:
                    plan = matches.group(1).strip()
                conversations.append({"from": "plan", "value": plan})

            # Parse final answer >>> <answer>
            if "<tool_call>" not in answer_content:
                matches = re.search(r"<answer>(.*)</answer>", answer_content, re.DOTALL)
                if not matches:
                    raise Exception("Parsing failed!", "<answer> content does not exist!")
                final_answer = matches.group(1).strip()
                matches = re.search(r"<think>(.*)</think>", answer_content, re.DOTALL)
                thought = matches.group(1).strip() if matches else ""
                conversations.append({"from": "think", "value": thought})
                conversations.append({"from": "final_answer", "value": final_answer})
                break

            # Parse current decision >>> <think>
            matches = re.search(r"<think>(.*)</think>", answer_content, re.DOTALL)
            if not matches:
                raise Exception("Parsing failed!", "<think> content does not exist!")
            thought = matches.group(1).strip()
            conversations.append({"from": "think", "value": thought})

            # Parse tool calls
            matches = re.search(r"<tool_call>(.*)</tool_call>", answer_content, re.DOTALL)
            if not matches:
                raise Exception("Parsing failed!", "<tool_call> content does not exist!")
            actions = matches.group(1).strip()
            try:
                try:
                    actions = json.loads(actions)
                except:
                    actions = eval(actions)
                if not isinstance(actions, list):
                    raise Exception("<tool_call> format does not meet requirements!")
            except Exception as exc:
                raise Exception("Parsing failed!", f"Failure reason: {exc}", f"Data: {actions}")
            conversations.append({"from": "action", "value": actions})

            # Tool execution results
            observations = []
            for action in actions:
                action_name = action.get("name", "")
                action_args = action.get("arguments", {})
                if action_name not in tool_caller:
                    raise Exception("Process exception!", f"Tool `{action_name}` does not exist!")
                try:
                    observation = tool_caller[action_name](action_args)
                    if print_log:
                        logger.info(f"Tool execution result: {observation}")
                except Exception as exc:
                    observation = f"Tool execution failed! Failure reason: {exc}"
                    if print_log:
                        logger.warning(f"Tool execution failed: {str(exc)}")
                if isinstance(observation, (dict, list)):
                    observation = json.dumps(observation, ensure_ascii=False)
                else:
                    observation = str(observation)
                observations.append({"tool_name": action_name, "tool_result": observation})
            conversations.append({"from": "observation", "value": json.dumps(observation, ensure_ascii=False)})

            # Update conversation history
            assistant_blk = {
                "role": "assistant",
                # "reasoning_content": thinking_content,
                "content": answer_content,
                # "tool_calls": tool_calls,
            }
            tool_blk_list = [
                {"role": "tool", "content": item["tool_result"]}
                for item in observations
            ]
            messages.extend([assistant_blk, *tool_blk_list])

        return conversations
    
    def invoke(self, print_log: bool, messages: list[dict], tool_desc_list: list[dict]) -> tuple[str, str, list[dict]]:
        """Invoke the model with retry mechanism."""
        thinking_content, answer_content, tool_calls = "", "", []
        for i in range(self.retry_times+1):
            if i > 0:
                if print_log:
                    logger.warning("Starting retry attempt {}...", i)
            try:
                thinking_content, answer_content, tool_calls = generate_for_tooluse(messages, tools=tool_desc_list)
                if not answer_content:
                    if print_log:
                        logger.error("Model inference exception! Exception reason: Content does not exist! Model response: {}", answer_content)
                    continue
                break
            except Exception as exc:
                if print_log:
                    logger.error("Model inference exception! Exception reason: {}", exc)
        else:
            raise Exception("Model inference failed!", "Maximum retry attempts reached!")
        return thinking_content, answer_content, tool_calls