import json
from typing import Dict, Iterator, List, Literal, Optional, Tuple, Union

from qwen_agent.agents.fncall_agent import FnCallAgent
from qwen_agent.llm import BaseChatModel
from qwen_agent.llm.schema import ASSISTANT, DEFAULT_SYSTEM_MESSAGE, Message
from qwen_agent.settings import MAX_LLM_CALL_PER_RUN
from qwen_agent.tools import BaseTool
from qwen_agent.utils.utils import format_as_text_message, merge_generate_cfgs
from openai import OpenAI
import time
from prompts import *


TOOL_DESC = (
    '{name_for_model}: Call this tool to interact with the {name_for_human} API. '
    'What is the {name_for_human} API useful for? {description_for_model} Parameters: {parameters} {args_format}')

class WebWalker(FnCallAgent):
    """This explorer agent use ReAct format to call tools"""

    def __init__(self,
                 function_list: Optional[List[Union[str, Dict, BaseTool]]] = None,
                 llm: Optional[Union[Dict, BaseChatModel]] = None,
                 system_message: Optional[str] = DEFAULT_SYSTEM_MESSAGE,
                 name: Optional[str] = None,
                 description: Optional[str] = None,
                 files: Optional[List[str]] = None,
                 **kwargs):
        super().__init__(function_list=function_list,
                         llm=llm,
                         system_message=system_message,
                         name=name,
                         description=description,
                         files=files,
                         **kwargs)
        self.extra_generate_cfg = merge_generate_cfgs(
            base_generate_cfg=self.extra_generate_cfg,
            new_generate_cfg={'stop': ['Observation:', 'Observation:\n']},
        )
        self.client = OpenAI(
            api_key=llm['api_key'], 
            base_url=llm['model_server'],
        )
        self.llm_cfg = llm
        self.momery = []

    def observation_information_extraction(self, query, observation):
        user_prompt = "- Query: {query}\n- Observation: {observation}".format(query=query, observation=observation)
        messages = [
            {'role': 'system', 'content': STSTEM_CRITIIC_INFORMATION},
            {'role': 'user', 'content':  user_prompt}]
        max_retries = 10
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.llm_cfg['model'],
                    response_format={"type": "json_object"},
                    messages=messages
                )
                print(response.choices[0].message.content)
                # response_content = json.loads(response.choices[0].message.content)
                if "true" in response.choices[0].message.content:
                    try:
                        return json.loads(response.choices[0].message.content)["information"]
                    except:
                        return response.choices[0].message.content
                else:
                    return None
            except Exception as e:
                print(e)
                if attempt < max_retries - 1:
                    time.sleep(1 * (2 ** attempt))  # Exponential backoff
                else:
                    raise e  # Raise the exception if the last retry fails

    def critic_information(self, query, memory):  
        memory = "-".join(memory)
        user_prompt = "- Query: {query}\n- Accumulated Information: {memory}".format(query = query, memory=memory)
        messages = [
            {'role': 'system', 'content': STSTEM_CRITIIC_ANSWER},
            {'role': 'user', 'content': user_prompt}]
        response = self.client.chat.completions.create(
            model=self.llm_cfg['model'],
            response_format={"type": "json_object"},
            messages=messages
        )
        max_retries = 10
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.llm_cfg['model'],
                    response_format={"type": "json_object"},
                    messages=messages
                )
                print(response.choices[0].message.content)
                if "true" in response.choices[0].message.content:
                    try:
                        return json.loads(response.choices[0].message.content)["answer"]
                    except:
                        return response.choices[0].message.content
                else:
                    return None
            
            except Exception as e:
                print(e)
                if attempt < max_retries - 1:
                    time.sleep(1 * (2 ** attempt))  # Exponential backoff
                else:
                    raise e  # Raise the exception if the last retry fails

    def _run(self, messages: List[Message], lang: Literal['en', 'zh'] = 'en', **kwargs) -> Iterator[List[Message]]:
        text_messages = self._prepend_react_prompt(messages, lang=lang)
        num_llm_calls_available = MAX_LLM_CALL_PER_RUN
        response: str = 'Thought: '
        query = self.llm_cfg["query"]
        action_count = self.llm_cfg.get("action_count", MAX_LLM_CALL_PER_RUN)
        num_llm_calls_available = action_count
        while num_llm_calls_available > 0:
            num_llm_calls_available -= 1
            output = []
            for output in self._call_llm(messages=text_messages):
                if output:
                    yield [Message(role=ASSISTANT, content=output[-1].content)]
            # Accumulate the current response
            if output:
                response += output[-1].content

            has_action, action, action_input, thought = self._detect_tool("\n"+output[-1].content)
            if not has_action:
                if "Final Answer: " in output[-1].content:
                    break
                else:
                    continue

            # Add the tool result
            query = self.llm_cfg["query"]
            observation = self._call_tool(action, action_input, messages=messages, **kwargs)
            stage1 = self.observation_information_extraction(query, observation)
            if stage1:
                self.momery.append(stage1+"\n")
                if len(self.momery) > 1:
                    yield [Message(role=ASSISTANT, content= "Memory:\n" + "-".join(self.momery)+"\"}")]
                else:
                    yield [Message(role=ASSISTANT, content= "Memory:\n" + "-" + self.momery[0]+"\"}")]
                stage2 = self.critic_information(query, self.momery)
                if stage2:
                    response = f'Final Answer: {stage2}'
                    yield [Message(role=ASSISTANT, content=response)]
                    break


            observation = f'\nObservation: {observation}\nThought: '
            response += observation
            # yield [Message(role=ASSISTANT, content=response)]

            if (not text_messages[-1].content.endswith('\nThought: ')) and (not thought.startswith('\n')):
                # Add the '\n' between '\nQuestion:' and the first 'Thought:'
                text_messages[-1].content += '\n'
            if action_input.startswith('```'):
                # Add a newline for proper markdown rendering of code
                action_input = '\n' + action_input
            text_messages[-1].content += thought + f'\nAction: {action}\nAction Input: {action_input}' + observation
            # print(text_messages[-1].content)

    def _prepend_react_prompt(self, messages: List[Message], lang: Literal['en', 'zh']) -> List[Message]:
        tool_descs = []
        for f in self.function_map.values():
            function = f.function
            name = function.get('name', None)
            name_for_human = function.get('name_for_human', name)
            name_for_model = function.get('name_for_model', name)
            assert name_for_human and name_for_model
            args_format = function.get('args_format', '')
            tool_descs.append(
                TOOL_DESC.format(name_for_human=name_for_human,
                                 name_for_model=name_for_model,
                                 description_for_model=function['description'],
                                 parameters=json.dumps(function['parameters'], ensure_ascii=False),
                                 args_format=args_format).rstrip())
        tool_descs = '\n\n'.join(tool_descs)
        tool_names = ','.join(tool.name for tool in self.function_map.values())
        text_messages = [format_as_text_message(m, add_upload_info=True, lang=lang) for m in messages]
        text_messages[-1].content = SYSTEM_EXPLORER.format(
            tool_descs=tool_descs,
            tool_names=tool_names,
            query=text_messages[-1].content,
        )
        return text_messages

    def _detect_tool(self, text: str) -> Tuple[bool, str, str, str]:
        special_func_token = '\nAction:'
        special_args_token = '\nAction Input:'
        special_obs_token = '\nObservation:'
        func_name, func_args = None, None
        i = text.rfind(special_func_token)
        j = text.rfind(special_args_token)
        k = text.rfind(special_obs_token)
        if 0 <= i < j:  # If the text has `Action` and `Action input`,
            if k < j:  # but does not contain `Observation`,
                # then it is likely that `Observation` is ommited by the LLM,
                # because the output text may have discarded the stop word.
                text = text.rstrip() + special_obs_token  # Add it back.
            k = text.rfind(special_obs_token)
            func_name = text[i + len(special_func_token):j].strip()
            func_args = text[j + len(special_args_token):k].strip()
            text = text[:i]  # Return the response before tool call, i.e., `Thought`
        return (func_name is not None), func_name, func_args, text
