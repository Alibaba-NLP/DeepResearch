import copy
import json
import logging
import os
from http import HTTPStatus
from pprint import pformat
from typing import Dict, Iterator, List, Optional, Literal, Union

import openai

if openai.__version__.startswith('0.'):
    from openai.error import OpenAIError  # noqa
else:
    from openai import OpenAIError

from qwen_agent.llm.base import ModelServiceError, register_llm
from qwen_agent.llm.function_calling import BaseFnCallModel, simulate_response_completion_with_chat
from qwen_agent.llm.schema import ASSISTANT, Message, FunctionCall
from qwen_agent.log import logger


@register_llm('oai')
class TextChatAtOAI(BaseFnCallModel):

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        self.model = self.model or 'gpt-4o-mini'
        cfg = cfg or {}

        api_base = cfg.get('api_base')
        api_base = api_base or cfg.get('base_url')
        api_base = api_base or cfg.get('model_server')
        api_base = (api_base or '').strip()

        # Better API key handling - don't use 'EMPTY' as default
        api_key = cfg.get('api_key')
        if not api_key:
            api_key = os.getenv('OPENAI_API_KEY')
        
        # Check if we have a valid API key
        if not api_key or api_key.strip() == '' or api_key.strip().upper() == 'EMPTY':
            raise ValueError(
                "OpenAI API key is required but not found. Please:\n"
                "1. Set the OPENAI_API_KEY environment variable, OR\n"
                "2. Pass 'api_key' in the configuration.\n"
                "Example: export OPENAI_API_KEY='your-key-here'"
            )
        
        api_key = api_key.strip()

        if openai.__version__.startswith('0.'):
            if api_base:
                openai.api_base = api_base
            if api_key:
                openai.api_key = api_key
            self._complete_create = openai.Completion.create
            self._chat_complete_create = openai.ChatCompletion.create
        else:
            api_kwargs = {}
            if api_base:
                api_kwargs['base_url'] = api_base
            if api_key:
                api_kwargs['api_key'] = api_key

            def _chat_complete_create(*args, **kwargs):
                # OpenAI API v1 does not allow the following args, must pass by extra_body
                extra_params = ['top_k', 'repetition_penalty']
                if any((k in kwargs) for k in extra_params):
                    kwargs['extra_body'] = copy.deepcopy(kwargs.get('extra_body', {}))
                    for k in extra_params:
                        if k in kwargs:
                            kwargs['extra_body'][k] = kwargs.pop(k)
                if 'request_timeout' in kwargs:
                    kwargs['timeout'] = kwargs.pop('request_timeout')

                client = openai.OpenAI(**api_kwargs)
                return client.chat.completions.create(*args, **kwargs)

            def _complete_create(*args, **kwargs):
                # OpenAI API v1 does not allow the following args, must pass by extra_body
                extra_params = ['top_k', 'repetition_penalty']
                if any((k in kwargs) for k in extra_params):
                    kwargs['extra_body'] = copy.deepcopy(kwargs.get('extra_body', {}))
                    for k in extra_params:
                        if k in kwargs:
                            kwargs['extra_body'][k] = kwargs.pop(k)
                if 'request_timeout' in kwargs:
                    kwargs['timeout'] = kwargs.pop('request_timeout')

                client = openai.OpenAI(**api_kwargs)
                return client.completions.create(*args, **kwargs)

            self._complete_create = _complete_create
            self._chat_complete_create = _chat_complete_create

    def _chat_stream(
        self,
        messages: List[Message],
        delta_stream: bool,
        generate_cfg: dict,
    ) -> Iterator[List[Message]]:
        messages = self.convert_messages_to_dicts(messages)
        try:
            response = self._chat_complete_create(model=self.model, messages=messages, stream=True, **generate_cfg)
            if delta_stream:
                for chunk in response:
                    if chunk.choices:
                        choice = chunk.choices[0]
                        if hasattr(choice.delta, 'reasoning_content') and choice.delta.reasoning_content:
                            yield [
                                Message(
                                    role=ASSISTANT,
                                    content='',
                                    reasoning_content=choice.delta.reasoning_content
                                )
                            ]
                        if hasattr(choice.delta, 'content') and choice.delta.content:
                            yield [Message(role=ASSISTANT, content=choice.delta.content, reasoning_content='')]
                        # 兼容 map agent 模型
                        if hasattr(choice.delta, 'tool_calls') and choice.delta.tool_calls:
                            function_name = choice.delta.tool_calls[0].function.name
                            function_call = {
                                'name': function_name,
                                'arguments': json.loads(choice.delta.tool_calls[0].function.arguments)
                            }
                            function_json = json.dumps(function_call, ensure_ascii=False)
                            yield [Message(role=ASSISTANT, content=f'<tool_call>{function_json}</tool_call>')]
                    logger.info(f'delta_stream message chunk: {chunk}')
            else:
                full_response = ''
                full_reasoning_content = ''
                for chunk in response:
                    if chunk.choices:
                        choice = chunk.choices[0]
                        if hasattr(choice.delta, 'reasoning_content') and choice.delta.reasoning_content:
                            full_reasoning_content += choice.delta.reasoning_content
                        if hasattr(choice.delta, 'content') and choice.delta.content:
                            full_response += choice.delta.content
                        # 兼容 map agent 模型
                        if hasattr(choice.delta, 'tool_calls') and choice.delta.tool_calls:
                            function_name = choice.delta.tool_calls[0].function.name
                            # function_call = FunctionCall(
                            #     name=function_name,
                            #     arguments=choice.delta.tool_calls[0].function.arguments,
                            # )
                            # yield [Message(role=ASSISTANT, content='', function_call=function_call)]
                            function_call = {
                                'name': function_name,
                                'arguments': json.loads(choice.delta.tool_calls[0].function.arguments)
                            }
                            function_json = json.dumps(function_call, ensure_ascii=False)
                            logger.info(json.dumps(function_call, ensure_ascii=False, indent=4))
                            full_response += f'<tool_call>{function_json}</tool_call>'
                        yield [Message(role=ASSISTANT, content=full_response, reasoning_content=full_reasoning_content)]
                    logger.info(f'message chunk: {chunk}')
        except OpenAIError as ex:
            raise ModelServiceError(exception=ex)

    def _chat_no_stream(
        self,
        messages: List[Message],
        generate_cfg: dict,
    ) -> List[Message]:
        messages = self.convert_messages_to_dicts(messages)
        try:
            response = self._chat_complete_create(model=self.model, messages=messages, stream=False, **generate_cfg)
            if hasattr(response.choices[0].message, 'reasoning_content'):
                return [
                    Message(role=ASSISTANT,
                            content=response.choices[0].message.content,
                            reasoning_content=response.choices[0].message.reasoning_content)
                ]
            else:
                return [Message(role=ASSISTANT, content=response.choices[0].message.content)]
        except OpenAIError as ex:
            raise ModelServiceError(exception=ex)

    def _chat_with_functions(
        self,
        messages: List[Message],
        functions: List[Dict],
        stream: bool,
        delta_stream: bool,
        generate_cfg: dict,
        lang: Literal['en', 'zh'],
    ) -> Union[List[Message], Iterator[List[Message]]]:
        # if delta_stream:
        #     raise NotImplementedError('Please use stream=True with delta_stream=False, because delta_stream=True'
        #                               ' is not implemented for function calling due to some technical reasons.')
        generate_cfg = copy.deepcopy(generate_cfg)
        for k in ['parallel_function_calls', 'function_choice', 'thought_in_content']:
            if k in generate_cfg:
                del generate_cfg[k]
        messages = simulate_response_completion_with_chat(messages)
        return self._chat(messages, stream=stream, delta_stream=delta_stream, generate_cfg=generate_cfg)

    def _chat(
        self,
        messages: List[Union[Message, Dict]],
        stream: bool,
        delta_stream: bool,
        generate_cfg: dict,
    ) -> Union[List[Message], Iterator[List[Message]]]:
        if stream:
            return self._chat_stream(messages, delta_stream=delta_stream, generate_cfg=generate_cfg)
        else:
            return self._chat_no_stream(messages, generate_cfg=generate_cfg)

    @staticmethod
    def convert_messages_to_dicts(messages: List[Message]) -> List[dict]:
        # TODO: Change when the VLLM deployed model needs to pass reasoning_complete.
        #  At this time, in order to be compatible with lower versions of vLLM,
        #  and reasoning content is currently not useful
        messages = [msg.model_dump() for msg in messages]

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'LLM Input:\n{pformat(messages, indent=2)}')
        return messages