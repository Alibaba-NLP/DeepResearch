import copy
import logging
import os
from pprint import pformat
from typing import Dict, Iterator, List, Optional

import openai

from qwen_agent.utils.utils import build_text_completion_prompt

if openai.__version__.startswith('0.'):
    from openai.error import OpenAIError  # noqa
else:
    from openai import OpenAIError

from qwen_agent.llm.base import ModelServiceError, register_llm
from qwen_agent.llm.function_calling import BaseFnCallModel
from qwen_agent.llm.schema import ASSISTANT, Message
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

        api_key = cfg.get('api_key')
        api_key = api_key or os.getenv('OPENAI_API_KEY')
        api_key = (api_key or 'EMPTY').strip()

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
                    if chunk.choices and hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                        yield [Message(ASSISTANT, chunk.choices[0].delta.content)]
            else:
                full_response = ''
                for chunk in response:
                    if chunk.choices and hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        yield [Message(ASSISTANT, full_response)]
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
            return [Message(ASSISTANT, response.choices[0].message.content)]
        except OpenAIError as ex:
            raise ModelServiceError(exception=ex)

    def _continue_assistant_response(
        self,
        messages: List[Message],
        generate_cfg: dict,
        stream: bool,
    ) -> Iterator[List[Message]]:
        if ('qwen' in self.model) and ('vl' not in self.model):
            # We can call the completion interface according to the chat template of Qwen
            # Only support for text llm
            try:
                return self._continue_assistant_response_by_completion(messages=messages,
                                                                       generate_cfg=generate_cfg,
                                                                       stream=stream)
            except OpenAIError:
                logger.warning(
                    'This OAI interface does not support the completion interface, we will use the chat completion interface.'
                )

        # For other models, the chat templates is uncertain, so use dialogue simulation to completion
        return super()._continue_assistant_response(messages=messages, generate_cfg=generate_cfg, stream=stream)

    def _continue_assistant_response_by_completion(
        self,
        messages: List[Message],
        generate_cfg: dict,
        stream: bool,
    ) -> Iterator[List[Message]]:
        prompt = build_text_completion_prompt(messages)
        logger.debug(f'LLM Input:\n{pformat(prompt, indent=2)}')
        response = self._complete_create(model=self.model, prompt=prompt, stream=True, **generate_cfg)
        it = self._full_stream_output(response)
        if stream:
            return it  # streaming the response
        else:
            *_, final_response = it  # return the final response without streaming
            return final_response

    @staticmethod
    def _full_stream_output(response) -> Iterator[List[Message]]:
        full_response = ''
        for chunk in response:
            if chunk.choices and hasattr(chunk.choices[0], 'text') and chunk.choices[0].text:
                full_response += chunk.choices[0].text
                yield [Message(ASSISTANT, full_response)]

    @staticmethod
    def convert_messages_to_dicts(messages: List[Message]) -> List[dict]:
        messages = [msg.model_dump() for msg in messages]
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'LLM Input:\n{pformat(messages, indent=2)}')
        return messages
