import os
from http import HTTPStatus
from pprint import pformat
from typing import Dict, Iterator, List, Optional

import dashscope

from qwen_agent.llm.base import ModelServiceError, register_llm
from qwen_agent.llm.function_calling import BaseFnCallModel
from qwen_agent.llm.schema import ASSISTANT, Message
from qwen_agent.log import logger


@register_llm('qwen_dashscope')
class QwenChatAtDS(BaseFnCallModel):

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        self.model = self.model or 'qwen-max'
        initialize_dashscope(cfg)

    def _chat_stream(
        self,
        messages: List[Message],
        delta_stream: bool,
        generate_cfg: dict,
    ) -> Iterator[List[Message]]:
        messages = [msg.model_dump() for msg in messages]
        if messages[-1]['role'] == ASSISTANT:
            messages[-1]['partial'] = True
        logger.debug(f'LLM Input:\n{pformat(messages, indent=2)}')
        response = dashscope.Generation.call(
            self.model,
            messages=messages,  # noqa
            result_format='message',
            stream=True,
            **generate_cfg)
        if delta_stream:
            return self._delta_stream_output(response)
        else:
            return self._full_stream_output(response)

    def _chat_no_stream(
        self,
        messages: List[Message],
        generate_cfg: dict,
    ) -> List[Message]:
        messages = [msg.model_dump() for msg in messages]
        if messages[-1]['role'] == ASSISTANT:
            messages[-1]['partial'] = True
        logger.debug(f'LLM Input:\n{pformat(messages, indent=2)}')
        response = dashscope.Generation.call(
            self.model,
            messages=messages,  # noqa
            result_format='message',
            stream=False,
            **generate_cfg)
        if response.status_code == HTTPStatus.OK:
            return [
                Message(role=ASSISTANT,
                        content=response.output.choices[0].message.content,
                        reasoning_content=response.output.choices[0].message.get('reasoning_content', ''),
                        extra={'model_service_info': response})
            ]
        else:
            raise ModelServiceError(code=response.code,
                                    message=response.message,
                                    extra={'model_service_info': response})

    def _continue_assistant_response(
        self,
        messages: List[Message],
        generate_cfg: dict,
        stream: bool,
    ) -> Iterator[List[Message]]:
        return self._chat(messages, stream=stream, delta_stream=False, generate_cfg=generate_cfg)

    @staticmethod
    def _delta_stream_output(response) -> Iterator[List[Message]]:
        for chunk in response:
            if chunk.status_code == HTTPStatus.OK:
                yield [
                    Message(role=ASSISTANT,
                            content=chunk.output.choices[0].message.content,
                            reasoning_content=chunk.output.choices[0].message.reasoning_content,
                            extra={'model_service_info': chunk})
                ]
            else:
                raise ModelServiceError(code=chunk.code, message=chunk.message, extra={'model_service_info': chunk})

    @staticmethod
    def _full_stream_output(response) -> Iterator[List[Message]]:
        full_content = ''
        full_reasoning_content = ''
        for chunk in response:
            if chunk.status_code == HTTPStatus.OK:
                if chunk.output.choices[0].message.get('reasoning_content', ''):
                    full_reasoning_content += chunk.output.choices[0].message.reasoning_content
                if chunk.output.choices[0].message.content:
                    full_content += chunk.output.choices[0].message.content
                yield [
                    Message(role=ASSISTANT,
                            content=full_content,
                            reasoning_content=full_reasoning_content,
                            extra={'model_service_info': chunk})
                ]
            else:
                raise ModelServiceError(code=chunk.code, message=chunk.message, extra={'model_service_info': chunk})


def initialize_dashscope(cfg: Optional[Dict] = None) -> None:
    cfg = cfg or {}

    api_key = cfg.get('api_key', '')
    base_http_api_url = cfg.get('base_http_api_url', None)
    base_websocket_api_url = cfg.get('base_websocket_api_url', None)

    if not api_key:
        api_key = os.getenv('DASHSCOPE_API_KEY', 'EMPTY')
    if not base_http_api_url:
        base_http_api_url = os.getenv('DASHSCOPE_HTTP_URL', None)
    if not base_websocket_api_url:
        base_websocket_api_url = os.getenv('DASHSCOPE_WEBSOCKET_URL', None)

    api_key = api_key.strip()
    if api_key in ('', 'EMPTY'):
        if dashscope.api_key is None or dashscope.api_key in ('', 'EMPTY'):
            logger.warning('No valid dashscope api_key found in cfg, environment variable `DASHSCOPE_API_KEY` or dashscope.api_key, the model call may raise errors.')
        else:
            logger.info('No dashscope api_key found in cfg, using the dashscope.api_key that has already been set.')
    else: # valid api_key
        if api_key != dashscope.api_key:
            logger.info('Setting the dashscope api_key.')
            dashscope.api_key = api_key
        # or do nothing since both keys are the same
  
    if base_http_api_url is not None:
        dashscope.base_http_api_url = base_http_api_url.strip()
    if base_websocket_api_url is not None:
        dashscope.base_websocket_api_url = base_websocket_api_url.strip()
