import copy
import os
import re
from http import HTTPStatus
from pprint import pformat
from typing import Dict, Iterator, List, Optional

import dashscope

from qwen_agent.llm.base import ModelServiceError, register_llm
from qwen_agent.llm.function_calling import BaseFnCallModel
from qwen_agent.llm.qwen_dashscope import initialize_dashscope
from qwen_agent.llm.schema import ASSISTANT, ContentItem, Message
from qwen_agent.log import logger


@register_llm('qwenvl_dashscope')
class QwenVLChatAtDS(BaseFnCallModel):

    @property
    def support_multimodal_input(self) -> bool:
        return True

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        self.model = self.model or 'qwen-vl-max'
        initialize_dashscope(cfg)

    def _chat_stream(
        self,
        messages: List[Message],
        delta_stream: bool,
        generate_cfg: dict,
    ) -> Iterator[List[Message]]:
        if delta_stream:
            raise NotImplementedError

        messages = _format_local_files(messages)
        messages = [msg.model_dump() for msg in messages]
        if 'partial' in generate_cfg:
            messages[-1]['partial'] = True
            del generate_cfg['partial']
        logger.debug(f'LLM Input:\n{pformat(messages, indent=2)}')
        response = dashscope.MultiModalConversation.call(model=self.model,
                                                         messages=messages,
                                                         result_format='message',
                                                         stream=True,
                                                         **generate_cfg)

        for chunk in response:
            if chunk.status_code == HTTPStatus.OK:
                yield _extract_vl_response(chunk)
            else:
                raise ModelServiceError(code=chunk.code, message=chunk.message, extra={'model_service_info': chunk})

    def _chat_no_stream(
        self,
        messages: List[Message],
        generate_cfg: dict,
    ) -> List[Message]:
        messages = _format_local_files(messages)
        messages = [msg.model_dump() for msg in messages]
        if 'partial' in generate_cfg:
            messages[-1]['partial'] = True
            del generate_cfg['partial']
        logger.debug(f'LLM Input:\n{pformat(messages, indent=2)}')
        response = dashscope.MultiModalConversation.call(model=self.model,
                                                         messages=messages,
                                                         result_format='message',
                                                         stream=False,
                                                         **generate_cfg)
        if response.status_code == HTTPStatus.OK:
            return _extract_vl_response(response=response)
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
        if messages[-1].role == ASSISTANT:
            generate_cfg['partial'] = True
        return self._chat(messages, stream=stream, delta_stream=False, generate_cfg=generate_cfg)


# DashScope Qwen-VL requires the following format for local files:
#   Linux & Mac: file:///home/images/test.png
#   Windows: file://D:/images/abc.png
def _format_local_files(messages: List[Message]) -> List[Message]:
    messages = copy.deepcopy(messages)
    for msg in messages:
        if isinstance(msg.content, list):
            for item in msg.content:
                if item.image:
                    item.image = _conv_fname(item.image)
                if item.audio:
                    item.audio = _conv_fname(item.audio)
                if item.video:
                    if isinstance(item.video, str):
                        item.video = _conv_fname(item.video)
                    else:
                        assert isinstance(item.video, list)
                        new_url = []
                        for fname in item.video:
                            new_url.append(_conv_fname(fname))
                        item.video = new_url
    return messages


def _conv_fname(fname: str) -> str:
    ori_fname = fname
    if not fname.startswith((
            'http://',
            'https://',
            'file://',
            'data:',  # base64 such as f"data:image/jpg;base64,{image_base64}"
    )):
        if fname.startswith('~'):
            fname = os.path.expanduser(fname)
        fname = os.path.abspath(fname)
        if os.path.isfile(fname):
            if re.match(r'^[A-Za-z]:\\', fname):
                fname = fname.replace('\\', '/')
            fname = 'file://' + fname
            return fname

    return ori_fname


def _extract_vl_response(response) -> List[Message]:
    output = response.output.choices[0].message
    text_content = []
    for item in output.content:
        if isinstance(item, str):
            text_content.append(ContentItem(text=item))
        else:
            for k, v in item.items():
                if k in ('text', 'box'):
                    text_content.append(ContentItem(text=v))
    return [Message(role=output.role, content=text_content, extra={'model_service_info': response})]
