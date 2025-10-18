import copy
import logging
import os
from pprint import pformat
from typing import List

from qwen_agent.llm import ModelServiceError
from qwen_agent.llm.base import register_llm
from qwen_agent.llm.oai import TextChatAtOAI
from qwen_agent.llm.schema import ContentItem, Message
from qwen_agent.log import logger
from qwen_agent.utils.utils import encode_image_as_base64


@register_llm('qwenvl_oai')
class QwenVLChatAtOAI(TextChatAtOAI):

    @property
    def support_multimodal_input(self) -> bool:
        return True

    @staticmethod
    def convert_messages_to_dicts(messages: List[Message]) -> List[dict]:
        new_messages = []

        for msg in messages:
            content = msg.content
            if isinstance(content, str):
                content = [ContentItem(text=content)]
            assert isinstance(content, list)

            new_content = []
            for item in content:
                t, v = item.get_type_and_value()
                if t == 'text':
                    new_content.append({'type': 'text', 'text': v})
                if t == 'image':
                    if v.startswith('file://'):
                        v = v[len('file://'):]
                    if not v.startswith(('http://', 'https://', 'data:')):
                        if os.path.exists(v):
                            v = encode_image_as_base64(v, max_short_side_length=1080)
                        else:
                            raise ModelServiceError(f'Local image "{v}" does not exist.')
                    new_content.append({'type': 'image_url', 'image_url': {'url': v}})

            new_msg = msg.model_dump()
            new_msg['content'] = new_content
            new_messages.append(new_msg)

        if logger.isEnabledFor(logging.DEBUG):
            lite_messages = copy.deepcopy(new_messages)
            for msg in lite_messages:
                for item in msg['content']:
                    if item.get('image_url', {}).get('url', '').startswith('data:'):
                        item['image_url']['url'] = item['image_url']['url'][:64] + '...'
            logger.debug(f'LLM Input:\n{pformat(lite_messages, indent=2)}')

        return new_messages
