import os
from typing import Dict, Optional

import openai

from qwen_agent.llm.base import register_llm
from qwen_agent.llm.oai import TextChatAtOAI


@register_llm('azure')
class TextChatAtAzure(TextChatAtOAI):

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        cfg = cfg or {}

        api_base = cfg.get('api_base')
        api_base = api_base or cfg.get('base_url')
        api_base = api_base or cfg.get('model_server')
        api_base = api_base or cfg.get('azure_endpoint')
        api_base = (api_base or '').strip()

        api_key = cfg.get('api_key')
        api_key = api_key or os.getenv('OPENAI_API_KEY')
        api_key = (api_key or 'EMPTY').strip()

        api_version = cfg.get('api_version', '2024-06-01')

        api_kwargs = {}
        if api_base:
            api_kwargs['azure_endpoint'] = api_base
        if api_key:
            api_kwargs['api_key'] = api_key
        if api_version:
            api_kwargs['api_version'] = api_version

        def _chat_complete_create(*args, **kwargs):
            client = openai.AzureOpenAI(**api_kwargs)
            return client.chat.completions.create(*args, **kwargs)

        self._chat_complete_create = _chat_complete_create
