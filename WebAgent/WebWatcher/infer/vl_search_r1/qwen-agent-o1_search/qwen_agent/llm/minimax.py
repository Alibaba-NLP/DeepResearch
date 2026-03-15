import copy
import os
from typing import Dict, Optional

from qwen_agent.llm.base import register_llm
from qwen_agent.llm.oai import TextChatAtOAI

MINIMAX_DEFAULT_API_BASE = 'https://api.minimax.io/v1'


@register_llm('minimax')
class TextChatAtMiniMax(TextChatAtOAI):
    """LLM provider for MiniMax (OpenAI-compatible API).

    MiniMax provides large language models accessible via an OpenAI-compatible
    API. Supported models include MiniMax-M2.5 and MiniMax-M2.5-highspeed,
    both offering 204K context length.

    Configuration example::

        llm_cfg = {
            'model': 'MiniMax-M2.5',
            'model_type': 'minimax',
            'generate_cfg': {
                'temperature': 0.7,
                'max_tokens': 4096,
            }
        }

    Environment variables:
        MINIMAX_API_KEY: Your MiniMax API key (required).
    """

    def __init__(self, cfg: Optional[Dict] = None):
        cfg = copy.deepcopy(cfg) or {}

        # Set MiniMax defaults before calling parent __init__
        if not cfg.get('model'):
            cfg['model'] = 'MiniMax-M2.5'

        # Use MiniMax API base URL if no custom server is specified
        if not cfg.get('api_base') and not cfg.get('base_url') and not cfg.get('model_server'):
            cfg['model_server'] = MINIMAX_DEFAULT_API_BASE

        # Use MINIMAX_API_KEY env var, fall back to OPENAI_API_KEY
        if not cfg.get('api_key'):
            api_key = os.getenv('MINIMAX_API_KEY') or os.getenv('OPENAI_API_KEY')
            if api_key:
                cfg['api_key'] = api_key

        # MiniMax requires temperature > 0
        generate_cfg = cfg.get('generate_cfg', {})
        temp = generate_cfg.get('temperature', None)
        if temp is not None and temp <= 0:
            generate_cfg['temperature'] = 0.01
            cfg['generate_cfg'] = generate_cfg

        super().__init__(cfg)
