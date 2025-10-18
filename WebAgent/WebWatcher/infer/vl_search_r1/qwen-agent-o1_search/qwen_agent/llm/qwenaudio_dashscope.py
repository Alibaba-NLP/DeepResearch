from typing import Dict, Optional

from qwen_agent.llm.base import register_llm
from qwen_agent.llm.qwenvl_dashscope import QwenVLChatAtDS


@register_llm('qwenaudio_dashscope')
class QwenAudioChatAtDS(QwenVLChatAtDS):

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        self.model = self.model or 'qwen-audio-turbo-latest'
