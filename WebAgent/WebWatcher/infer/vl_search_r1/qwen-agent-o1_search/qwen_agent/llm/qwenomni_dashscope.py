from typing import Dict, Optional

from qwen_agent.llm.base import register_llm
from qwen_agent.llm.qwenvl_dashscope import QwenVLChatAtDS


@register_llm('qwenomni_dashscope')
class QwenOmniChatAtDS(QwenVLChatAtDS):
    # TODO: Currently, the interface is incomplete

    @property
    def support_multimodal_output(self) -> bool:
        return True

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        self.model = self.model or 'qwen-audio-turbo-latest'
