import copy
from typing import Union

from .azure import TextChatAtAzure
from .base import LLM_REGISTRY, BaseChatModel, ModelServiceError
from .oai import TextChatAtOAI
from .openvino import OpenVINO
from .qwen_dashscope import QwenChatAtDS
from .qwenaudio_dashscope import QwenAudioChatAtDS
from .qwenomni_dashscope import QwenOmniChatAtDS
from .qwenvl_dashscope import QwenVLChatAtDS
from .qwenvl_oai import QwenVLChatAtOAI


def get_chat_model(cfg: Union[dict, str] = 'qwen-plus') -> BaseChatModel:
    """The interface of instantiating LLM objects.

    Args:
        cfg: The LLM configuration, one example is:
          cfg = {
              # Use the model service provided by DashScope:
              'model': 'qwen-max',
              'model_server': 'dashscope',

              # Use your own model service compatible with OpenAI API:
              # 'model': 'Qwen',
              # 'model_server': 'http://127.0.0.1:7905/v1',

              # (Optional) LLM hyper-parameters:
              'generate_cfg': {
                  'top_p': 0.8,
                  'max_input_tokens': 6500,
                  'max_retries': 10,
              }
          }

    Returns:
        LLM object.
    """
    if isinstance(cfg, str):
        cfg = {'model': cfg}

    if 'model_type' in cfg:
        model_type = cfg['model_type']
        if model_type in LLM_REGISTRY:
            if model_type in ('oai', 'qwenvl_oai'):
                if cfg.get('model_server', '').strip() == 'dashscope':
                    cfg = copy.deepcopy(cfg)
                    cfg['model_server'] = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
            return LLM_REGISTRY[model_type](cfg)
        else:
            raise ValueError(f'Please set model_type from {str(LLM_REGISTRY.keys())}')

    # Deduce model_type from model and model_server if model_type is not provided:

    if 'azure_endpoint' in cfg:
        model_type = 'azure'
        return LLM_REGISTRY[model_type](cfg)

    if 'model_server' in cfg:
        if cfg['model_server'].strip().startswith('http'):
            model_type = 'oai'
            return LLM_REGISTRY[model_type](cfg)

    model = cfg.get('model', '')

    if '-vl' in model.lower():
        model_type = 'qwenvl_dashscope'
        return LLM_REGISTRY[model_type](cfg)

    if '-audio' in model.lower():
        model_type = 'qwenaudio_dashscope'
        return LLM_REGISTRY[model_type](cfg)

    if 'qwen' in model.lower():
        model_type = 'qwen_dashscope'
        return LLM_REGISTRY[model_type](cfg)

    raise ValueError(f'Invalid model cfg: {cfg}')


__all__ = [
    'BaseChatModel',
    'QwenChatAtDS',
    'TextChatAtOAI',
    'TextChatAtAzure',
    'QwenVLChatAtDS',
    'QwenVLChatAtOAI',
    'QwenAudioChatAtDS',
    'QwenOmniChatAtDS',
    'OpenVINO',
    'get_chat_model',
    'ModelServiceError',
]
