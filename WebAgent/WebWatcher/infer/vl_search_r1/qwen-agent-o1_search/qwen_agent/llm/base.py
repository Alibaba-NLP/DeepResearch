import copy
import json
import os
import random
import time
from abc import ABC, abstractmethod
from pprint import pformat
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple, Union

from qwen_agent.llm.schema import ASSISTANT, DEFAULT_SYSTEM_MESSAGE, SYSTEM, USER, Message
from qwen_agent.log import logger
from qwen_agent.settings import DEFAULT_MAX_INPUT_TOKENS
from qwen_agent.utils.tokenization_qwen import tokenizer
from qwen_agent.utils.utils import (extract_text_from_message, format_as_multimodal_message, format_as_text_message,
                                    has_chinese_messages, json_dumps_compact, merge_generate_cfgs, print_traceback)

LLM_REGISTRY = {}


def register_llm(model_type):

    def decorator(cls):
        LLM_REGISTRY[model_type] = cls
        return cls

    return decorator


class ModelServiceError(Exception):

    def __init__(self,
                 exception: Optional[Exception] = None,
                 code: Optional[str] = None,
                 message: Optional[str] = None,
                 extra: Optional[dict] = None):
        if exception is not None:
            super().__init__(exception)
        else:
            super().__init__(f'\nError code: {code}. Error message: {message}')
        self.exception = exception
        self.code = code
        self.message = message
        self.extra = extra


class BaseChatModel(ABC):
    """The base class of LLM"""

    @property
    def support_multimodal_input(self) -> bool:
        # Does the model support multimodal input natively? It affects how we preprocess the input.
        return False

    @property
    def support_multimodal_output(self) -> bool:
        # Does the model generate multimodal outputs beyond texts? It affects how we post-process the output.
        return False

    def __init__(self, cfg: Optional[Dict] = None):
        cfg = cfg or {}
        self.model = cfg.get('model', '').strip()
        generate_cfg = copy.deepcopy(cfg.get('generate_cfg', {}))
        cache_dir = cfg.get('cache_dir', generate_cfg.pop('cache_dir', None))
        self.max_retries = generate_cfg.pop('max_retries', 0)
        self.generate_cfg = generate_cfg

        if cache_dir:
            try:
                import diskcache
            except ImportError:
                print_traceback(is_error=False)
                logger.warning('Caching disabled because diskcache is not installed. Please `pip install diskcache`.')
                cache_dir = None
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            self.cache = diskcache.Cache(directory=cache_dir)
        else:
            self.cache = None

    def quick_chat(self, prompt: str) -> str:
        *_, responses = self.chat(messages=[Message(role=USER, content=prompt)])
        assert len(responses) == 1
        assert not responses[0].function_call
        assert isinstance(responses[0].content, str)
        return responses[0].content

    def chat(
        self,
        messages: List[Union[Message, Dict]],
        functions: Optional[List[Dict]] = None,
        stream: bool = True,
        delta_stream: bool = False,
        extra_generate_cfg: Optional[Dict] = None,
    ) -> Union[List[Message], List[Dict], Iterator[List[Message]], Iterator[List[Dict]]]:
        """LLM chat interface.

        Args:
            messages: Inputted messages.
            functions: Inputted functions for function calling. OpenAI format supported.
            stream: Whether to use streaming generation.
            delta_stream: Whether to stream the response incrementally.
              (1) When False (recommended): Stream the full response every iteration.
              (2) When True: Stream the chunked response, i.e, delta responses.
            extra_generate_cfg: Extra LLM generation hyper-paramters.

        Returns:
            the generated message list response by llm.
        """

        # Unify the input messages to type List[Message]:
        messages = copy.deepcopy(messages)
        _return_message_type = 'dict'
        new_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                new_messages.append(Message(**msg))
            else:
                new_messages.append(msg)
                _return_message_type = 'message'
        messages = new_messages

        # RM think for non-qwq
        if 'qwq' not in self.model.lower():
            SPECIAL_THOUGHT_TOKEN = '<think>'
            new_messages = []
            for msg in messages:
                if msg.role == ASSISTANT:
                    if isinstance(msg.content, str):
                        if SPECIAL_THOUGHT_TOKEN in msg.content:
                            msg.content = _rm_think(msg.content)
                    elif isinstance(msg.content, list):
                        for i, item in enumerate(msg.content):
                            if item.text:
                                if SPECIAL_THOUGHT_TOKEN in item.text:
                                    item.text = _rm_think(item.text)
                                    msg.content[i] = item
                                break
                new_messages.append(msg)
            messages = new_messages

        # Cache lookup:
        if self.cache is not None:
            cache_key = dict(messages=messages, functions=functions, extra_generate_cfg=extra_generate_cfg)
            cache_key: str = json_dumps_compact(cache_key, sort_keys=True)
            cache_value: str = self.cache.get(cache_key)
            if cache_value:
                cache_value: List[dict] = json.loads(cache_value)
                if _return_message_type == 'message':
                    cache_value: List[Message] = [Message(**m) for m in cache_value]
                if stream:
                    cache_value: Iterator[List[Union[Message, dict]]] = iter([cache_value])
                return cache_value

        if stream and delta_stream:
            logger.warning(
                'Support for `delta_stream=True` is deprecated. '
                'Please use `stream=True and delta_stream=False` or `stream=False` instead. '
                'Using `delta_stream=True` makes it difficult to implement advanced postprocessing and retry mechanisms.'
            )

        generate_cfg = merge_generate_cfgs(base_generate_cfg=self.generate_cfg, new_generate_cfg=extra_generate_cfg)
        if 'seed' not in generate_cfg:
            generate_cfg['seed'] = random.randint(a=0, b=2**30)
        if 'lang' in generate_cfg:
            lang: Literal['en', 'zh'] = generate_cfg.pop('lang')
        else:
            lang: Literal['en', 'zh'] = 'zh' if has_chinese_messages(messages) else 'en'

        if messages[0].role != SYSTEM:
            messages = [Message(role=SYSTEM, content=DEFAULT_SYSTEM_MESSAGE)] + messages

        # Not precise. It's hard to estimate tokens related with function calling and multimodal items.
        max_input_tokens = generate_cfg.pop('max_input_tokens', DEFAULT_MAX_INPUT_TOKENS)
        if max_input_tokens > 0:
            messages = _truncate_input_messages_roughly(
                messages=messages,
                max_tokens=max_input_tokens,
            )

        if functions:
            fncall_mode = True
        else:
            fncall_mode = False
        if 'function_choice' in generate_cfg:
            fn_choice = generate_cfg['function_choice']
            valid_fn_choices = [f.get('name', f.get('name_for_model', None)) for f in (functions or [])]
            valid_fn_choices = ['auto', 'none'] + [f for f in valid_fn_choices if f]
            if fn_choice not in valid_fn_choices:
                raise ValueError(f'The value of function_choice must be one of the following: {valid_fn_choices}. '
                                 f'But function_choice="{fn_choice}" is received.')
            if fn_choice == 'none':
                fncall_mode = False

        # Note: the preprocessor's behavior could change if it receives function_choice="none"
        messages = self._preprocess_messages(messages, lang=lang, generate_cfg=generate_cfg, functions=functions)
        if not self.support_multimodal_input:
            messages = [format_as_text_message(msg, add_upload_info=False) for msg in messages]

        if not fncall_mode:
            for k in ['parallel_function_calls', 'function_choice']:
                if k in generate_cfg:
                    del generate_cfg[k]

        def _call_model_service():
            if fncall_mode:
                return self._chat_with_functions(
                    messages=messages,
                    functions=functions,
                    stream=stream,
                    delta_stream=delta_stream,
                    generate_cfg=generate_cfg,
                    lang=lang,
                )
            else:
                # TODO: Optimize code structure
                if messages[-1].role == ASSISTANT:
                    assert not delta_stream, 'Continuation mode does not currently support `delta_stream`'
                    return self._continue_assistant_response(messages, generate_cfg=generate_cfg, stream=stream)
                else:
                    return self._chat(
                        messages,
                        stream=stream,
                        delta_stream=delta_stream,
                        generate_cfg=generate_cfg,
                    )

        if stream and delta_stream:
            # No retry for delta streaming
            output = _call_model_service()
        elif stream and (not delta_stream):
            output = retry_model_service_iterator(_call_model_service, max_retries=self.max_retries)
        else:
            output = retry_model_service(_call_model_service, max_retries=self.max_retries)

        if isinstance(output, list):
            assert not stream
            logger.debug(f'LLM Output:\n{pformat([_.model_dump() for _ in output], indent=2)}')
            output = self._postprocess_messages(output, fncall_mode=fncall_mode, generate_cfg=generate_cfg)
            if not self.support_multimodal_output:
                output = _format_as_text_messages(messages=output)
            if self.cache:
                self.cache.set(cache_key, json_dumps_compact(output))
            return self._convert_messages_to_target_type(output, _return_message_type)
        else:
            assert stream
            if delta_stream:
                # Hack: To avoid potential errors during the postprocessing of stop words when delta_stream=True.
                # Man, we should never have implemented the support for `delta_stream=True` in the first place!
                generate_cfg = copy.deepcopy(generate_cfg)  # copy to avoid conflicts with `_call_model_service`
                assert 'skip_stopword_postproc' not in generate_cfg
                generate_cfg['skip_stopword_postproc'] = True
            output = self._postprocess_messages_iterator(output, fncall_mode=fncall_mode, generate_cfg=generate_cfg)

            def _format_and_cache() -> Iterator[List[Message]]:
                o = []
                for o in output:
                    if o:
                        if not self.support_multimodal_output:
                            o = _format_as_text_messages(messages=o)
                        yield o
                if o and (self.cache is not None):
                    self.cache.set(cache_key, json_dumps_compact(o))

            return self._convert_messages_iterator_to_target_type(_format_and_cache(), _return_message_type)

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

    @abstractmethod
    def _chat_with_functions(
        self,
        messages: List[Union[Message, Dict]],
        functions: List[Dict],
        stream: bool,
        delta_stream: bool,
        generate_cfg: dict,
        lang: Literal['en', 'zh'],
    ) -> Union[List[Message], Iterator[List[Message]]]:
        raise NotImplementedError

    def _continue_assistant_response(
        self,
        messages: List[Message],
        generate_cfg: dict,
        stream: bool,
    ) -> Iterator[List[Message]]:
        raise NotImplementedError

    @abstractmethod
    def _chat_stream(
        self,
        messages: List[Message],
        delta_stream: bool,
        generate_cfg: dict,
    ) -> Iterator[List[Message]]:
        raise NotImplementedError

    @abstractmethod
    def _chat_no_stream(
        self,
        messages: List[Message],
        generate_cfg: dict,
    ) -> List[Message]:
        raise NotImplementedError

    def _preprocess_messages(
        self,
        messages: List[Message],
        lang: Literal['en', 'zh'],
        generate_cfg: dict,
        functions: Optional[List[Dict]] = None,
    ) -> List[Message]:
        add_multimodel_upload_info = False
        if functions or (not self.support_multimodal_input):
            add_multimodel_upload_info = True
        messages = [
            format_as_multimodal_message(msg,
                                         add_upload_info=True,
                                         add_multimodel_upload_info=add_multimodel_upload_info,
                                         lang=lang) for msg in messages
        ]
        return messages

    def _postprocess_messages(
        self,
        messages: List[Message],
        fncall_mode: bool,
        generate_cfg: dict,
    ) -> List[Message]:
        messages = [
            format_as_multimodal_message(msg, add_upload_info=False, add_multimodel_upload_info=False)
            for msg in messages
        ]
        if not generate_cfg.get('skip_stopword_postproc', False):
            stop = generate_cfg.get('stop', [])
            messages = _postprocess_stop_words(messages, stop=stop)
        return messages

    def _postprocess_messages_iterator(
        self,
        messages: Iterator[List[Message]],
        fncall_mode: bool,
        generate_cfg: dict,
    ) -> Iterator[List[Message]]:
        pre_msg = []
        for pre_msg in messages:
            yield self._postprocess_messages(pre_msg, fncall_mode=fncall_mode, generate_cfg=generate_cfg)
        logger.debug(f'LLM Output:\n{pformat([_.model_dump() for _ in pre_msg], indent=2)}')

    def _convert_messages_to_target_type(self, messages: List[Message],
                                         target_type: str) -> Union[List[Message], List[Dict]]:
        if target_type == 'message':
            return [Message(**x) if isinstance(x, dict) else x for x in messages]
        elif target_type == 'dict':
            return [x.model_dump() if not isinstance(x, dict) else x for x in messages]
        else:
            raise NotImplementedError

    def _convert_messages_iterator_to_target_type(
            self, messages_iter: Iterator[List[Message]],
            target_type: str) -> Union[Iterator[List[Message]], Iterator[List[Dict]]]:
        for messages in messages_iter:
            yield self._convert_messages_to_target_type(messages, target_type)


def _format_as_text_messages(messages: List[Message]) -> List[Message]:
    for msg in messages:
        if isinstance(msg.content, list):
            for item in msg.content:
                assert item.type == 'text'
        else:
            assert isinstance(msg.content, str)
    messages = [format_as_text_message(msg, add_upload_info=False) for msg in messages]
    return messages


def _postprocess_stop_words(messages: List[Message], stop: List[str]) -> List[Message]:
    messages = copy.deepcopy(messages)

    # Make sure it stops before stop words.
    trunc_messages = []
    for msg in messages:
        truncated = False
        trunc_content = []
        for i, item in enumerate(msg.content):
            item_type, item_text = item.get_type_and_value()
            if item_type == 'text':
                truncated, item.text = _truncate_at_stop_word(text=item_text, stop=stop)
            trunc_content.append(item)
            if truncated:
                break
        msg.content = trunc_content
        trunc_messages.append(msg)
        if truncated:
            break
    messages = trunc_messages

    # It may ends with partial stopword 'Observation' when the full stopword is 'Observation:'.
    # The following post-processing step removes partial stop words.
    partial_stop = []
    for s in stop:
        s = tokenizer.tokenize(s)[:-1]
        if s:
            s = tokenizer.convert_tokens_to_string(s)
            partial_stop.append(s)
    partial_stop = sorted(set(partial_stop))
    last_msg = messages[-1].content
    for i in range(len(last_msg) - 1, -1, -1):
        item_type, item_text = last_msg[i].get_type_and_value()
        if item_type == 'text':
            for s in partial_stop:
                if item_text.endswith(s):
                    last_msg[i].text = item_text[:-len(s)]
            break

    return messages


def _truncate_at_stop_word(text: str, stop: List[str]):
    truncated = False
    for s in stop:
        k = text.find(s)
        if k >= 0:
            truncated = True
            text = text[:k]
    return truncated, text


def _truncate_input_messages_roughly(messages: List[Message], max_tokens: int) -> List[Message]:
    sys_msg = messages[0]
    assert sys_msg.role == SYSTEM  # The default system is prepended if none exists
    if len([m for m in messages if m.role == SYSTEM]) >= 2:
        raise ModelServiceError(
            code='400',
            message='The input messages must contain no more than one system message. '
            ' And the system message, if exists, must be the first message.',
        )

    turns = []
    for m in messages[1:]:
        if m.role == USER:
            turns.append([m])
        else:
            if turns:
                turns[-1].append(m)
            else:
                raise ModelServiceError(
                    code='400',
                    message='The input messages (excluding the system message) must start with a user message.',
                )

    def _count_tokens(msg: Message) -> int:
        return tokenizer.count_tokens(extract_text_from_message(msg, add_upload_info=True))

    def _truncate_message(msg: Message, max_tokens: int):
        if isinstance(msg.content, str):
            content = tokenizer.truncate(msg.content, max_token=max_tokens)
        else:
            text = []
            for item in msg.content:
                if not item.text:
                    return None
                text.append(item.text)
            text = '\n'.join(text)
            content = tokenizer.truncate(text, max_token=max_tokens)
        return Message(role=msg.role, content=content)

    available_token = max_tokens - _count_tokens(sys_msg)
    token_cnt = 0
    new_messages = []
    for i in range(len(messages) - 1, 0, -1):
        cur_token_cnt = _count_tokens(messages[i])
        if cur_token_cnt <= available_token:
            new_messages = [messages[i]] + new_messages
            available_token -= cur_token_cnt
        else:
            if (messages[i].role == USER) and (i != len(messages) - 1):
                _msg = _truncate_message(messages[i], max_tokens=available_token)
                if _msg:
                    new_messages = [_msg] + new_messages
                break
            else:
                token_cnt = (max_tokens - available_token) + cur_token_cnt
                break
    new_messages = [messages[0]] + new_messages
    if len(new_messages) < 2:
        raise ModelServiceError(
            code='400',
            message=f'The input messages exceed the maximum context length ({max_tokens} tokens) after '
            f'keeping only the system message and the latest one user message (around {token_cnt} tokens). '
            'To configure the context limit, please specifiy "max_input_tokens" in the model generate_cfg. '
            f'Example: generate_cfg = {{..., "max_input_tokens": {(token_cnt // 100 + 1) * 100}}}',
        )
    return new_messages


def retry_model_service(
    fn,
    max_retries: int = 10,
) -> Any:
    """Retry a function"""

    num_retries, delay = 0, 1.0
    while True:
        try:
            return fn()

        except ModelServiceError as e:
            num_retries, delay = _raise_or_delay(e, num_retries, delay, max_retries)


def retry_model_service_iterator(
    it_fn,
    max_retries: int = 10,
) -> Iterator:
    """Retry an iterator"""

    num_retries, delay = 0, 1.0
    while True:
        try:
            for rsp in it_fn():
                yield rsp
            break

        except ModelServiceError as e:
            num_retries, delay = _raise_or_delay(e, num_retries, delay, max_retries)


def _raise_or_delay(
    e: ModelServiceError,
    num_retries: int,
    delay: float,
    max_retries: int = 10,
    max_delay: float = 300.0,
    exponential_base: float = 2.0,
) -> Tuple[int, float]:
    """Retry with exponential backoff"""

    if max_retries <= 0:  # no retry
        raise e

    # Bad request, e.g., incorrect config or input
    if e.code == '400':
        raise e

    # If harmful input or output detected, let it fail
    if e.code == 'DataInspectionFailed':
        raise e
    if 'inappropriate content' in str(e):
        raise e

    # Retry is meaningless if the input is too long
    if 'maximum context length' in str(e):
        raise e

    logger.warning('ModelServiceError - ' + str(e).strip('\n'))

    if num_retries >= max_retries:
        raise ModelServiceError(exception=Exception(f'Maximum number of retries ({max_retries}) exceeded.'))

    num_retries += 1
    jitter = 1.0 + random.random()
    delay = min(delay * exponential_base, max_delay) * jitter
    time.sleep(delay)
    return num_retries, delay


def _rm_think(text: str) -> str:
    if '</think>' in text:
        return text.split('</think>')[-1].lstrip()
    return text
