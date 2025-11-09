import json
import os
import sys
import time
import traceback
from typing import Any
from dotenv import load_dotenv
import requests
import tiktoken  

from qwen_agent.tools.gpt4o.base import BaseAPIClient
from qwen_agent.tools.gpt4o.constant import SUPPORT_ARGS
from qwen_agent.tools.gpt4o.utils import truncate_long_strings, APIException, get_args, openai_ret_wrapper

load_dotenv()
API_KEY = os.getenv('DASHSCOPE_API', '')

class OpenAIAPIClient(BaseAPIClient):
    def __init__(self,
                 call_url="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
                 api_key=API_KEY,
                 timeout=30,
                 verbose_num=1):
        self.call_url = call_url
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": self.api_key
        }
        self.timeout = timeout
        self.verbose_num = verbose_num
        super(OpenAIAPIClient, self).__init__(
            time_out=timeout,
            verbose_num=verbose_num)

    @staticmethod
    def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
        """Return the number of tokens used by a list of messages."""
        if 'audio' in model:
            return -1
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            print(f"Warning: model {model} not found. Using cl100k_base encoding.")
            encoding = tiktoken.get_encoding("cl100k_base")
        tokens_per_message = 3
        tokens_per_name = 1
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(
                    encoding.encode(value, disallowed_special=(encoding.special_tokens_set - {'<|endoftext|>'})))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens

    def _track_call(self, func_name: str, payload: Any) -> Any:
        if self._call_track[func_name] < self.verbose_num:
            self._call_track[func_name] += 1
            if self._call_track[func_name] == 1:
                print(f'Log Info for {self.verbose_num} CALLS:')
            print(f"CALL[{self._call_track[func_name]}]: {func_name}")
            print(f'url:' + self.call_url)
            print(f'headers: {truncate_long_strings(self.headers, max_len=30)}')
            print(f'payload: \n{json.dumps(truncate_long_strings(payload), indent=2, ensure_ascii=False)}')

    def _track_response(self, func_name: str, ret_json: dict) -> Any:
        if self._resp_track[func_name] < self.verbose_num:
            self._resp_track[func_name] += 1
            if self._resp_track[func_name] == 1:
                print(f'Log Info for {self.verbose_num} RESPONSE:')
            print(
                f"RESPONSE[{self._resp_track[func_name]}]: {func_name}, response: {json.dumps(truncate_long_strings(ret_json), indent=2, ensure_ascii=False)}")

    def _check_tokens(self, model, messages):
        pass

    def call(self, **kwargs):
        tenant = kwargs.pop('tenant', None)
        kwargs.pop('pre', None)
        payload = {'tenant': tenant} if tenant else dict()
        for key, value in kwargs.items():
            if value is not None and key in SUPPORT_ARGS:
                payload[key] = value
        assert 'model' in payload
        self._track_call(func_name=payload['model'], payload=payload)
        # message_tokens = OpenAIAPIClient.num_tokens_from_messages(payload['messages'], payload['model'])
        for i in range(self.max_try):
            try:
                ret = requests.post(self.call_url, json=payload,
                                    headers=self.headers, timeout=self.timeout)
                ret_json = ret.json()
                if self.call_url.startswith('http://47.88.8.18') and payload['model'].startswith('claude'):
                    ret_json = openai_ret_wrapper(ret_json, 'mit', 'claude')
                self._track_response(func_name=payload['model'], ret_json=ret_json)
                if ret.status_code != 200:
                    raise APIException(f"http status_code: {ret.status_code}\n{ret.content}")

                if 'choices' not in ret_json:
                    raise APIException(f"Error: {ret_json}")
                for output in ret_json['choices']:
                    if 'finish_reason' not in output:
                        raise APIException(f"Error: {ret_json}")
                    if output['finish_reason'].lower() not in ['stop', 'function_call', 'eos', 'end_turn']:
                        raise APIException(f'openai finish with error...\n{ret_json}')
                # assert message_tokens == ret_json['usage']['prompt_tokens']
                return ret_json
            except APIException as e:
                print(''.join(traceback.format_exception(*sys.exc_info())))
                time.sleep(self.retry_sleep)
        raise APIException('Max Retry!!!')


def test_llm_call(prompt, call_llm='gpt-4o', evaluate=False):

    if evaluate and call_llm == 'gpt-4o':
        model = 'gpt-4o-2024-08-06'
    elif call_llm == 'gpt-4o' and not evaluate:
        model = 'chatgpt-4o-latest'
    elif call_llm == 'gemini-2.5-pro':
        model = 'gemini-2.5-pro-preview-03-25'
    elif call_llm == 'gpt-o1':
        model = 'o1-preview-2024-09-12'
    else:
        model = call_llm
    
    kwargs = dict(
        model=model,
        messages= prompt,
        stream=False,
    )
    timeout = 1800
   
    call_url = kwargs.pop('call_url', None)
    authorization = kwargs.pop('authorization', None)
    if call_url and authorization:
        api = OpenAIAPIClient(call_url=call_url,
                              api_key=authorization,
                              timeout=timeout)
    else:
        api = OpenAIAPIClient(timeout=timeout)
    response = api.call(**kwargs)
    return response



def test_audio_output_call(args):
    messages = [{'role': 'user',
                 'content': '你叫林小桃，昵称“桃桃”，23岁女生，身高 168cm，体型苗条，皮肤白皙，长发飘飘，眼睛明亮”。她性格外向，e人，喜欢用夸张的语气和表情和人交流，特别健谈。小桃说话声音甜美，尝尝带有笑意、语调柔和，节奏适中，善用语气变化和撒娇表达情感, 情感丰富且夸张。会用“哇哦，哎呀，对呀, 哦？是嘛？咦？那个，就是说，嗯……”等词汇表达。\n 假设在场景：傍晚，我和同事在办公室茶水间讨论最近的新兴技术趋势。\n\n在以上人设和场景下，你感觉怎么样呀？（不要超过120个字）'}]
    kwargs = dict(
        model="gpt-4o-audio-preview",
        modalities=["text", "audio"],
        audio={"voice": 'voice_name', "format": "wav"},
        messages=messages,
        n=1,
        temperature=1.0,
        mit_spider=True,
        # api_seq=1
    )
    kwargs.update(args.__dict__)
    api = OpenAIAPIClient()
    ret_json = api.call(**kwargs)
    print(truncate_long_strings(ret_json))


if __name__ == '__main__':
    args = get_args()
    test_modality = args.pop('modality', 'text')
    if test_modality == 'text':
        test_llm_call(args)
    elif test_modality == 'audio':
        test_audio_output_call(args)
