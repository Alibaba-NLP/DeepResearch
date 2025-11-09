import json
import os
import random
import re
from typing import Dict, Optional, Union

import requests

from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.utils.utils import has_chinese_chars


class CIServiceError(Exception):

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


START_CODE = """
import signal
def _m6_code_interpreter_timeout_handler(signum, frame):
    raise TimeoutError("CODE_INTERPRETER_TIMEOUT")
signal.signal(signal.SIGALRM, _m6_code_interpreter_timeout_handler)
def input(*args, **kwargs):
    raise NotImplementedError('Python input() function is disabled.')
import os
import math
import re
import json
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

import numpy as np
import pandas as pd

from sympy import Eq, symbols, solve
"""


def code_interpreter_dash(input,
                          files=[],
                          user_token=None,
                          request_id=None,
                          clear=False,
                          api_key=None,
                          user_id=None,
                          x_dashscope_uid=None):
    url = os.getenv('CODE_INTERPRETER_URL',
                    'https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation')
    model = os.getenv('CODE_INTERPRETER_MODEL', 'plugin-invoke')
    tool_id = os.getenv('CODE_INTERPRETER_TOOL_ID', 'code_interpreter')

    if clear:
        input = "\nget_ipython().run_line_magic('reset', '-f')\n" + START_CODE + input

    data = {
        'model': model,
        'input': {
            'tool_id': tool_id,
            'user_id': user_id,
            'plugin_attributes': {
                'header': {
                    'request_id': request_id or ''.join(random.choices('0123456789', k=10))
                },
                'payload': {
                    'input': input,
                    'files': files
                }
            },
        }
    }

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
        'x-dashscope-uid': x_dashscope_uid,
        'x-dashscope-user-token': user_token,
    }

    response_ori = requests.post(url, headers=headers, data=json.dumps(data))
    response = response_ori.json()

    if response_ori.status_code != 200:
        raise CIServiceError(code=str(response_ori.status_code),
                             message='Code Server Error',
                             extra={'ci_service_info': response})
    else:
        if response['output']['header']['statusCode'] == 200:
            out = response['output']['output']
        else:
            out = response['output']['header']['statusMessage']
        return out


code_pattern = re.compile(r'```\n(.*)\n```', re.DOTALL)


@register_tool('code_interpreter_http')
class CodeInterpreterHttp(BaseTool):
    description = 'Python代码沙盒，可用于执行Python代码。'
    parameters = [{'name': 'code', 'type': 'string', 'description': '待执行的代码', 'required': True}]

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        self.api_key = self.cfg.get('api_key', os.getenv('DASHSCOPE_API_KEY'))
        self.user_id = self.cfg.get('user_id', '19998497382695491')
        self.x_dashscope_uid = self.cfg.get('x_dashscope_uid', 'tujianhong')
        self.user_token = self.cfg.get('user_token', '123456')
        self.request_id = self.cfg.get('request_id', '')

    def call(self, params: Union[str, dict], **kwargs) -> str:
        # print(f"params:\n{params}")
        params = json.loads(params) if isinstance(params, str) else params
        code = params['code']
        files = params.get('files', [])
        clear = params.get('clear', False)

        output = code_interpreter_dash(input=code,
                                       files=files,
                                       user_token=self.user_token,
                                       request_id=self.request_id,
                                       clear=clear,
                                       api_key=self.api_key,
                                       user_id=self.user_id,
                                       x_dashscope_uid=self.x_dashscope_uid)
        print(f'output:\n{output}')
        if output.startswith('error:\n'):
            output = output[len('error:\n'):]

        if output.startswith('```'):
            output = output[len('```'):]
        if output.endswith('```'):
            output = output[:-len('```')]
        output = output.strip()
        return output if output else 'Finished execution.'

    @property
    def args_format(self) -> str:
        fmt = self.cfg.get('args_format')
        if fmt is None:
            if has_chinese_chars([self.name_for_human, self.name, self.description, self.parameters]):
                fmt = '此工具的输入应为Markdown代码块。'
            else:
                fmt = 'Enclose the code within triple backticks (`) at the beginning and end of the code.'
        return fmt


if __name__ == '__main__':
    tool = CodeInterpreterHttp()
    code = "print('hello world')"
    print(tool.call({'code': code}))
    code = '1 / 0'
    print(tool.call({'code': code}))
