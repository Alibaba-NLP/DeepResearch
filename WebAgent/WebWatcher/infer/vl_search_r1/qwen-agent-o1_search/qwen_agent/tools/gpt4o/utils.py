import argparse
import json
import os
import sys
from typing import Any
import pandas as pd


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class CustomNamespace(argparse.Namespace):
    def pop(self, key, default=None):
        return self.__dict__.pop(key, default)


class ArgumentParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("-i", "--in-file", type=str, required=False)
        # 不输入out_file默认是in_file文件名去掉后缀_{model}_result.jsonl
        self.parser.add_argument("-o", "--out-file", type=str, default=None)
        self.parser.add_argument("--num-workers", type=int, default=1)
        self.parser.add_argument("--local-cache", action='store_true')
        # 优先用输入文件中自带的args
        self.parser.add_argument("-m", "--model", type=str, default='gpt-4o')
        self.parser.add_argument("--seed", type=int, default=None)
        self.parser.add_argument("--top-p", type=float, default=None)
        self.parser.add_argument("--top-k", type=int, default=None)
        self.parser.add_argument("--temperature", type=float, default=None)
        self.parser.add_argument("--max-tokens", type=int, default=None)
        self.parser.add_argument("--max-completion-tokens", type=int, default=None)
        self.parser.add_argument("--repetition-penalty", type=float, default=None)
        self.parser.add_argument("--presence-penalty", type=float, default=None)
        self.parser.add_argument("--system", type=str, default=None)
        self.parser.add_argument("--stop", nargs="+", default=None)
        self.parser.add_argument("--n", type=int, default=None)
        self.parser.add_argument("--pre", action='store_true', help="是否预发环境")
        self.parser.add_argument("--completion", action='store_true')
        self.parser.add_argument("--timeout", type=int, default=1800)
        self.parser.add_argument("--call-url", type=str, default=None)
        self.parser.add_argument("--authorization", type=str, default=None)

    def parse_args(self) -> argparse.Namespace:
        known_args, unknown_args = self.parser.parse_known_args(namespace=CustomNamespace())
        # 合并已知参数和未知参数
        known_args.__dict__.update(self._parse_unknown_args(unknown_args))

        return known_args

    def _parse_unknown_args(self, unknown_args: list) -> dict:
        result = {}
        i = 0
        while i < len(unknown_args):
            arg = unknown_args[i]

            # 跳过非参数项
            if not arg.startswith('-'):
                i += 1
                continue

            # 获取参数名
            key = arg.lstrip('-')
            if '=' in key:
                key, value = key.split('=', 1)
            else:
                if i + 1 < len(unknown_args) and not unknown_args[i + 1].startswith('-'):
                    value = unknown_args[i + 1]
                    i += 2
                else:
                    value = True
                    i += 1

            # 类型转换
            result[key] = self._convert_value(value)

        return result

    @staticmethod
    def _convert_value(value: str) -> Any:
        """将字符串值转换为适当的类型"""
        if isinstance(value, bool):
            return value

        # 布尔值转换
        if value.lower() in ('true', 'yes', '1'):
            return True
        if value.lower() in ('false', 'no', '0'):
            return False

        # 数值转换
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            # 如果转换失败，返回原始字符串
            return value


def get_args():
    parser = ArgumentParser()
    args = parser.parse_args()
    return args


def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def is_bool(s):
    if s.lower() in ('yes', 'true', 'no', 'false'):
        return True


def parse_args():
    args = {}
    i = 1
    while i < len(sys.argv):
        if sys.argv[i].startswith('--'):
            key = sys.argv[i][2:]  # 去掉 '--' 前缀
            if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith('--'):
                value = sys.argv[i + 1]
                if value.isnumeric():
                    value = int(value)
                elif is_float(value):
                    value = float(value)
                elif is_bool(value):
                    value = True if value.lower() in ['yes', 'true'] else False
                args[key] = value
                i += 2
            else:
                args[key] = True
                i += 1
        else:
            i += 1
    return args


def compare_dict_structure(dict1, dict2):
    if not isinstance(dict1, dict) or not isinstance(dict2, dict):
        return type(dict1) is type(dict2)
    if set(dict1.keys()) != set(dict2.keys()):
        return False
    for key in dict1:
        if not compare_dict_structure(dict1[key], dict2[key]):
            return False
    return True


def dict_to_sorted_str(d):
    if isinstance(d, dict):
        return '{' + ', '.join(f'{repr(k)}: {dict_to_sorted_str(v)}' for k, v in sorted(d.items())) + '}'
    elif isinstance(d, list):
        return '[' + ', '.join(dict_to_sorted_str(x) for x in d) + ']'
    else:
        return repr(d)


def truncate_long_strings(d, max_len=100, head=False):
    if isinstance(d, dict):
        return {k: truncate_long_strings(v, max_len, head) for k, v in d.items()}
    elif isinstance(d, list):
        return [truncate_long_strings(item) for item in d]
    elif isinstance(d, str) and len(d) > max_len:
        if head:
            return '...' + d[-max_len:]
        else:
            return d[:max_len] + '...'
    else:
        return d


class APIException(Exception):
    def __init__(self, message, error_code=None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)

    def __str__(self):
        if self.error_code:
            return f'Error {self.error_code}: {self.message}'
        return self.message


def load_file2list(filename: str, sheet_name=0, header=0, update_by='uuid'):
    if not os.path.exists(filename):
        return []
    name, ext = os.path.splitext(filename)
    if ext.endswith('jsonl'):
        records = list()
        with open(filename, 'r') as file:
            for idx, line in enumerate(file):
                if len(line.strip()) == 0:
                    continue
                records.append(json.loads(line))
    else:
        if ext.endswith('xls') or filename.endswith('xlsx'):
            df = pd.read_excel(filename, sheet_name=sheet_name, keep_default_na=False, header=header)
        elif ext.endswith('csv'):
            df = pd.read_csv(filename, keep_default_na=False, header=header)
        elif ext.endswith('json'):
            df = pd.DataFrame(json.load(open(filename, "r")))
        else:
            raise f'No implementation for {ext} filetype!'
        json_str = df.to_json(orient='records', force_ascii=False)
        records = json.loads(json_str)
    if update_by is not None and len(records) > 0 and update_by in records[0]:
        update_d = dict()
        for record in records:
            update_d[record[update_by]] = record
        records = update_d.values()
    return records


def openai_ret_wrapper(ret, channel, model):
    # TODO: To be multimodal compatible
    if channel.lower() == 'mit' and model.lower().startswith('claude'):
        try:
            new_ret = dict()
            new_ret['usage'] = ret.get('usage')
            new_ret['choices'] = list()
            new_ret['request_id'] = ret.get('uid')
            new_ret['choices'].append({'finish_reason': ret.get('stop_reason'),
                                       'index': 0,
                                       'message':
                                           {
                                               'role': 'assistant',
                                               'content': ret.get("content")[0]['text']
                                           }
                                       }
                                      )
            return new_ret
        except TypeError:
            return None
    else:
        raise NotImplementedError
