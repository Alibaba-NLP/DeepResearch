import os
from typing import Dict, Optional, Union

from qwen_agent.settings import DEFAULT_WORKSPACE
from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.utils.utils import read_text_from_file, save_text_to_file


class KeyNotExistsError(ValueError):
    pass


@register_tool('storage')
class Storage(BaseTool):
    """
    This is a special tool for data storage
    """
    description = '存储和读取数据的工具'
    parameters = [{
        'name': 'operate',
        'type': 'string',
        'description': '数据操作类型，可选项为["put", "get", "delete", "scan"]之一，分别为存数据、取数据、删除数据、遍历数据',
        'required': True
    }, {
        'name': 'key',
        'type': 'string',
        'description': '数据的路径，类似于文件路径，是一份数据的唯一标识，不能为空，默认根目录为`/`。存数据时，应该合理的设计路径，保证路径含义清晰且唯一。',
        'default': '/'
    }, {
        'name': 'value',
        'type': 'string',
        'description': '数据的内容，仅存数据时需要'
    }]

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        self.root = self.cfg.get('storage_root_path', os.path.join(DEFAULT_WORKSPACE, 'tools', self.name))
        os.makedirs(self.root, exist_ok=True)

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)
        operate = params['operate']
        key = params.get('key', '/')
        if key.startswith('/'):
            key = key[1:]

        if operate == 'put':
            assert 'value' in params
            return self.put(key, params['value'])
        elif operate == 'get':
            return self.get(key)
        elif operate == 'delete':
            return self.delete(key)
        else:
            return self.scan(key)

    def put(self, key: str, value: str, path: Optional[str] = None) -> str:
        path = path or self.root

        # one file for one key value pair
        path = os.path.join(path, key)

        path_dir = path[:path.rfind('/') + 1]
        if path_dir:
            os.makedirs(path_dir, exist_ok=True)

        save_text_to_file(path, value)
        return f'Successfully saved {key}.'

    def get(self, key: str, path: Optional[str] = None) -> str:
        path = path or self.root
        if not os.path.exists(os.path.join(path, key)):
            raise KeyNotExistsError(f'Get Failed: {key} does not exist')
        return read_text_from_file(os.path.join(path, key))

    def delete(self, key, path: Optional[str] = None) -> str:
        path = path or self.root
        path = os.path.join(path, key)
        if os.path.exists(path):
            os.remove(path)
            return f'Successfully deleted {key}'
        else:
            return f'Delete Failed: {key} does not exist'

    def scan(self, key: str, path: Optional[str] = None) -> str:
        path = path or self.root
        path = os.path.join(path, key)
        if os.path.exists(path):
            if not os.path.isdir(path):
                return 'Scan Failed: The scan operation requires passing in a folder path as the key.'
            # All key-value pairs
            kvs = {}
            for root, dirs, files in os.walk(path):
                for file in files:
                    k = os.path.join(root, file)[len(path):]
                    if not k.startswith('/'):
                        k = '/' + k
                    v = read_text_from_file(os.path.join(root, file))
                    kvs[k] = v
            return '\n'.join([f'{k}: {v}' for k, v in kvs.items()])
        else:
            return f'Scan Failed: {key} does not exist.'
