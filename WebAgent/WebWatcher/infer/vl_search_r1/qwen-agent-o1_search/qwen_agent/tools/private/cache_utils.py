import os
import json
import fcntl


class JSONLCache:
    def __init__(self, cache_file):
        self.cache_file = cache_file
        self.cache = {}
        self._read_cache()
        
    def _lock_file(self, file, lock_type=fcntl.LOCK_EX):
        """ 获取文件锁 """
        fcntl.flock(file, lock_type)

    def _unlock_file(self, file):
        """ 释放文件锁 """
        fcntl.flock(file, fcntl.LOCK_UN)

    def _read_cache(self):
        """ 读取缓存文件 """
        if not os.path.exists(self.cache_file):
            return
        with open(self.cache_file, 'r') as f:
            self._lock_file(f, fcntl.LOCK_SH)  # 共享锁
            try:
                for line in f:
                    data = json.loads(line)
                    self.cache[data['key']] = data['value']
            finally:
                self._unlock_file(f)

    def _save_cache(self):
        """ 保存缓存到文件 """
        with open(self.cache_file, 'w') as f:
            self._lock_file(f, fcntl.LOCK_EX)  # 排它锁
            try:
                for key, value in self.cache.items():
                    data = {'key': key, 'value': value}
                    f.write(json.dumps(data, ensure_ascii=False) + '\n')
            finally:
                self._unlock_file(f)

    def update_cache(self):
        self._read_cache()
        print(f'cache file updated: {self.cache_file}')
        self._save_cache()
        print(f'cache size: {len(self.cache)}')
    
    def get(self, key, default=None):
        """ 获取缓存值 """
        return self.cache.get(key, default)
    
    def set(self, key, value):
        """ 设置缓存值 """
        self.cache[key] = value

