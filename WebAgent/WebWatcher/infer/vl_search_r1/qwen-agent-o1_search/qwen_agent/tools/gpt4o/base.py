import collections
from abc import ABC
from typing import Dict, Any


class BaseAPIClient(ABC):
    _call_track: Dict[str, int] = collections.defaultdict(int)
    _resp_track: Dict[str, int] = collections.defaultdict(int)

    def __init__(self,
                 call_sleep=0,
                 retry_sleep=10,
                 max_try=3,
                 time_out=180,
                 verbose_num=1,
                 ):
        self.call_sleep = call_sleep
        self.retry_sleep = retry_sleep
        self.max_try = max_try
        self.timeout = time_out
        self.verbose_num = verbose_num

    def _track_call(self, func_name: str, payload: Any) -> Any:
        if self._call_track[func_name] < self.verbose_num:
            self._call_track[func_name] += 1
            print(f"First calling {func_name}, payload: {payload}")

    def _track_response(self, func_name: str, response: Any) -> Any:
        if self._resp_track[func_name] < self.verbose_num:
            self._resp_track[func_name] += 1
            print(f"First call resp: {func_name}, response: {response}")
