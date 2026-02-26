from typing import List
from ...client import BaseRequest
from ...util import convert_struct_list,convert_basic_list,convert_struct,convert_basic
from datetime import datetime


class TaobaoOpenuidGetBytradeRequest(BaseRequest):

    def __init__(
        self,
        tid: int = None
    ):
        """
            订单ID
        """
        self._tid = tid

    @property
    def tid(self):
        return self._tid

    @tid.setter
    def tid(self, tid):
        if isinstance(tid, int):
            self._tid = tid
        else:
            raise TypeError("tid must be int")


    def get_api_name(self):
        return "taobao.openuid.get.bytrade"

    def to_dict(self):
        request_dict = {}
        if self._tid is not None:
            request_dict["tid"] = convert_basic(self._tid)

        return request_dict

    def get_file_param_dict(self):
        file_param_dict = {}
        return file_param_dict

