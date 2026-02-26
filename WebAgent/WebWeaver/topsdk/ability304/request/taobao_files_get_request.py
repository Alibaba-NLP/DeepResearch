from typing import List
from  ...client import BaseRequest
from  ...util import convert_struct_list,convert_basic_list,convert_struct,convert_basic
from datetime import datetime


class TaobaoFilesGetRequest(BaseRequest):

    def __init__(
        self,
        status: int = None,
        start_date: datetime = None,
        end_date: datetime = None
    ):
        """
            下载链接状态。1:未下载。2:已下载
        """
        self._status = status
        """
            搜索开始时间
        """
        self._start_date = start_date
        """
            搜索结束时间
        """
        self._end_date = end_date

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, status):
        if isinstance(status, int):
            self._status = status
        else:
            raise TypeError("status must be int")

    @property
    def start_date(self):
        return self._start_date

    @start_date.setter
    def start_date(self, start_date):
        if isinstance(start_date, datetime):
            self._start_date = start_date
        else:
            raise TypeError("start_date must be datetime")

    @property
    def end_date(self):
        return self._end_date

    @end_date.setter
    def end_date(self, end_date):
        if isinstance(end_date, datetime):
            self._end_date = end_date
        else:
            raise TypeError("end_date must be datetime")


    def get_api_name(self):
        return "taobao.files.get"

    def to_dict(self):
        request_dict = {}
        if self._status is not None:
            request_dict["status"] = convert_basic(self._status)

        if self._start_date is not None:
            request_dict["start_date"] = convert_basic(self._start_date)

        if self._end_date is not None:
            request_dict["end_date"] = convert_basic(self._end_date)

        return request_dict

    def get_file_param_dict(self):
        file_param_dict = {}
        return file_param_dict

