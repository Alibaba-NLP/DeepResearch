from typing import List
from ...client import BaseRequest
from  ...util import convert_struct_list,convert_basic_list,convert_struct,convert_basic
from datetime import datetime


class TaobaoAppipGetRequest(BaseRequest):

    def __init__(
        self
    ):


    def get_api_name(self):
        return "taobao.appip.get"

    def to_dict(self):
        request_dict = {}
        return request_dict

    def get_file_param_dict(self):
        file_param_dict = {}
        return file_param_dict

