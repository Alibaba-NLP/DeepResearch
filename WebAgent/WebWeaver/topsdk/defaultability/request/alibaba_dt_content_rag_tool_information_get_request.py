from typing import List
from ...client import BaseRequest
from ...util import convert_struct_list,convert_basic_list,convert_struct,convert_basic
from datetime import datetime


class AlibabaDtContentRagToolInformationGetRequest(BaseRequest):

    def __init__(
        self,
        web_tool_request: object = None
    ):
        """
            请求体
        """
        self._web_tool_request = web_tool_request

    @property
    def web_tool_request(self):
        return self._web_tool_request

    @web_tool_request.setter
    def web_tool_request(self, web_tool_request):
        if isinstance(web_tool_request, object):
            self._web_tool_request = web_tool_request
        else:
            raise TypeError("web_tool_request must be object")


    def get_api_name(self):
        return "alibaba.dt.content.rag.tool.information.get"

    def to_dict(self):
        request_dict = {}
        if self._web_tool_request is not None:
            request_dict["web_tool_request"] = convert_struct(self._web_tool_request)

        return request_dict

    def get_file_param_dict(self):
        file_param_dict = {}
        return file_param_dict

class AlibabaDtContentRagToolInformationGetWebToolRequest:
    def __init__(
        self,
        extra: dict = None,
        name: str = None,
        app_key: str = None,
        params: dict = None
    ):
        """
            额外信息-可不传
        """
        self.extra = extra
        """
            工具名称
        """
        self.name = name
        """
            app key
        """
        self.app_key = app_key
        """
            参数
        """
        self.params = params

