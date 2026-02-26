from typing import List
from ...client import BaseRequest
from ...util import convert_struct_list,convert_basic_list,convert_struct,convert_basic
from datetime import datetime


class TaobaoTopSdkFeedbackUploadRequest(BaseRequest):

    def __init__(
        self,
        type: str = None,
        content: str = None
    ):
        """
            1、回传加密信息
        """
        self._type = type
        """
            具体内容，json形式
        """
        self._content = content

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, type):
        if isinstance(type, str):
            self._type = type
        else:
            raise TypeError("type must be str")

    @property
    def content(self):
        return self._content

    @content.setter
    def content(self, content):
        if isinstance(content, str):
            self._content = content
        else:
            raise TypeError("content must be str")


    def get_api_name(self):
        return "taobao.top.sdk.feedback.upload"

    def to_dict(self):
        request_dict = {}
        if self._type is not None:
            request_dict["type"] = convert_basic(self._type)

        if self._content is not None:
            request_dict["content"] = convert_basic(self._content)

        return request_dict

    def get_file_param_dict(self):
        file_param_dict = {}
        return file_param_dict

