from typing import List
from ...client import BaseRequest
from ...util import convert_struct_list,convert_basic_list,convert_struct,convert_basic
from datetime import datetime


class TaobaoTopAuthTokenCreateRequest(BaseRequest):

    def __init__(
        self,
        code: str = None,
        uuid: str = None
    ):
        """
            授权code，grantType==authorization_code 时需要
        """
        self._code = code
        """
            非必填，与生成code的uuid配对，使用方式参考文档
        """
        self._uuid = uuid

    @property
    def code(self):
        return self._code

    @code.setter
    def code(self, code):
        if isinstance(code, str):
            self._code = code
        else:
            raise TypeError("code must be str")

    @property
    def uuid(self):
        return self._uuid

    @uuid.setter
    def uuid(self, uuid):
        if isinstance(uuid, str):
            self._uuid = uuid
        else:
            raise TypeError("uuid must be str")


    def get_api_name(self):
        return "taobao.top.auth.token.create"

    def to_dict(self):
        request_dict = {}
        if self._code is not None:
            request_dict["code"] = convert_basic(self._code)

        if self._uuid is not None:
            request_dict["uuid"] = convert_basic(self._uuid)

        return request_dict

    def get_file_param_dict(self):
        file_param_dict = {}
        return file_param_dict

