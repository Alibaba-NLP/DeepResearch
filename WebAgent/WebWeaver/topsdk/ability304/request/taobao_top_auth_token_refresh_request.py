from typing import List
from ...client import BaseRequest
from ...util import convert_struct_list,convert_basic_list,convert_struct,convert_basic
from datetime import datetime


class TaobaoTopAuthTokenRefreshRequest(BaseRequest):

    def __init__(
        self,
        refresh_token: str = None
    ):
        """
            grantType==refresh_token 时需要
        """
        self._refresh_token = refresh_token

    @property
    def refresh_token(self):
        return self._refresh_token

    @refresh_token.setter
    def refresh_token(self, refresh_token):
        if isinstance(refresh_token, str):
            self._refresh_token = refresh_token
        else:
            raise TypeError("refresh_token must be str")


    def get_api_name(self):
        return "taobao.top.auth.token.refresh"

    def to_dict(self):
        request_dict = {}
        if self._refresh_token is not None:
            request_dict["refresh_token"] = convert_basic(self._refresh_token)

        return request_dict

    def get_file_param_dict(self):
        file_param_dict = {}
        return file_param_dict

