from typing import List
from ...client import BaseRequest
from ...util import convert_struct_list,convert_basic_list,convert_struct,convert_basic
from datetime import datetime


class TaobaoOpenuidGetBymixnickRequest(BaseRequest):

    def __init__(
        self,
        mix_nick: str = None
    ):
        """
            无线类应用获取到的混淆的nick
        """
        self._mix_nick = mix_nick

    @property
    def mix_nick(self):
        return self._mix_nick

    @mix_nick.setter
    def mix_nick(self, mix_nick):
        if isinstance(mix_nick, str):
            self._mix_nick = mix_nick
        else:
            raise TypeError("mix_nick must be str")


    def get_api_name(self):
        return "taobao.openuid.get.bymixnick"

    def to_dict(self):
        request_dict = {}
        if self._mix_nick is not None:
            request_dict["mix_nick"] = convert_basic(self._mix_nick)

        return request_dict

    def get_file_param_dict(self):
        file_param_dict = {}
        return file_param_dict

