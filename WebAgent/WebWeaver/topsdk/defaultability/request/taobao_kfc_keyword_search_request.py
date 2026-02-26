from typing import List
from ...client import BaseRequest
from ...util import convert_struct_list,convert_basic_list,convert_struct,convert_basic
from datetime import datetime


class TaobaoKfcKeywordSearchRequest(BaseRequest):

    def __init__(
        self,
        nick: str = None,
        apply: str = None,
        content: str = None
    ):
        """
            发布信息的淘宝会员名，可以不传
        """
        self._nick = nick
        """
            应用点，分为一级应用点、二级应用点。其中一级应用点通常是指某一个系统或产品，比如淘宝的商品应用（taobao_auction）；二级应用点，是指一级应用点下的具体的分类，比如商品标题(title)、商品描述(content)。不同的二级应用可以设置不同关键词。

这里的apply参数是由一级应用点与二级应用点合起来的字符（一级应用点+"."+二级应用点），如taobao_auction.title。


通常apply参数是不需要传递的。如有特殊需求（比如特殊的过滤需求，需要自己维护一套自己词库），需传递此参数。
        """
        self._apply = apply
        """
            需要过滤的文本信息
        """
        self._content = content

    @property
    def nick(self):
        return self._nick

    @nick.setter
    def nick(self, nick):
        if isinstance(nick, str):
            self._nick = nick
        else:
            raise TypeError("nick must be str")

    @property
    def apply(self):
        return self._apply

    @apply.setter
    def apply(self, apply):
        if isinstance(apply, str):
            self._apply = apply
        else:
            raise TypeError("apply must be str")

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
        return "taobao.kfc.keyword.search"

    def to_dict(self):
        request_dict = {}
        if self._nick is not None:
            request_dict["nick"] = convert_basic(self._nick)

        if self._apply is not None:
            request_dict["apply"] = convert_basic(self._apply)

        if self._content is not None:
            request_dict["content"] = convert_basic(self._content)

        return request_dict

    def get_file_param_dict(self):
        file_param_dict = {}
        return file_param_dict

