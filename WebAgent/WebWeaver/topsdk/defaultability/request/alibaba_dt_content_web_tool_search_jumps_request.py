from typing import List
from ...client import BaseRequest
from ...util import convert_struct_list,convert_basic_list,convert_struct,convert_basic
from datetime import datetime


class AlibabaDtContentWebToolSearchJumpsRequest(BaseRequest):

    def __init__(
        self,
        two_page_jump_search_query: object = None,
        extend_params: dict = None
    ):
        """
            二跳参数
        """
        self._two_page_jump_search_query = two_page_jump_search_query
        """
            额外参数，用于校验
        """
        self._extend_params = extend_params

    @property
    def two_page_jump_search_query(self):
        return self._two_page_jump_search_query

    @two_page_jump_search_query.setter
    def two_page_jump_search_query(self, two_page_jump_search_query):
        if isinstance(two_page_jump_search_query, object):
            self._two_page_jump_search_query = two_page_jump_search_query
        else:
            raise TypeError("two_page_jump_search_query must be object")

    @property
    def extend_params(self):
        return self._extend_params

    @extend_params.setter
    def extend_params(self, extend_params):
        if isinstance(extend_params, dict):
            self._extend_params = extend_params
        else:
            raise TypeError("extend_params must be dict")


    def get_api_name(self):
        return "alibaba.dt.content.web.tool.search.jumps"

    def to_dict(self):
        request_dict = {}
        if self._two_page_jump_search_query is not None:
            request_dict["two_page_jump_search_query"] = convert_struct(self._two_page_jump_search_query)

        if self._extend_params is not None:
            request_dict["extend_params"] = convert_struct(self._extend_params)

        return request_dict

    def get_file_param_dict(self):
        file_param_dict = {}
        return file_param_dict

class AlibabaDtContentWebToolSearchJumpsWebPageUrlInfo:
    def __init__(
        self,
        tag: str = None,
        url: str = None
    ):
        """
            区分是否定向采集，一般情况下可留空   specified  指定url解析，支持pdf
        """
        self.tag = tag
        """
            url
        """
        self.url = url

class AlibabaDtContentWebToolSearchJumpsTwoPageJumpSearchQuery:
    def __init__(
        self,
        urls: list = None,
        cache: bool = None
    ):
        """
            一跳url列表
        """
        self.urls = urls
        """
            是否走缓存，若为false会强制实时采集
        """
        self.cache = cache

