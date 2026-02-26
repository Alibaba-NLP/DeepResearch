from typing import List
from ...client import BaseRequest
from ...util import convert_struct_list,convert_basic_list,convert_struct,convert_basic
from datetime import datetime


class AlibabaAidataSiteDataGetRequest(BaseRequest):

    def __init__(
        self,
        query: str = None,
        top_n: str = None,
        order: str = None,
        order_by: str = None,
        recall_type: str = None,
        extend_param: dict = None
    ):
        """
            用户查询请求
        """
        self._query = query
        """
            前n个匹配的结果，如 "10"
        """
        self._top_n = top_n
        """
            按升序asc,降序desc排列，,如"asc"
        """
        self._order = order
        """
            用户排序的指标，可以传多个，用英文","分隔 likes	点赞数；collects	收藏数；share_count 分享数；comments	评论数
        """
        self._order_by = order_by
        """
            召回方式，vector 向量匹配召回， multiple 多路召回（包括文本+向量）
        """
        self._recall_type = recall_type
        """
              "extendParam":{          //Map型拓展参数(均为非必传)      "appKey":"xxxx",						//开通AiData项目空间后的appKey      "channelType": "xhs"     // 指定召回数据来源，参数为空时则表示 xhs, 还可以填写 zhihu      "isRerank":true,        //是否开启重排序      "isRewrite":false,      //是否开启query重写，true开启重写会根据搜索引擎检索信息进行重写，false不开启重新      "publishStartTime":"1714561883000",  //笔记发布时间起始时间（格式为时间戳毫秒）      "publishEndTime":"1714661883000",  //笔记发布时间结束时间（格式为时间戳毫秒）      "maximalFans":"100",           //笔记作者最高粉丝数限制      "minimalLikes":"100"           //笔记最低点赞数限制        }
        """
        self._extend_param = extend_param

    @property
    def query(self):
        return self._query

    @query.setter
    def query(self, query):
        if isinstance(query, str):
            self._query = query
        else:
            raise TypeError("query must be str")

    @property
    def top_n(self):
        return self._top_n

    @top_n.setter
    def top_n(self, top_n):
        if isinstance(top_n, str):
            self._top_n = top_n
        else:
            raise TypeError("top_n must be str")

    @property
    def order(self):
        return self._order

    @order.setter
    def order(self, order):
        if isinstance(order, str):
            self._order = order
        else:
            raise TypeError("order must be str")

    @property
    def order_by(self):
        return self._order_by

    @order_by.setter
    def order_by(self, order_by):
        if isinstance(order_by, str):
            self._order_by = order_by
        else:
            raise TypeError("order_by must be str")

    @property
    def recall_type(self):
        return self._recall_type

    @recall_type.setter
    def recall_type(self, recall_type):
        if isinstance(recall_type, str):
            self._recall_type = recall_type
        else:
            raise TypeError("recall_type must be str")

    @property
    def extend_param(self):
        return self._extend_param

    @extend_param.setter
    def extend_param(self, extend_param):
        if isinstance(extend_param, dict):
            self._extend_param = extend_param
        else:
            raise TypeError("extend_param must be dict")


    def get_api_name(self):
        return "alibaba.aidata.site.data.get"

    def to_dict(self):
        request_dict = {}
        if self._query is not None:
            request_dict["query"] = convert_basic(self._query)

        if self._top_n is not None:
            request_dict["top_n"] = convert_basic(self._top_n)

        if self._order is not None:
            request_dict["order"] = convert_basic(self._order)

        if self._order_by is not None:
            request_dict["order_by"] = convert_basic(self._order_by)

        if self._recall_type is not None:
            request_dict["recall_type"] = convert_basic(self._recall_type)

        if self._extend_param is not None:
            request_dict["extend_param"] = convert_struct(self._extend_param)

        return request_dict

    def get_file_param_dict(self):
        file_param_dict = {}
        return file_param_dict

