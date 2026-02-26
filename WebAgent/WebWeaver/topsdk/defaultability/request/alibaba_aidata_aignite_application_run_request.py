from typing import List
from topsdk.client import BaseRequest
from topsdk.util import convert_struct_list,convert_basic_list,convert_struct,convert_basic
from datetime import datetime


class AlibabaAidataAigniteApplicationRunRequest(BaseRequest):

    def __init__(
        self,
        token: str = None,
        aignite_application_execute_req_dto: object = None
    ):
        """
            查看调用文档
        """
        self._token = token
        """
            入参
        """
        self._aignite_application_execute_req_dto = aignite_application_execute_req_dto

    @property
    def token(self):
        return self._token

    @token.setter
    def token(self, token):
        if isinstance(token, str):
            self._token = token
        else:
            raise TypeError("token must be str")

    @property
    def aignite_application_execute_req_dto(self):
        return self._aignite_application_execute_req_dto

    @aignite_application_execute_req_dto.setter
    def aignite_application_execute_req_dto(self, aignite_application_execute_req_dto):
        if isinstance(aignite_application_execute_req_dto, object):
            self._aignite_application_execute_req_dto = aignite_application_execute_req_dto
        else:
            raise TypeError("aignite_application_execute_req_dto must be object")


    def get_api_name(self):
        return "alibaba.aidata.aignite.application.run"

    def to_dict(self):
        request_dict = {}
        if self._token is not None:
            request_dict["token"] = convert_basic(self._token)

        if self._aignite_application_execute_req_dto is not None:
            request_dict["aignite_application_execute_req_dto"] = convert_struct(self._aignite_application_execute_req_dto)

        return request_dict

    def get_file_param_dict(self):
        file_param_dict = {}
        return file_param_dict

class AlibabaAidataAigniteApplicationRunAigniteApplicationExecuteReqDTO:
    def __init__(
        self,
        inputs: dict = None,
        application_id: int = None
    ):
        """
            入参
        """
        self.inputs = inputs
        """
            应用id
        """
        self.application_id = application_id

