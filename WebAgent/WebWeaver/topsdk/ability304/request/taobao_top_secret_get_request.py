from typing import List
from ...client import BaseRequest
from ...util import convert_struct_list,convert_basic_list,convert_struct,convert_basic
from datetime import datetime


class TaobaoTopSecretGetRequest(BaseRequest):

    def __init__(
        self,
        secret_version: int = None,
        random_num: str = None,
        customer_user_id: int = None
    ):
        """
            秘钥版本号
        """
        self._secret_version = secret_version
        """
            伪随机数
        """
        self._random_num = random_num
        """
            自定义用户id
        """
        self._customer_user_id = customer_user_id

    @property
    def secret_version(self):
        return self._secret_version

    @secret_version.setter
    def secret_version(self, secret_version):
        if isinstance(secret_version, int):
            self._secret_version = secret_version
        else:
            raise TypeError("secret_version must be int")

    @property
    def random_num(self):
        return self._random_num

    @random_num.setter
    def random_num(self, random_num):
        if isinstance(random_num, str):
            self._random_num = random_num
        else:
            raise TypeError("random_num must be str")

    @property
    def customer_user_id(self):
        return self._customer_user_id

    @customer_user_id.setter
    def customer_user_id(self, customer_user_id):
        if isinstance(customer_user_id, int):
            self._customer_user_id = customer_user_id
        else:
            raise TypeError("customer_user_id must be int")


    def get_api_name(self):
        return "taobao.top.secret.get"

    def to_dict(self):
        request_dict = {}
        if self._secret_version is not None:
            request_dict["secret_version"] = convert_basic(self._secret_version)

        if self._random_num is not None:
            request_dict["random_num"] = convert_basic(self._random_num)

        if self._customer_user_id is not None:
            request_dict["customer_user_id"] = convert_basic(self._customer_user_id)

        return request_dict

    def get_file_param_dict(self):
        file_param_dict = {}
        return file_param_dict

