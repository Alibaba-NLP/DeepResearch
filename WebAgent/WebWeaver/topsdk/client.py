from datetime import datetime
from topsdk.util import get_sign
from urllib.parse import urlencode
from abc import ABC,abstractmethod
import requests
import json

P_APPKEY = "app_key"
P_METHOD = "method"
P_SESSION = "session"
P_VERSION = "v"
P_FORMAT = "format"
P_TIMESTAMP = "timestamp"
P_SIGN = "sign"
P_SIGN_METHOD = "sign_method"
P_PARTNER_ID = "partner_id"
P_SIMPLIFY = "simplify"

P_CODE = "code"
P_SUB_CODE = "sub_code"
P_MSG = "msg"
P_SUB_MSG = "sub_msg"
P_REQUEST_ID = "request_id"

class TopApiClient:

    def __init__(self, appkey: str, app_sercet: str, top_gateway_url: str, simplify: bool = True, timeout = 10, proxy = None,verify_ssl = True):
        self.appkey = appkey
        self.app_sercet = app_sercet
        self.top_gateway_url = top_gateway_url
        self.format = "json"
        self.version = "2.0"
        self.sign_method = "hmac-sha256"
        self.simplify = simplify
        self.timeout = timeout
        self.proxy = proxy
        self.verify_ssl = verify_ssl


    def execute(self, api_code: str, request_dict: dict,file_dict: dict):
        return self.execute_with_session(api_code, request_dict, file_dict, "")

    def execute_with_session(self, api_code: str, request_dict: dict, file_dict: dict, session: str):
        public_param = {
            P_METHOD: api_code,
            P_APPKEY: self.appkey,
            P_FORMAT: self.format,
            P_VERSION: self.version,
            P_TIMESTAMP: datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            P_SIGN_METHOD: self.sign_method,
            P_SIMPLIFY: str(self.simplify).lower(),
            P_PARTNER_ID: "new_python3_sdk"
        }
        if session:
            public_param[P_SESSION] = session
        sign = get_sign(public_param.copy(), request_dict, self.app_sercet, self.sign_method)
        public_param[P_SIGN] = sign
        url = self.top_gateway_url + "?" + urlencode(public_param)
        if file_dict:
            response = requests.post(url, data=request_dict, files=file_dict, timeout=self.timeout,proxies=self.proxy,verify=self.verify_ssl)
        else:
            response = requests.post(url, data=request_dict, timeout=self.timeout,proxies=self.proxy ,verify=self.verify_ssl)
        if response.status_code is not 200:
            raise Exception('invalid http status ' + str(response.status_code) + ',detail body:' + response.text)
        jsonobj = json.loads(response.text)
        if "error_response" in jsonobj:
            error = TopException()
            if P_CODE in jsonobj["error_response"] :
                error.top_code = jsonobj["error_response"][P_CODE]
            if P_MSG in jsonobj["error_response"] :
                error.msg = jsonobj["error_response"][P_MSG]
            if P_SUB_CODE in jsonobj["error_response"] :
                error.sub_code = jsonobj["error_response"][P_SUB_CODE]
            if P_SUB_MSG in jsonobj["error_response"] :
                error.sub_msg = jsonobj["error_response"][P_SUB_MSG]
            if P_REQUEST_ID in jsonobj["error_response"]:
                error.request_id = jsonobj["error_response"][P_REQUEST_ID]
            raise error
        return jsonobj


class TopException(Exception):
    # ===========================================================================
    # 业务异常类
    # ===========================================================================
    def __init__(self):
        self.top_code = None
        self.msg = None
        self.sub_code = None
        self.sub_msg = None
        self.request_id = None

    def mix_str(self,pstr):
        if (isinstance(pstr, str)):
            return pstr
        elif (isinstance(pstr, str)):
            return pstr.encode('utf-8')
        else:
            return str(pstr)

    def __str__(self, *args, **kwargs) -> str:
        sb = "top_code=" + self.mix_str(self.top_code) + \
             " msg=" + self.mix_str(self.msg) + \
             " sub_code=" + self.mix_str(self.sub_code) + \
             " sub_msg=" + self.mix_str(self.sub_msg) + \
             " request_id=" + self.mix_str(self.request_id)
        return sb

class BaseRequest(ABC):

    @abstractmethod
    def to_dict(self):
        """
        TOPAPI Request类需要实现此方法, 转换Request对象为dict
        """

    @abstractmethod
    def get_api_name(self):
        """
        TOPAPI Request类需要实现此方法, 获取api名称
        """

    @abstractmethod
    def get_file_param_dict(self):
        """
        TOPAPI Request类需要实现此方法, 获取文件类型dict
        """
