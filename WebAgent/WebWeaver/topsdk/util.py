import hmac
import json
from datetime import datetime, date
from hashlib import md5, sha256


def get_sign(param_dict: dict, request_dict: dict, app_secret: str,sign_method: str) -> str:
    if sign_method == "hmac-sha256":
        return get_sign_with_hmac_sha256(param_dict, request_dict, app_secret)
    elif sign_method == "md5":
        return get_sign_with_md5(param_dict, request_dict, app_secret)
    else:
        raise Exception("Unsupported sign method: "+sign_method)


def get_sign_with_md5(param_dict: dict, request_dict: dict, app_secret: str) -> str:
    """
    使用md5计算签名
    :param param_dict: 公共参数字典
    :param request_dict: api参数字典
    :param app_secret: 密钥
    :return:
    """
    param_dict.update(request_dict)
    keys = list(param_dict.keys())
    keys.sort()
    parameters = "%s%s%s" % (app_secret,
                             str().join('%s%s' % (k, param_dict[k]) for k in keys if not isinstance(param_dict[k],bytes)),
                             app_secret)
    sign = md5(parameters.encode("utf-8")).hexdigest().upper()
    return sign


def get_sign_with_hmac_sha256(param_dict: dict, request_dict: dict, app_secret: str):
    """
    使用hmac sha256计算签名
    :param param_dict: 公共参数字典
    :param request_dict: api参数字典
    :param app_secret: 密钥
    :return:
    """
    param_dict.update(request_dict)
    keys = list(param_dict.keys())
    keys.sort()
    parameters = str().join('%s%s' % (k, param_dict[k]) for k in keys if not isinstance(param_dict[k], bytes))
    sign = hmac.new(app_secret.encode("utf-8"), parameters.encode("utf-8"), digestmod=sha256).hexdigest().upper()
    return str(sign)

def json_default(value):
    if isinstance(value, (datetime, date)):
        return value.strftime('%Y-%m-%d %H:%M:%S')
    else:
        return value.__dict__

class TopJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        else:
            return json.JSONEncoder.default(self, obj)


def convert_basic(param):
    """
    转换基本类型
    :param param:
    :return:
    """
    if isinstance(param, (datetime,date)):
        return param.strftime("%Y-%m-%d %H:%M:%S")
    elif isinstance(param, bool):
        return str(param).lower()
    elif isinstance(param, bytes):
        return param
    else:
        return str(param)


def convert_basic_list(param):
    """
    转换字符串、数字类型列表，列表 -> 逗号分割字符串
    :param param:
    :return:
    """
    if isinstance(param, (list, tuple, set)):
        return ",".join(convert_basic(i) for i in param)
    else:
        return param


def convert_struct(param):
    """
    转换dict 对象类型参数，转json字符串
    :param param:
    :return:
    """
    if isinstance(param, str):
        return param
    else:
        return json.dumps(param, cls=TopJsonEncoder, default=json_default, ensure_ascii=False)

def convert_struct_list(param):
    """
    转换复杂类型列表
    :param param:
    :return:
    """
    if isinstance(param,(set,list,tuple)):
        return json.dumps(param, cls=TopJsonEncoder, default=json_default, ensure_ascii=False)
    else:
        return str(param)
