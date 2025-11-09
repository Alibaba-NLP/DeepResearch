import os
import requests
import random
import copy
import datetime
import string


region = os.getenv('CONTENT_SAFETY_REGION', 'ap-southeast-1')
client_endpoint = os.getenv('CONTENT_SAFETY_CLIENT_ENDPOINT', 'ep-t4ni0919c32c2e968a6e')
server_endpoint = os.getenv('CONTENT_SAFETY_SERVER_ENDPOINT', 'epsrv-t4nuxurcnq3orqz22oh2')
code = os.getenv('CONTENT_SAFETY_CODE', 'bailian_qwen_international_aigc_mtee_sns_unify_check')
business_name = os.getenv('CONTENT_SAFETY_BUSINESS_NAME', 'aliyun.bailian.international_AIGC')
first_product_name = os.getenv('CONTENT_SAFETY_FIRST_PRODUCT_NAME', 'qwen_international')


url = f"http://{client_endpoint}.{server_endpoint}.{region}.privatelink.aliyuncs.com/com.alibaba.security.tenant.common.service.RequestService/1.0.0_content_aigc_vpc_sg/request"

headers = {
    "Host": "mtee3-hsf-http.ali",
    "Content-Type": "application/json",
    "Http-Rpc-Type": "JsonContent"
}

payload_template = {
    "argsTypes": ["java.lang.String", "java.util.Map"],
    "argsObjs":[
        code,
        {
            "r": {
                "businessName": business_name,
                "firstProductName": first_product_name,
                "gmtCreate": "2024-04-23 13:13:45",
                "srcId": "qwen_chat_websearch-20240423-bcc95aec42ea",
                "operateType": "search",
                "csiUser": {
                    "id": "qwen_chat_websearch"
                },
                "csiAigc": {
                    "generateStage": "query",
                    "sceneType": "txt2txt",
                    "userInputTexts": [
                        {
                            "content": ""
                        }
                    ]
                },
                "ext": {
                    "originalUrl": ""
                }
            }
        }
    ]
}


def generate_random_string(length=12):
    # 使用字母和数字生成随机字符串
    characters = string.ascii_lowercase + string.digits
    return ''.join(random.choice(characters) for _ in range(length))


def get_current_time(format="%Y-%m-%d"):
    return datetime.datetime.now().strftime(format)


# 所有取值可能如下：
#  "-1"：未命中
#  "0" : 正常
#  "1" : 违规 
#  "3" : 疑似 (建议收到该值前台内容先屏蔽，等待异步识别结论，也称为“CC”）
def csi(text: str, doc_url: str, scene: str = "search") -> str:
    payload = copy.deepcopy(payload_template)
    payload["argsObjs"][1]["r"]["csiAigc"]["userInputTexts"][0]["content"] = text
    payload["argsObjs"][1]["r"]["ext"]["originalUrl"] = doc_url
    payload["argsObjs"][1]["r"]["csiUser"]["id"] = f"qwen_chat_{scene}"
    payload["argsObjs"][1]["r"]["operateType"] = scene
    payload["argsObjs"][1]["r"]["srcId"] = f'qwen_chat_{scene}-{get_current_time("%Y-%m-%d")}-{generate_random_string()}'
    payload["argsObjs"][1]["r"]["gmtCreate"] = get_current_time("%Y-%m-%d %H:%M:%S")

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        code = response.json()["data"]["code"]
        if code == "SUCCESS":
            csi_result = response.json()["data"]["result"]
            return csi_result
    except:
        pass
    return "-1"