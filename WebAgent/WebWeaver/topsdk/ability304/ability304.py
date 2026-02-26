from ..client import TopApiClient, BaseRequest

class Ability304:

    def __init__(self, client: TopApiClient):
        self._client = client

    """
        获取授权账号对应的OpenUid
    """
    def taobao_openuid_get(self, request: BaseRequest, session: str):
        return self._client.execute_with_session(request.get_api_name(), request.to_dict(), request.get_file_param_dict(), session)
    """
        通过订单获取对应买家的openUID
    """
    def taobao_openuid_get_bytrade(self, request: BaseRequest, session: str):
        return self._client.execute_with_session(request.get_api_name(), request.to_dict(), request.get_file_param_dict(), session)
    """
        TOPDNS配置
    """
    def taobao_httpdns_get(self, request: BaseRequest):
        return self._client.execute(request.get_api_name(), request.to_dict(), request.get_file_param_dict())
    """
        获取TOP通道解密秘钥
    """
    def taobao_top_secret_get(self, request: BaseRequest):
        return self._client.execute(request.get_api_name(), request.to_dict(), request.get_file_param_dict())
    """
        通过mixnick转换openuid
    """
    def taobao_openuid_get_bymixnick(self, request: BaseRequest):
        return self._client.execute(request.get_api_name(), request.to_dict(), request.get_file_param_dict())
    """
        业务文件获取
    """
    def taobao_files_get(self, request: BaseRequest):
        return self._client.execute(request.get_api_name(), request.to_dict(), request.get_file_param_dict())
    """
        刷新Access Token
    """
    def taobao_top_auth_token_refresh(self, request: BaseRequest):
        return self._client.execute(request.get_api_name(), request.to_dict(), request.get_file_param_dict())
    """
        获取Access Token
    """
    def taobao_top_auth_token_create(self, request: BaseRequest):
        return self._client.execute(request.get_api_name(), request.to_dict(), request.get_file_param_dict())
    """
        sdk信息回调
    """
    def taobao_top_sdk_feedback_upload(self, request: BaseRequest):
        return self._client.execute(request.get_api_name(), request.to_dict(), request.get_file_param_dict())
    """
        获取ISV发起请求服务器IP
    """
    def taobao_appip_get(self, request: BaseRequest):
        return self._client.execute(request.get_api_name(), request.to_dict(), request.get_file_param_dict())
