from topsdk.client import TopApiClient, BaseRequest

class Defaultability:

    def __init__(self, client: TopApiClient):
        self._client = client

    """
        AiData平台aignite应用运行
    """
    def alibaba_aidata_aignite_application_run(self, request: BaseRequest):
        return self._client.execute(request.get_api_name(), request.to_dict(), request.get_file_param_dict())
    """
        aidata提供给通义资讯工具类服务
    """
    def alibaba_dt_general_tool_get(self, request: BaseRequest):
        return self._client.execute(request.get_api_name(), request.to_dict(), request.get_file_param_dict())
    """
        aidata站点数据获取
    """
    def alibaba_aidata_site_data_get(self, request: BaseRequest):
        return self._client.execute(request.get_api_name(), request.to_dict(), request.get_file_param_dict())
    """
        关键词过滤匹配
    """
    def taobao_kfc_keyword_search(self, request: BaseRequest, session: str):
        return self._client.execute_with_session(request.get_api_name(), request.to_dict(), request.get_file_param_dict(), session)
    """
        aidata提供给通义资讯工具类服务
    """
    def alibaba_dt_content_rag_tool_information_get(self, request: BaseRequest):
        return self._client.execute(request.get_api_name(), request.to_dict(), request.get_file_param_dict())
    """
        aidata提供给通义web搜索二跳服务
    """
    def alibaba_dt_content_web_tool_search_jumps(self, request: BaseRequest):
        return self._client.execute(request.get_api_name(), request.to_dict(), request.get_file_param_dict())
