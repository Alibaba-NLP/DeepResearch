"""
wiki-18 RAG retrieval system

主要函数：
- get_tool: 获取工具

作者: Evan Zuo
日期: 2025年7月
"""
# %%
import json
import re

import pandas as pd
import requests
from loguru import logger


def format_results(results, queries) -> str:
    """
    Format the retrieved documents into a readable string
    
    Args:
        results: Dictionary containing retrieval results
        queries: List of query strings
    
    Returns:
        str: Formatted string of all results
    """
    output = ""
    result_list = results["result"]
    
    for i, (query, documents) in enumerate(zip(queries, result_list)):
        output += f"\nQuery {i+1}: {query}\n"
        output += "-" * 50 + "\n"
        
        for j, doc in enumerate(documents):
            output += f"Document {j+1}:\n"
            if isinstance(doc, dict):
                doc_res = doc["document"]

                id = doc_res["id"]
                content = doc_res["contents"]
                title = doc_res.get("title", None)

                if title is None:
                    title = content.split("\n")[0]

                output += f"ID: {id}\n"
                output += f"Title: {title}\n"
                output += f"Text: {content}\n"  # Show full content

            output += "\n"
    
    return output


def get_tool(tool_name: str):
    """
    创建一个可配置的工具

    :param tool_name: 要运行的工具名称
    :returns: 已配置的工具
    """
    url = "http://10.200.64.10/10-flash-e2e-agent/retrieve"

    headers = {
        "Content-Type": "application/json;charset=utf-8",
    }

    def get_observation(tool_parameters: dict = None):
        """
        执行工具.
        
        :param tool_parameters: 工具传入的参数
        :return: 工具执行结果
        """
        queries = tool_parameters.get("query_list", None)
        if not queries:
            return f"工具执行过程中没有传入query_list"

        # 配置请求体
        payload = {
            "queries": queries,
            "topk": 3,
            "return_scores": True
        }
        logger.debug("工具调用:\n{}", payload)
        try:
            # 发送POST请求
            response = requests.post(url, json=payload, headers=headers, timeout=120)
        except Exception as e:
            return f"工具调用时出错：{e}"

        # 处理异常请求
        if response.status_code != 200:
            logger.error("请求失败!\n请求状态码: {}\n应答数据:\n{}", response.status_code, response.text)
            return f"工具执行过程中发生意外错误! 工具返回: {response.text}"
        
        # 返回状态码和解码后的内容
        response_data = response.json()

        return format_results(response_data, queries)

    return get_observation if tool_name == "search" else None


if __name__ == '__main__':
    """"""
    tool_name = "Get_Url_Content"
    tool_parameters = {"query_list": ["sushi"]}
    tool = get_tool(tool_name)
    content = tool(tool_parameters)
    print(content)

# %%
