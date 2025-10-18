import os
import re
import json
import time
import random
import requests
import datetime
from typing import Union, List
from dataclasses import dataclass
from functools import wraps
import atexit
import uuid

from qwen_agent.tools.private.cache_utils import JSONLCache
from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.tools.private.sfilter import multi_call_sfilter
from qwen_agent.log import logger

MAX_CHAR = int(os.getenv("MAX_CHAR", default=28000))
SEARCH_ENGINE = os.getenv("SEARCH_ENGINE", "google")
SEARCH_STRATEGY = os.getenv("SEARCH_STRATEGY", "rerank")
QWEN_SEARCH_KEY = os.getenv("QWEN_SEARCH_KEY")
TEXT_SEARCH_KEY = os.getenv("TEXT_SEARCH_KEY")
USER_NAME = os.getenv("UserName", "xinyu")

# knowledge
KNOWLEDGE_SNIPPET = """## 来自 {source} 的内容：

```
{content}
```"""

KNOWLEDGE_PROMPT = """# 知识库

{knowledge_snippets}"""


@dataclass
class SearchItem:
    title: str = ""
    body: str = ""
    href: str = ""
    time: int = 0
    exclusive: bool = False
    relevance: float = 0  # important items have higher scores
    original_order: int = -1  # for stable sort
    source: str = ""
    host_logo: str = ""


def get_current_date_str() -> str:
    beijing_time = datetime.datetime.utcnow() + datetime.timedelta(hours=8)
    cur_time = beijing_time.timetuple()
    date_str = f"当前时间：{cur_time.tm_year}年{cur_time.tm_mon}月{cur_time.tm_mday}日，星期"
    date_str += ["一", "二", "三", "四", "五", "六", "日"][cur_time.tm_wday]
    date_str += f"{cur_time.tm_hour}时{cur_time.tm_min}分"
    date_str += "。"
    return date_str


def convert_to_timestamp(input_val, time_format="%Y-%m-%d %H:%M:%S"):
    if isinstance(input_val, (int, float)):
        return int(input_val)
    elif input_val.isdigit():  # 假设时间戳字符串全为数字
        return int(input_val)
    elif input_val == " ":
        return 0
    else:
        try:
            datetime_obj = datetime.strptime(input_val, time_format)
            return int(datetime_obj.timestamp())
        except Exception:
            # 如果时间戳格式有误，则返回0
            return 0


def _rm_html(text: str) -> str:
    _HTML_TAG_RE = re.compile(r" ?</?(a|span|em|br).*?> ?")
    text = text.replace("\xa0", " ")
    text = text.replace("\t", "")  # quark uses \t to split chinese words
    text = text.replace("...", "……")
    text = _HTML_TAG_RE.sub("", text)
    text = text.strip()
    if text.endswith("……"):
        text = text[: -len("……")]
    return text


def parse_web_search_result(results, **kwargs) -> List[SearchItem]:
    search_items: List[SearchItem] = []
    for doc in results:
        search_items.append(
            SearchItem(
                title=doc.get("title", "") or "",
                body=doc.get("snippet", "") or "",
                href=doc.get("url", "") or "",
                time=convert_to_timestamp(doc.get("timestamp", "0")),
                source=doc.get("hostname", "") or "",
                relevance=doc.get("_score", 0.0) or 0.0,
                host_logo=doc.get("hostlogo", "") or "",
            )
        )

    for i, item in enumerate(search_items):
        item.original_order = i
        item.href = item.href.replace(" ", "%20").strip() or "expired_url"
        item.href = item.href.replace("chatm6.sm.cn", "quark.sm.cn")

    return search_items


def web_search_knowledge(content) -> str:
    search_items = parse_web_search_result(content)
    # Using the same format repeatedly may harm the response's diversity.
    timestamp_templates = [
        "（搜索结果收录于{}年{}月{}日）",
        "（{}年{}月{}日）",
        "（来自{}年{}月{}日的资料）",
        "（{}年{}月{}日的资料）",
        "（该信息的时间戳是{}年{}月{}日）",
        "（资料日期为{}年{}月{}日）",
        "（消息于{}年{}月{}日发布）",
        "（发布时间是{}年{}月{}日）",
        "（撰于{}年{}月{}日）",
        "（截至{}年{}月{}日）",
    ]
    random.shuffle(timestamp_templates)

    max_char = MAX_CHAR
    cnt_char = 0
    text_result = []
    for i, item in enumerate(search_items):
        if item.time > 0:
            t = time.localtime(item.time // 1000) if len(str(item.time)) == 13 else time.localtime(item.time)
            if i < len(timestamp_templates):
                k = i
            else:
                k = random.randint(0, len(timestamp_templates) - 1)
            text_timestamp = timestamp_templates[k].format(t.tm_year, t.tm_mon, t.tm_mday)
        else:
            text_timestamp = ""

        snippet = f"title: {_rm_html(item.title)}\nurl:{item.href}\nsnippet:{_rm_html(item.body)}".strip()
        text_snippet = snippet.replace("\n", "\\n")
        text_result.append(f'"{text_snippet}"{text_timestamp}')

        cnt_char += len(snippet)
        if cnt_char > max_char:
            break

    text_result_str = "\n\n".join(text_result).strip()
    while len(text_result) > 1 and len(text_result_str) > max_char:
        text_result.pop(-1)
        text_result_str = "\n\n".join(text_result).strip()

    return text_result_str


def tool_call_knowledge(tool_output, **kwargs) -> str:
    prompt = """以下通过权威渠道的实时信息可能有助于你回答问题，请优先参考：#以下根据实际返回选择"""
    for item in tool_output:
        if item.get("tool", "") == "oil_price":
            prompt = prompt + "\n 油价信息:" + item.get("result", "")
        elif item.get("tool", "") == "gold_price":
            prompt = prompt + "\n 金价信息:" + item.get("result", "")
        elif item.get("tool", "") == "exchange":
            prompt = prompt + "\n 汇率信息:" + item.get("result", "")
        elif item.get("tool", "") == "stock":
            prompt = prompt + "\n 股市信息:" + item.get("result", "")
        elif item.get("tool", "") == "silver_price":
            prompt = prompt + "\n 银价信息:" + item.get("result", "")
        elif item.get("tool", "") == "weather":
            prompt = prompt + "\n 天气信息:" + item.get("result", "")
        elif item.get("tool", "") == "calender":
            prompt = prompt + "\n 万年历信息:" + item.get("result", "")

    return prompt


def get_online_prompt(search_docs, queries, topk=10, topk_readpage=10, toolResult=[]):
    enable_readpage = os.getenv('NLP_WEB_SEARCH_ENABLE_READPAGE', 'false').lower() in ('y', 'yes', 't', 'true', '1', 'on')
    enable_sfilter = os.getenv('NLP_WEB_SEARCH_ENABLE_SFILTER', 'false').lower() in ('y', 'yes', 't', 'true', '1', 'on')

    if enable_readpage:
        if enable_sfilter:
            i = 0
            search_docs = multi_call_sfilter(queries[0], search_docs)
        else:
            i = 0
            for search_doc in search_docs:
                if i < topk_readpage and len(search_doc.get("web_main_body", "")) > len(search_doc["snippet"]):
                    search_doc["snippet"] = search_doc["web_main_body"][:4000]
                    i += 1
                    logger.info('page read!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    search_docs = search_docs[:topk]
    return f"## 来自web_search的内容：\n\n```{web_search_knowledge(search_docs)}```"


enable_search_cache = os.getenv('NLP_WEB_SEARCH_ENABLE_CACHE', 'false').lower() in ('y', 'yes', 't', 'true', '1', 'on')
cache = JSONLCache(os.path.join(os.path.dirname(__file__), "search_cache.jsonl"))

if enable_search_cache:
    atexit.register(cache.update_cache)

def search_cache_decorator(func):
    @wraps(func)
    def wrapper(self, queries, *args, **kwargs):
        if enable_search_cache:
            key = str(sorted(queries))
            cached_result = cache.get(key)
            if cached_result is not None:
                return cached_result
            result = func(self, queries, *args, **kwargs)
            cache.set(key, result)
        else:
            result = func(self, queries, *args, **kwargs)
        return result
    return wrapper



@register_tool("web_search", allow_overwrite=True)
class WebSearch(BaseTool):
    name = "web_search"
    description = "Utilize the web search engine to retrieve relevant information based on multiple queries."
    parameters = {
        "type": "object",
        "properties": {
            "queries": {
                "type": "array",
                "items": {"type": "string", "description": "The search query."},
                "description": "The list of search queries.",
            }
        },
        "required": ["queries"],
    }

    @search_cache_decorator
    def google(self, queries: List[str], max_retry: int = 10):
        only_cache = os.getenv('NLP_WEB_SEARCH_ONLY_CACHE', 'true').lower() in ('y', 'yes', 't', 'true', '1', 'on')
        url = "http://101.37.167.147/gw/v1/api/msearch-sp/qwen-search"
        headers = {"Authorization": f"Bearer {TEXT_SEARCH_KEY}", "Content-Type": "application/json", "Host": "pre-nlp-cn-hangzhou.aliyuncs.com"}
        template = {
            "rid": str(uuid.uuid4()),
            "scene": "dolphin_search_google_nlp",
            "uq": queries[0],
            "debug": False,
            "fields": [],
            "page": 1,
            "rows": 10,
            "customConfigInfo": {
                "multiSearch": False,
                "qpMultiQueryConfig": queries,
                "qpMultiQuery": True,
                "qpMultiQueryHistory": [],
                "qpSpellcheck": False,
                "qpEmbedding": False,
                "knnWithScript": False,
                "rerankSize": 10,
                "qpTermsWeight": False,
                "qpToolPlan": False,
                "inspection": False, #关闭绿网
                "readpage": False,
                "readpageConfig": {"tokens": 4000, "topK": 10, "onlyCache": only_cache},
            },
            "rankModelInfo": {
                "default": {
                    "features": [
                        {"name": "static_value", "field": "_weather_score", "weights": 1.0},
                        {
                            "name": "qwen-rerank",
                            "fields": ["hostname", "title", "snippet", "timestamp_format"],
                            "weights": 1,
                            "threshold": -50,
                            "max_length": 512,
                            "rank_size": 100,
                            "norm": False,
                        },
                    ],
                    "aggregate_algo": "weight_avg",
                }
            },
            "headers": {
                "__d_head_qto": 20000,
                "__d_head_app": USER_NAME
                },
        }

        resp = ""
        # breakpoint()
        for _ in range(max_retry):
            try:
                resp = requests.post(url, headers=headers, data=json.dumps(template))
                rst = json.loads(resp.text)
                docs = rst["data"]["docs"]
                assert len(docs) != 0, template['rid'] + "搜索为空，重试"
                return docs
            except Exception as e:
                print("Meet error when search query:", resp, queries, e, template['rid'])
                print("retrying")
                time.sleep(1 * (_ + 1))
                continue
        return []

    def quark(self, queries: List[str], max_retry: int = 10):
        url = "http://101.37.167.147/gw/v1/api/msearch-sp/qwen-search-poc"
        headers = {"Authorization": f"Bearer {QWEN_SEARCH_KEY}", 
                    "Content-Type": "application/json",
                    "Host": "pre-nlp-cn-hangzhou.aliyuncs.com"}
        template = {
            "rid": "uniform-eval-" + str(uuid.uuid4()),
            "scene": "dolphin_search_quark_nlp",
            "uq": queries[0],
            "debug": False,
            "fields": [],
            "page": 1,
            "rows": 10,
            "customConfigInfo": {
                "multiSearch": False,
                "qpMultiQueryConfig": queries,
                "qpMultiQuery": True,
                "qpMultiQueryHistory": [],
                "qpSpellcheck": False,
                "qpEmbedding": False,
                "knnWithScript": False,
                "rerankSize": 10,
                "qpTermsWeight": False,
                "qpToolPlan": False,
                "inspection": False, #关闭绿网
                "readpage": True,
                "readpageConfig": {"tokens": 4000, "topK": 10, "onlyCache": False},
            },
            "rankModelInfo": {
                "default": {
                    "features": [
                        {"name": "static_value", "field": "_weather_score", "weights": 1.0},
                        {
                            "name": "qwen-rerank",
                            "fields": ["hostname", "title", "snippet", "timestamp_format"],
                            "weights": 1,
                            "threshold": -50,
                            "max_length": 512,
                            "rank_size": 100,
                            "norm": False,
                        },
                    ],
                    "aggregate_algo": "weight_avg",
                }
            },
            "headers": {
                "__d_head_qto": 20000,
                "__d_head_app": USER_NAME
                },
        }

        resp = ""
        for _ in range(max_retry):
            try:
                resp = requests.post(url, headers=headers, data=json.dumps(template))
                rst = json.loads(resp.text)
                docs = rst["data"]["docs"]
                assert len(docs) != 0, template['rid'] + " 搜索为空，重试"
                return docs
            except Exception as e:
                print("Meet error when search query:", resp, queries, e, template['rid'])
                print("retrying")
                time.sleep(1 * (_ + 1))
                continue
        return []
    
    def baike(self, queries: List[str], max_retry: int = 10):
        only_cache = os.getenv('NLP_WEB_SEARCH_ONLY_CACHE', 'true').lower() in ('y', 'yes', 't', 'true', '1', 'on')
        url = "http://dolphin.msearch.cn-hangzhou.alicontainer.com/dolphin/api/search"
        headers = {"Content-Type": "application/json"}
        docs = []
        for query in queries:
            payload = json.dumps({
                "rid": "uniform-eval-" + str(uuid.uuid4()),
                "headers": {
                "__d_head_qto": 100000
                },
                "scene": "demo_index_baike_ch_0106", 
                "uq": query,
                "type": "content,title",
                "debug": False,
                "fields": ["content"],
                "customConfigInfo": {
                    "multiSearch": False,
                    "qpMultiQueryConfig": [],
                    "qpMultiQuery": False,
                    "qpMultiQueryHistory": [],
                    "qpSpellcheck": False,
                    "qpEmbedding": False,
                    "knnWithScript": False,
                    "rerankSize": 10,
                    "qpTermsWeight": False,
                    "qpToolPlan": False,
                    "inspection": False,
                    "readpageConfig": {"tokens": 4000, "topK": 10, "onlyCache": only_cache},
                    },
                "rankModelInfo": {
                    "default": {
                        "features": [
                            {"name": "static_value", "field": "_weather_score", "weights": 1.0},
                            {
                                "name": "qwen-rerank",
                                "fields": ["hostname", "title", "snippet", "timestamp_format"],
                                "weights": 1,
                                "threshold": -50,
                                "max_length": 512,
                                "rank_size": 100,
                                "norm": False,
                            },
                        ],
                        "aggregate_algo": "weight_avg",
                    }
                },
                "headers": {
                    "__d_head_qto": 20000,
                    "__d_head_app": USER_NAME
                },
                "page": 1,
                "rows": 10
                })
           

            resp = ""
            for _ in range(max_retry):
                try:
                    resp = requests.request("POST", url, headers=headers, data=payload)
                    rst = json.loads(resp.text)
                    p_docs = rst["data"]["docs"]
                    for d in p_docs:
                        d["snippet"] = d["content"]
                        d.pop("content")

                    assert len(p_docs) != 0, "搜索为空，重试"
                    docs.append(p_docs)
                    break
                except Exception as e:
                    print("Meet error when search query:", resp, query, e)
                    print("retrying")
                    time.sleep(1 * ( _ + 1))
                    continue
        res = []
        for j in range(10):
            for i in range(len(docs)):
                if j < len(docs[i]):
                    res.append(docs[i][j])
        return res[:10]

    def _choose_search_engine(self, search_engine_name):
        engines = {"google": self.google, "quark": self.quark, "baike": self.baike}
        if search_engine_name not in engines:
            raise ValueError(f"Unsupported search engine: {search_engine_name}")
        return engines[search_engine_name]

    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            params = self._verify_json_format_args(params)
            queries = params["queries"]
        except:
            return "[Web Search] Invalid request format: Input must be a JSON object containing 'queries' field"

        if len(queries) == 0:
            return "[Web Search] Empty search queries."

        search_method = self._choose_search_engine(SEARCH_ENGINE)
        ctxs = []
        # for q in queries:
        if SEARCH_STRATEGY == "incremental":
            for q in queries:
                ctxs += search_method([q], max_retry=10)
        else:
            ctxs = search_method(queries, max_retry=10)
        resp = get_online_prompt(ctxs, queries=queries, topk=len(ctxs), topk_readpage=len(ctxs), toolResult=[])
        return resp


if __name__ == "__main__":
    # os.environ['NLP_WEB_SEARCH_ONLY_CACHE'] = 'false'
    # os.environ['NLP_WEB_SEARCH_ENABLE_READPAGE'] = 'true'
    # os.environ['NLP_WEB_SEARCH_ENABLE_SFILTER'] = 'true'
    print(WebSearch().call({"queries": ["'Boston Terrier dog black and white short face compact build'"]}))