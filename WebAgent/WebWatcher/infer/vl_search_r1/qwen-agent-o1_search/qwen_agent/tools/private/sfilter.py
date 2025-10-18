import re, requests
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
from tqdm import tqdm
from qwen_agent.log import logger


SFILTER_MAX_WORKERS = int(os.getenv('NLP_WEB_SEARCH_SFILTER_MAX_WORKERS', 4))

def multi_call_sfilter(query, ctxs):
    args_list = [{'query':query, 'ctx':i} for i in ctxs]
    results = call_func_in_threads(sfilter, args_list)
    return results

def call_func_in_threads(func, args_list: List[Dict[str, Any]]) -> List[Any]:
    max_workers = SFILTER_MAX_WORKERS
    results = [None] * len(args_list)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(func, **args): i for i, args in enumerate(args_list)}
        for future in as_completed(futures):
            index = futures[future]
            results[index] = future.result()
    return results

def _replace_with_separator(text, separator, regexs):
    replacement = r"\1" + separator + r"\2"
    result = text
    for regex in regexs:
        result = regex.sub(replacement, result)
    return result

def split_sentence_func(text, best=True):
    _SEPARATOR = r'@'
    _RE_SENTENCE = re.compile(r'(\S.+?[.!?])(?=\s+|$)|(\S.+?)(?=[\n]|$)', re.UNICODE)
    _AB_SENIOR = re.compile(r'([A-Z][a-z]{1,2}\.)\s(\w)', re.UNICODE)
    _AB_ACRONYM = re.compile(r'(\.[a-zA-Z]\.)\s(\w)', re.UNICODE)
    _UNDO_AB_SENIOR = re.compile(r'([A-Z][a-z]{1,2}\.)' + _SEPARATOR + r'(\w)', re.UNICODE)
    _UNDO_AB_ACRONYM = re.compile(r'(\.[a-zA-Z]\.)' + _SEPARATOR + r'(\w)', re.UNICODE)
    sentences = []
    text = re.sub(r'([。！？?])([^”’])', r"\1\n\2", text)
    text = re.sub(r'(\.{6})([^”’])', r"\1\n\2", text)
    text = re.sub(r'(…{2})([^”’])', r"\1\n\2", text)
    text = re.sub(r'([。！？?][”’])([^，。！？?])', r'\1\n\2', text)
    for chunk in text.split("\n"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if not best:
            sentences.append(chunk)
            continue
        processed = _replace_with_separator(chunk, _SEPARATOR, [_AB_SENIOR, _AB_ACRONYM])
        sents = list(_RE_SENTENCE.finditer(processed))
        if not sents:
            sentences.append(chunk)
            continue
        for sentence in sents:
            sentence = _replace_with_separator(sentence.group(), r" ", [_UNDO_AB_SENIOR, _UNDO_AB_ACRONYM])
            sentences.append(sentence)
    return sentences


def sfilter(query, ctx, max_retry=20):
    url = "https://poc-dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
    api_key = os.environ.get('NLP_DASH_KEY', '')
    assert api_key

    ## 首先判断ctx中是否包含web_main_body，如果不包含，则直接返回snippet
    if 'web_main_body' not in ctx or len(ctx.get("web_main_body", "")) <= len(ctx["snippet"]):
        return ctx
    
    all_sentence_list = split_sentence_func(ctx['snippet'] + '\n' + ctx['web_main_body'])
    all_sentence = ''
    for i in range(len(all_sentence_list)):
        all_sentence += f'句子 {i + 1}: {all_sentence_list[i]}\n'

    prompt_extraction_sft_ch = '''指令：请提取与查询相关的句子并输出其索引。如果没有相关句子，直接输出0。 
查询：{query} 
句子：{sentence} 
输出：'''

    headers = {
              'Authorization': api_key,
              'Content-Type': 'application/json',
            }
    payload = json.dumps({
              "model": "pre-sfilter-0.1-server",
              "input": {
                "prompt": f'''<|im_start|>system\n<|im_end|>\n<|im_start|>user\n{prompt_extraction_sft_ch.format(query=query, sentence=all_sentence)}<|im_end|>\n<|im_start|>assistant\n'''
              },
              "parameters":{
                "enable_search": False,
                "temperature": 0,
                "max_tokens": 150,
                "presence_penalty": 1
            }
            })
    
    retries = 0
    relevant_sentence_id = None
    while relevant_sentence_id is None and retries <= max_retry:
        try: 
            relevant_sentence_id = requests.request("POST", url, headers=headers, data=payload, timeout=10).text
            relevant_sentence_id = json.loads(relevant_sentence_id)['output']['text']
        except Exception as e:
            print(e)

    if not relevant_sentence_id:
        relevant_sentence_id == '0'
    
    ## 如果返回0，说明没有相关句子，直接返回snippet
    if relevant_sentence_id == '0':
        compressed_text = ctx['snippet']
    else:
        try:
            relevant_sentence_id = relevant_sentence_id.split('\n')[0]
            split_sentence_id = sorted([int(elem) for elem in relevant_sentence_id.strip().split(' ')])
            split_sentence_text = [all_sentence_list[id - 1] for id in split_sentence_id]
            compressed_text = '\n'.join(split_sentence_text)
        except:
            relevant_sentence_id = '0'
            compressed_text = ctx['snippet']

    # logger.info(f'sfilter_res{relevant_sentence_id}')  
    ctx['snippet'] = compressed_text

    return ctx

if __name__ == '__main__':
    ctx = {
            "snippet": "综上所述，转融通出借在增加市场流动性、降低市场波动性以及为证券公司带来盈利等方面都具有积极作用，因此被视为市场的利好因素。这不仅能促进市场的稳定发展，还能为投资者提供更多投资选择和",
            "_score_QtcTeacherQtc": 1.38867,
            "title": "转融通出借是利空还是利好-百度知道",
            "_weather_score": 0.0,
            "_score": 0.4496745467185974,
            "url": "https://zhidao.baidu.com/question/692881750669217932.html",
            "_score_mergeScoreRelevance": 1.0229,
            "sc_name": "structure_web_info",
            "timestamp_format": "2024-06-26 06:35:53",
            "web_main_body": "转融通出借是利好。\n转融通出借对股票市场来说是一个积极的信号。转融通指的是证券公司将自有或客户的证券资产出借给交易所或其他投资者进行交易，增加了市场的证券供给，有助于稳定市场供需关系。\n第一，转融通出借增加了市场的流动性。当证券公司将其持有的证券进行出借时，意味着市场上增加了可供交易的证券数量，有助于满足投资者的买卖需求，增强市场的流动性。\n第二，转融通出借有助于降低证券市场的波动性。通过增加市场供给，可以平稳市场价格，减少因供需失衡导致的剧烈波动。这对于维护市场的稳定，保护投资者的利益是有积极作用的。\n第三，转融通出借对于证券公司而言也是一种盈利模式。证券公司可以通过出借证券资产获得一定的收益，这有助于提升证券公司的盈利能力，进一步推动其业务创新和发展。\n综上所述，转融通出借在增加市场流动性、降低市场波动性以及为证券公司带来盈利等方面都具有积极作用，因此被视为市场的利好因素。这不仅能促进市场的稳定发展，还能为投资者提供更多投资选择和机会。",
            "hostname": "百度知道",
            "hostlogo": "https://gw.alicdn.com/L1/723/1555644928/27/59/5d/27595df8e57a01f4e32b2570e3997af4.jpeg",
            "_scores":
            {
                "cross_ranker": 0.4496745467185974,
                "static_value(_weather_score)": 0.0
            },
            "_score_FISRT_PARAL_AUTHORITY_SCORE": 0.0,
            "_score_CLICKG_SCORE_0": 1.5235,
            "_id": "quark-18",
            "timestamp": 1719354953
        }
    query = '转融通出借是有助于降低证券市场的波动性，还是有害？'
    multi_call_sfilter(query, [ctx]*10)
    for i in tqdm(range(10000)):
        ctx = sfilter(query, ctx)
        print(len(ctx['web_main_body']))
        print(ctx['snippet'])