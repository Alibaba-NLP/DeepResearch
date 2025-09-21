"""
input:
    - query/goal: str
    - Docs: List[file]/List[url]
    - file type: 'pdf', 'docx', 'pptx', 'txt', 'html', 'csv', 'tsv', 'xlsx', 'xls', 'doc', 'zip', '.mp4', '.mov', '.avi', '.mkv', '.webm', '.mp3', '.wav', '.aac', '.ogg', '.flac'
output:
    - answer: str
    - useful_information: str
"""
import sys
import os
import re
import time
import copy
import json
from typing import Dict, Iterator, List, Literal, Tuple, Union, Any, Optional
import json5
import asyncio
from openai import OpenAI, AsyncOpenAI
import pdb
import bdb

from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.agents import Assistant
from qwen_agent.llm import BaseChatModel
from qwen_agent.settings import DEFAULT_WORKSPACE, DEFAULT_MAX_INPUT_TOKENS
from qwen_agent.llm.schema import ASSISTANT, USER, FUNCTION, Message, DEFAULT_SYSTEM_MESSAGE, SYSTEM, ROLE
from qwen_agent.tools import BaseTool
from qwen_agent.log import logger
from qwen_agent.utils.tokenization_qwen import count_tokens, tokenizer
from qwen_agent.settings import DEFAULT_WORKSPACE, DEFAULT_MAX_INPUT_TOKENS

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir)) 
sys.path.append('../../')  

from file_tools.file_parser import SingleFileParser, compress
from file_tools.video_agent import VideoAgent

FILE_SUMMARY_PROMPT = """
Please process the following file content and user goal to extract relevant information:

## **File Content** 
{file_content}

## **User Goal**
{goal}

## **Task Guidelines**
1. **Content Scanning for Rational**: Locate the **specific sections/data** directly related to the user's goal within the file content
2. **Key Extraction for Evidence**: Identify and extract the **most relevant information** from the content, you never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.
3. **Summary Output for Summary**: Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal.
""".strip()


async def file_parser(params, **kwargs):
    """Parse files with automatic path resolution"""
    urls = params.get('files', [])
    if isinstance(urls, str):
        urls = [urls]

    resolved_urls = []
    for url in urls:
        if isinstance(url, list):
            for sub_url in url:
                if sub_url.startswith(("http://", "https://")):
                    resolved_urls.append(sub_url)
                else:
                    abs_path = os.path.abspath(sub_url)
                    if os.path.exists(abs_path):
                        resolved_urls.append(abs_path)
                    else:
                        resolved_urls.append(sub_url)
        else:
            if url.startswith(("http://", "https://")):
                resolved_urls.append(url)
            else:
                abs_path = os.path.abspath(url)
                if os.path.exists(abs_path):
                    resolved_urls.append(abs_path)
                else:
                    resolved_urls.append(url)

    results = []
    file_results = []
    for url in resolved_urls:
        try:
            result = SingleFileParser().call(json.dumps({'url': url}), **kwargs)
            results.append(f"# File: {os.path.basename(url)}\n{result}")
            file_results.append(result)
        except Exception as e:
            results.append(f"# Error processing {os.path.basename(url)}: {str(e)}")
    if count_tokens(json.dumps(results)) < DEFAULT_MAX_INPUT_TOKENS:
        return results
    else:
        return compress(file_results)

# @register_tool("file_parser")
class FileParser(BaseTool):
    name = "parse_file"
    description = "This is a tool that can be used to parse multiple user uploaded local files such as PDF, DOCX, PPTX, TXT, CSV, XLSX, DOC, ZIP, MP4, MP3."
    parameters = [
        {
            'name': 'files',
            'type': 'array',
            'array_type': 'string',
            'description': 'The file name of the user uploaded local files to be parsed.',
            'required': True
        }
    ]

    async def call(self, params, file_root_path):
        file_name = params["files"]
        outputs = []
        
        file_path = []
        omnifile_path = []
        for f_name in file_name:
            if '.mp3' not in f_name:
                file_path.append(os.path.join(file_root_path, f_name))
            else:
                omnifile_path.append(os.path.join(file_root_path, f_name))

        if len(file_path):
            params = {'files': file_path}
            response = await file_parser(params)
            response = response[:30000]

            parsed_file_content = ' '.join(response)
            outputs.extend([f'File token number: {len(parsed_file_content.split())}\nFile content:\n']+response)

        
        if len(omnifile_path):
            params['files'] = omnifile_path
            agent = VideoAgent()
            res = await agent.call(params)

            res = json.loads(res)
            outputs += res
        
        return outputs
