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
import copy
import json
from typing import Dict, Iterator, List, Literal, Tuple, Union, Any, Optional
import json5
import asyncio
from openai import OpenAI

from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.agents import Assistant
from qwen_agent.llm import BaseChatModel
from qwen_agent.llm.schema import ASSISTANT, USER, FUNCTION, Message, DEFAULT_SYSTEM_MESSAGE, SYSTEM, ROLE
from qwen_agent.tools import BaseTool
from qwen_agent.log import logger
from qwen_agent.utils.tokenization_qwen import count_tokens, tokenizer
from qwen_agent.settings import DEFAULT_WORKSPACE, DEFAULT_MAX_INPUT_TOKENS

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))  

from file_tools.video_analysis import VideoAnalysis


async def video_analysis(params, **kwargs):
    """Modified video_analysis to handle multiple URLs"""
    print(params)
    files = params.get('files', [])
    prompt = params.get('prompt', '')

    # Ensure URLs are in a list

    # Process each URL
    results = []
    for file in files:
        try:
            # Create parameters for each URL
            single_url_params = json.dumps({
                'url': file,
                'prompt': prompt
            })
            # Call the original VideoAnalysis tool
            result = VideoAnalysis().call(single_url_params, **kwargs)
            results.append(f"# Video: {os.path.basename(file)}\n{result}")
        except Exception as e:
            results.append(f"# Error processing {os.path.basename(file)}: {str(e)}")

    return results


@register_tool("VideoAgent")
class VideoAgent(BaseTool):
    description = "Video/audio analysis with object detection, text extraction, scene understanding, and metadata analysis."
    parameters = [
        {
            'name': 'query',
            'type': 'string',
            'description': 'Detailed question/instruction for analysis.',
            'required': True
        },
        {
            'name': 'files',
            'type': 'array',
            'array_type': 'string',
            'description': 'The files to be analyzed.',
            'required': True
        }
    ]

    async def call(self, params):
        response = await video_analysis(params)
        return json.dumps(response, ensure_ascii=False)


if __name__ == "__main__":
    agent = VideoAgent()
    params = {
        'query': "Could you help me out with this assignment? Our professor sprung it on us at the end of class Friday, and I'm still trying to figure it out. The question he asked us was about an anagram. I've attached an audio recording of the question that he asked, so if you could please take a listen and give me the answer, I'd really appreciate the help. Please limit your response to the anagram text that could be generated from the original line which fulfills the professor's request, without any other commentary. Also, please don't include any punctuation in your response.",
        'files': ["datas/2b3ef98c-cc05-450b-a719-711aee40ac65.mp3"]
    }
    response = asyncio.run(agent.call(params))
    print(response)
