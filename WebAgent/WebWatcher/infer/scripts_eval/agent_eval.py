import base64
import json
import re
import requests
import math
from io import BytesIO
import os

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from contextlib import redirect_stdout, redirect_stderr
import sys
from openai import OpenAI
from PIL import Image, ImageDraw
from transformers import AutoProcessor
import argparse

from mmrag_r1.llm_agent.qwen_tool_call import Qwen_agent
import datetime

sys_prompt = """\
You are a Web Information Seeking Master. Your task is to thoroughly seek the internet for information and provide accurate answers to visual questions.
As you proceed, adhere to the following principles:

1. Decompose the original visual question into sub-questions and solve them step by step. Summarize the knowledge obtained from the previous round of dialogue, then think about what is next sub-question.
2. Whether you can answer the question or not, you should describe the image in detail. if the image includes multiple sub-image, you should describe each one separately.
3. You should provide the final answer within 10 turns, regardless of whether all valid information has been collected.
"""

prompt_ins =  '''\
You are an intelligent agent engaged in a conversation with a user. The user poses a question and provides a corresponding image for context. As an agent, you approach the problem with care and methodical precision, following a multi-step process to arrive at a solution. You utilize a variety of tools, ensuring that the information gathered from each one is cross-validated before you reach a final answer. Rather than relying on any single tool for accuracy, you employ multiple tools iteratively to prioritize the comprehensiveness and reliability of your responses.
<tools>
{
  "name": "web_search",
  "description": "Call this tool to interact with the web_search API. You will receive the top 10 text excerpts from Google's text search engine using text as the search query.",
  "parameters": {
    "type": "object",
    "properties": {
      "queries": {
        "type": "array",
        "items": {
          "type": "string",
          "description": "The search query."
          },
        "description": "The list of search queries."
        }
      },
    "required": [
      "queries"
      ]
    }
},
{
  "name": "VLSearchImage",
  "description": "Call this tool to receive the top 10 images and corresponding descriptions from Google's image search engine. You can only search the input image and cannot conduct additional searches on the results obtained from the initial search. You'd better use this tool only once",
  "parameters": {
    "type": "object",
    "properties": {
        "image_urls": {
            "type": "array",
            "items": {"type": "string", "description": "The search image url."},
            "description": "The list of search image url."
      }
    },
    "required": [
      "image_urls"
    ]

  }
},
{
    "name": "visit",
    "description": "visit a webpage and return the summary of webpage.",
    "parameters": {
        "type": "object",
        "properties": {
        "url": {
            "type": "string",
            "description": "the url you want to explore."
        },
        "goal": {
            "type": "string",
            "description": "the goal of the visit for the webpage."
        }
        },
        "required": ["url","goal"]
    }
},
{
    "name": "code_interpreter",
    "description": "Call this tool to execute Python code for calculation, data analysis, or content extraction tasks.",
    "parameters": {
        "type": "object",
        "properties": {
        "code": {
            "type": "string",
            "description": "The Python code to execute."
        },
        "required": ["code"]
    }
}
</tools>

The assistant starts with one or more cycles of (thinking about which tool to use -> performing tool call -> waiting for tool response), and ends with (thinking about the answer -> answer of the question). The thinking processes, tool calls, tool responses, and answer are enclosed within their tags. There could be multiple thinking processes, tool calls, tool call parameters and tool response parameters.

Example response:
<think> thinking process here </think>
<tool_call>
{"name": "tool name here", "arguments": {"parameter name here": parameter value here, "another parameter name here": another parameter value here, ...}}
</tool_call>
<tool_response>
{"name": "tool name here", "content": {"result name here": result value here, "another result name here": another 
result value here, ...}}
</tool_response>
<think> thinking process here </think>
<tool_call>
{"name": "another tool name here", "arguments": {...}}
</tool_call>
<tool_response>
{"name": "another tool name here", "content": {...}}
</tool_response>
(more thinking processes, tool calls and tool responses here)
<think> thinking process here </think>
<answer> answer here </answer>

Input Question:{Question}
Input image:{Image_url}
'''


class OmniSearch:
    def __init__(self, 
                base_url='http://0.0.0.0:8001/v1', 
                api_key='EMPTY'):
        
        self.client = OpenAI(base_url=base_url, api_key=api_key)

        self.max_pixels = 1024 * 28 * 28
        self.min_pixels = 256 * 28 * 28
        self.repeated_nums = 1
        self.max_steps = 12

        self.qwen_agent = Qwen_agent(function_list=['web_search','VLSearchImage','visit','code_interpreter'])
        self.processor = AutoProcessor.from_pretrained(os.getenv('VLLM_MODEL', ''))

    def process_image(self, image):
        if isinstance(image, dict):
            image = Image.open(BytesIO(image['bytes']))
        elif isinstance(image, str):
            image = Image.open(image)

        if (image.width * image.height) > self.max_pixels:
            resize_factor = math.sqrt(self.max_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if (image.width * image.height) < self.min_pixels:
            resize_factor = math.sqrt(self.min_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        byte_stream = BytesIO()
        image.save(byte_stream, format="JPEG")
        byte_array = byte_stream.getvalue()
        base64_encoded_image = base64.b64encode(byte_array)
        base64_string = base64_encoded_image.decode("utf-8")
        base64_qwen = f"data:image;base64,{base64_string}"

        return image, base64_qwen
    
    def search(self,query):
        if isinstance(query,str):
            query = [query]
        # search_response = requests.get(self.search_url, params={"queries": query})
        search_results = search_response.json()
        image_path_list = [result['image_file'] for result in search_results[0]]
        return image_path_list
    
    def run_main(self, sample):
        self.image_raw = []
        self.image_input = []
        self.image_path = []
        test_image_dir = os.getenv("IMAGE_DIR")
        test_image_name = os.path.basename(sample['file_path'])
        image_raw = Image.open(os.path.join(test_image_dir, test_image_name))
        _, test_img_base64 = self.process_image(image_raw)

        messages = [dict(
                role="system",
                content=[
                    {
                        "type": "text",
                        "text": sys_prompt,
                    }
                ]
            ),
            dict(
                role="user",
                content=[
                    {
                        "type": "text",
                        "text": prompt_ins.replace("{Image_url}", sample['file_path']).replace("{Question}", sample['prompt']),
                    },
                    {
                        'type': 'image_url',
                        'image_url': {
                            'url': test_img_base64
                        }
                    }
                ]
            )]

        max_steps = self.max_steps
        while True:
            ## assistant
            gen_times = 10
            while True:
                if gen_times < 0:
                    return 'time_out', messages, 'No answer'
                try:
                    response = self.client.chat.completions.create(
                        model=os.getenv('VLLM_MODEL', ''),
                        messages=messages,
                        stream=False,
                        top_p=0.95,
                        temperature=0.6
                    )
                    response_content = response.choices[0].message.content
                    # break
                    if response_content:
                        break
                    else:
                        raise Exception('vllm model failed, retrying...')
                except Exception as e:
                    print(e)
                    gen_times -= 1
            
            messages.append(dict(
                role="assistant",
                content=[{
                    "type": "text",
                    "text": response_content
                }]
            ))


            ## think
            pattern = r'<think>(.*?)</think>'
            match = re.search(pattern, response_content, re.DOTALL)
            # thought = match.group(1)
            if match:
                thought = match.group(1)
            else:
                pattern = r'<think>(.*?)\n\n<Sub-Question>'
                print("[no valid <think> tag]")
                # print(f"messages:{messages}")

            print("response_content: ", response_content)
            ## opration
            pattern = r'<(tool_call|answer)>(.*?)</\1>'
            match = re.search(pattern, response_content, re.DOTALL)
            if match:
                raw_content = match.group(0)
                content = match.group(2).strip()  # Return only the content inside the tags
                action = match.group(1)
            else:
                print(f"[No match tool_call or answer]")
                content = ''
                action = None

            print("action: ", action)
            ## whether end
            if action is None:
                user_content=[{
                    'type': 'text',
                    'text': 'please use valid tool or answer the question.' 
                }]
            elif action == 'answer':
                return 'answer', messages, content
            elif max_steps==0:
                return 'time_out', messages, 'No answer'
            elif action == 'tool_call':
                # request_para = None
                try:
                    try:
                        request_para = json.loads(content)
                        print(request_para)
                    except Exception as e:
                        # Step 1: 修复 queries 字段中带双引号的字符串
                        content = re.sub(r'"\s*queries\s*"\s*:\s*\[\s*""(.*?)""\s*\]', r'"queries": ["\1"]', content)
                        # Step 2: 如果还有 "" 替换成 "
                        content = content.replace('""', '"')
                        request_para = json.loads(content)

                    if request_para is None:
                        raise Exception(f"Invalid request parameters. request_para is None, content:{content}")

                    img_save_path = 'scripts_eval/scripts_eval/images/search_image/' + datetime.datetime.now().strftime("%m%d")
                    if not os.path.exists(img_save_path):
                        os.makedirs(img_save_path)

                    if request_para['name'] in ['VLSearchImage', 'vlsearchimage']:
                        user_query = sample.get('prompt', '')
                        search_results = self.qwen_agent._call_tool(
                            request_para['name'],
                            request_para['arguments']
                        )
                    else:
                        search_results = self.qwen_agent._call_tool(
                            request_para['name'],
                            request_para['arguments'],
                            img_save_path=img_save_path,
                            byte=True
                        )
                    print(search_results)
                    
                except Exception as e:
                    print(e, f"Invalid request parameters. request_para is None, content:{content}")
                    request_para = None
                    
                if request_para is None:
                    user_content = [{
                        'type': 'text',
                        'text': 'please use valid tool or answer the question.'
                    }]
                elif request_para['name'] in ['VLSearchImage']:
                    ## prefix
                    user_content = [{
                        'type': 'text',
                        'text': '<tool_response>'
                    }]

                    ## content
                    images_path = re.findall(r"Image: (.*?), Text:", search_results)
                    text_description = re.findall(r"Text: (.*?)\nImage:", search_results)
                    text_description_last = re.findall(r"Text: (.+)$", search_results)
                    text_description_list = text_description + text_description_last
                    # images_path = []
                    if len(images_path)>0:
                        for image_path in images_path:
                            image_raw = Image.open(image_path)
                            image_input, img_base64 = self.process_image(image_raw)
                            user_content.append({
                                'type': 'image_url',
                                'image_url': {
                                    'url': img_base64
                                }
                            })
                        user_content.append({
                            'type': 'text',
                            'text': search_results
                        })
                    else:
                        user_content.append({
                            'type': 'text',
                            'text': search_results
                        })
                    
                    ## suffix
                    user_content.append({
                        'type': 'text',
                        'text': '</tool_response>'
                    })
                    
                elif request_para['name'] in ['web_search','visit','Visit']:
                    user_content=[{
                        'type': 'text',
                        'text': '<tool_response>' + search_results + '</tool_response>'
                    }]

                elif request_para['name'] in ['Code_Interpreter', 'code_interpreter', 'PythonInterpreter']:
                    if isinstance(search_results, dict):
                        code_result = json.dumps(search_results, ensure_ascii=False)
                    else:
                        code_result = str(search_results)
                    print("Generated Code: ", code_result)
                    user_content = [{
                        'type': 'text',
                        'text': f'<tool_response>{code_result}</tool_response>'
                    }]

                else:
                    user_content = [{
                        'type': 'text',
                        'text': 'please use valid tool or answer the question.'
                    }]
                
            max_steps -= 1
            if max_steps == 0:
                user_content.append({
                    'type': 'text',
                    'text': 'please answer the question now with answer in <answer> ... </answer>' 
                })
            messages.append(dict(
                role='user',
                content=user_content
            ))

    def infer(self, sample):
        try:
            status, messages, content = self.run_main(sample)
        except Exception as e:
            sample["response"] = e
            sample["gen"] = 'No Answer'
            print(e)
            return sample

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        sample["response"] = text
        sample["gen"] = content
        
        return sample
    
    def infer_with_timeout_retry(self, sample, max_retry=2, timeout_seconds=300):
        for attempt in range(max_retry+1):
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self.infer, sample)
                try:
                    result = future.result(timeout=timeout_seconds)
                    return result
                except TimeoutError:
                    print(f"[Timeout] Inference timeout for sample {sample.get('file_path','N/A')}, retry {attempt+1}/{max_retry}")
                except Exception as e:
                    print(f"[Exception] Inference error for sample {sample.get('file_path', 'N/A')}: {e}, retry {attempt+1}/{max_retry}")
        print(f"[Fail] Inference failed after {max_retry+1} attempts for sample {sample.get('file_path','N/A')}")
        sample['response'] = 'Timeout/Error'
        sample['gen'] = 'Timeout/Error'
        return sample

    def eval(self, input_file_list, output_file):
        data = []
        for input_file in input_file_list:
            with open(input_file,'r') as f:
                data.extend([json.loads(line) for line in f])
        max_workers = 20
        results = [None] * len(data)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {executor.submit(self.infer_with_timeout_retry, sample): idx for idx, sample in enumerate(data)}
            for future in tqdm(as_completed(future_to_index), total=len(data), desc="Inference Progress"):
                idx = future_to_index[future]
                try:
                    res = future.result()
                except Exception as e:
                    print(f"[Fatal] Sample {data[idx].get('file_path','')} error: {e}")
                    res = {'response': 'Timeout/Error', 'gen': 'Timeout/Error'}
                results[idx] = res
                with open(output_file, 'a') as f:
                    json.dump(res, f, ensure_ascii=False)
                    f.write('\n')


if __name__ == '__main__':
    agent = OmniSearch()

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--eval_data", type=str, required=True)
    args = parser.parse_args()
        
    output_file = args.output_file

    if args.eval_data == 'hle':
        agent.eval(['vl_search_r1/eval_data/hle.jsonl'], output_file)
    elif args.eval_data == 'gaia':
        agent.eval(['vl_search_r1/eval_data/gaia.jsonl'], output_file)
    elif args.eval_data == 'livevqa':
        agent.eval(['vl_search_r1/eval_data/livevqa.jsonl'], output_file)
    elif args.eval_data == 'mmsearch':
        agent.eval(['vl_search_r1/eval_data/mmsearch.jsonl'], output_file)
    elif args.eval_data == 'simplevqa':
        agent.eval(['vl_search_r1/eval_data/simplevqa.jsonl'], output_file)
    elif args.eval_data == 'bc_vl_v1':
        agent.eval(['vl_search_r1/eval_data/bc_vl_v1.jsonl'], output_file)
    elif args.eval_data == 'bc_vl_v2':
        agent.eval(['vl_search_r1/eval_data/bc_vl_v2.jsonl'], output_file)
    else:
        raise ValueError('Invalid eval_data')
