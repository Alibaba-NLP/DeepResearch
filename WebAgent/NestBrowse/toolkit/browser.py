import os
import re
import time
import json
import yaml
import asyncio
import aiohttp
import tiktoken
import requests
from typing import Dict, List, Optional, Union

from toolkit.mcp_client import *
from toolkit.tool_explore import process_response


class Visit:
    tool_schema = {
        "type": "function",
        "function": {
            "name": "visit",
            "description": "Visit the webpage and return a summary of its content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL of the webpage to visit.",
                    },
                    "goal": {
                        "type": "string",
                        "description": "The goal or intent of visiting the webpage.",
                    },
                },
                "required": ["url", "goal"],
            }
        }
    }

    async def call(self, params, **kwargs):
        try:
            if isinstance(params, str):
                params = json.loads(params)
            elif isinstance(params, dict):
                pass
            else:
                raise ValueError
            url = params['url']
            goal = params['goal']
        except:
            return "[visit] Invalid request format: Input must be a JSON object containing `url` and `goal` field."

        try:
            client = kwargs.get('client')
            lock = kwargs.get("lock")
            tokenizer = kwargs.get("tokenizer")
            sem = kwargs.get("sem")
            async with lock:
                response = await client.call_tool('browser_navigate', {'url': url})
            raw_response_text = response.content[0].text
        except Exception as e:
            print(f"\n\n\n\n{str(e)}\n\n\n\n")
            return '[visit] Visit error: server-side errors.'

        if not response.isError:
            try:
                response_text, record = await process_response(raw_response_text, goal, os.getenv("SUMMARY_MODEL_NAME", os.getenv("MODEL_NAME")), tokenizer, sem)
                break
            except:
                response_text = "Evidence in page: \n" + "The provided webpage content could not be accessed. Please check the input format." + "\n\n" + "Summary: \n" + "The webpage content could not be processed, and therefore, no information is available."
                record = []

            response_text = f"The useful information in {url} for user goal {goal} as follows: \n\n" + response_text
            return f'[visit] {response_text}', record
        else:
            return f'[visit] Visit error: {raw_response_text}'
    

class Click:
    tool_schema = {
        "type": "function",
        "function": {
            "name": "click",
            "description": "Click the identified element based on the reference index and return a summary of the content after clicking. You are only allowed to click items that come from the latest visit/click tool's clickable results (you can find them in the `Evidence in page` section).",
            "parameters": {
                "type": "object",
                "properties": {
                    "ref": {
                        "type": "string",
                        "description": "The unique identifier for the element to be clicked on the current page. You must use a ref taken from a notation like [ref=XXX], where XXX is the unique identifier.",
                    },
                    "goal": {
                        "type": "string",
                        "description": "The goal or intent of performing this click.",
                    },
                },
                "required": ["ref", "goal"],
            }
        }
    }

    async def call(self, params, **kwargs):
        try:
            if isinstance(params, str):
                params = json.loads(params)
            elif isinstance(params, dict):
                pass
            else:
                raise ValueError
            ref = params['ref']
            goal = params['goal']
        except:
            return f"[click] Invalid request format: Input must be a JSON object containing `ref` and `goal` field."
        
        try:
            client = kwargs.get('client')
            lock = kwargs.get("lock")
            tokenizer = kwargs.get("tokenizer")
            sem = kwargs.get("sem")
            async with lock:
                response = await client.call_tool('browser_click', {'ref': ref, 'element': ''})
            raw_response_text = response.content[0].text
        except:
            return '[click] Click error: server-side errors.'
        
        if not response.isError:
            try:
                response_text, record = await process_response(raw_response_text, goal, os.getenv("SUMMARY_MODEL_NAME", os.getenv("MODEL_NAME")), tokenizer, sem)
                break
            except:
                response_text = "Evidence in page: \n" + "The provided webpage content could not be accessed. Please check the input format." + "\n\n" + "Summary: \n" + "The webpage content could not be processed, and therefore, no information is available."
                record = []

            response_text = f"The useful information after clicking [ref={ref}] for user goal {goal} as follows: \n\n" + response_text
            return f'[click] {response_text}', record
        else:
            return f'[click] Click error: {raw_response_text}'


class Fill:
    tool_schema = {
        "type": "function",
        "function": {
            "name": "fill",
            "description": "Enter text content into the input field and return the filled state. You are only allowed to fill items that come from the latest visit/click tool's fillable results (you can find them in the `Evidence in page` section).",
            "parameters": {
                "type": "object",
                "properties": {
                    "ref": {
                        "type": "string",
                        "description": "The unique identifier for the element to be filled. You must use a ref taken from a notation like [ref=XXX], where XXX is the unique identifier.",
                    },
                    "text": {
                        "type": "string",
                        "description": "The content entered into the textbox.",
                    },
                },
                "required": ["ref", "text"],
            }
        }
    }

    async def call(self, params, **kwargs):
        try:
            if isinstance(params, str):
                params = json.loads(params)
            elif isinstance(params, dict):
                pass
            else:
                raise ValueError
            ref = params['ref']
            text = params['text']
        except:
            return "[fill] Invalid request format: Input must be a JSON object containing `ref` and `text` fields."

        try:
            client = kwargs.get('client')  
            lock = kwargs.get("lock")
            async with lock:
                response = await client.call_tool('browser_type', {
                    'ref': ref,
                    'submit': False,
                    'text': text,
                    'element': ""
                })
            response_text = response.content[0].text
        except:
            return '[fill] Fill error: server-side errors.'

        if not response.isError:
            return f'[fill] Successfully filled `{text}` into the field [ref={ref}].'
        else:
            return f'[fill] Fill error: {response_text}'