#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%
"""
OpenAI GPT-4o Model Interface

Provides a unified calling interface for OpenAI GPT-4o model, used for text generation tasks in the WebSailor framework.

Main features:
- Support for system and user message generation
- Automatic configuration management and parameter settings
- Error handling and retry mechanisms
- Streaming and non-streaming response support

Author: Evan Zuo
Date: January 2025
"""

import time
from typing import Optional, Dict, Any, List
from tenacity import retry, stop_after_attempt, wait_exponential
import httpx

from websailor.utils.logger import get_logger

try:
    import openai
    from openai import OpenAI
except ImportError:
    raise ImportError("Please install openai library: pip install openai")

logger = get_logger(__name__)

# @retry(
#     stop=stop_after_attempt(3),
#     wait=wait_exponential(multiplier=1, min=4, max=10)
# )
def generate(
    api_key: str,
    system: str,
    user: str,
    model: str = "gpt-4o",
    temperature: float = 0.3,
    max_tokens: int = 8124,
    top_p: float = 1.0,
) -> str:
    """
    Generate text response
    
    Args:
        system: System prompt
        user: User input
        model: Model name
        temperature: Temperature parameter, controls randomness
        max_tokens: Maximum number of tokens to generate
        top_p: Core sampling parameter
        
    Returns:
        Generated text content
        
    Raises:
        ValueError: Parameter error
        Exception: API call failure
    """
    
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.openai.com/v1",
        timeout=httpx.Timeout(60.0, connect=10.0)  # Set timeout
    )
        
    try:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
        
        # Use streaming response
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stream=True  # Enable streaming output
        )
        
        # Collect complete response
        collected_content = []
        retry_count = 0
        max_retries = 3
        
        while retry_count < max_retries:
            try:
                # Process streaming response
                for chunk in response:
                    if chunk.choices[0].delta.content is not None:
                        content_piece = chunk.choices[0].delta.content
                        print(content_piece, end="", flush=True)  # Real-time printing
                        collected_content.append(content_piece)
                break  # If successful, exit retry loop
            except (httpx.RemoteProtocolError, httpx.ReadTimeout) as e:
                retry_count += 1
                if retry_count >= max_retries:
                    logger.error(f"Streaming transmission failed, retried {retry_count} times: {e}")
                    raise
                logger.warning(f"流式传输中断，正在重试 ({retry_count}/{max_retries}): {e}")
                time.sleep(2 ** retry_count)  # 指数退避
                continue
        
        print()  # 打印换行
        
        # 合并所有内容
        full_content = "".join(collected_content)
        return full_content.strip() if full_content else ""
        
    except openai.RateLimitError as e:
        logger.warning(f"OpenAI API 速率限制: {e}")
        raise
    except openai.AuthenticationError as e:
        logger.error(f"OpenAI API 认证失败: {e}")
        raise
    except openai.APIError as e:
        logger.error(f"OpenAI API 错误: {e}")
        raise
    except httpx.TimeoutException as e:
        logger.error(f"请求超时: {e}")
        raise
    except Exception as e:
        logger.error(f"GPT-4o 调用失败: {e}")
        raise