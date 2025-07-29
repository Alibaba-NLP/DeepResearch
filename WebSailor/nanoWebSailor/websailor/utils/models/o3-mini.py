#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%
"""
OpenAI GPT-4o Model Interface

Provides a unified calling interface for OpenAI GPT-4o model, used for text generation tasks in the WebSailor framework.

Main features:
- Supports system and user message generation
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
import json

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
    model: str = "o3-mini",
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
        Exception: API call failed
    """
    
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.openai.com/v1",
        # timeout=httpx.Timeout(60.0, connect=10.0)  # 设置超时时间
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
            stream=True,  # Enable streaming output
        )
        
        # Collect complete response
        collected_content = []
        thinking_content = []
        current_thinking = []  # For temporarily storing current thinking content
        is_thinking = False    # Flag to mark if currently collecting thinking content
        retry_count = 0
        max_retries = 3
        
        while retry_count < max_retries:
            try:
                # Process streaming response
                for chunk in response:
                    if chunk.choices[0].delta.content is not None:
                        content_piece = chunk.choices[0].delta.content
                        
                        # Check if thinking process starts
                        if "Thinking:" in content_piece:
                            is_thinking = True
                            current_thinking = [content_piece]
                            print("\033[33mThinking: ", end="", flush=True)  # Yellow display for thinking process
                        # Check if thinking process ends (by detecting JSON start)
                        elif content_piece.strip().startswith("{") and is_thinking:
                            is_thinking = False
                            thinking_content.extend(current_thinking)
                            current_thinking = []
                            print("\033[0m")  # Reset color
                            collected_content.append(content_piece)
                            print(content_piece, end="", flush=True)
                        # Process content during thinking
                        elif is_thinking:
                            current_thinking.append(content_piece)
                            print("\033[33m" + content_piece + "\033[0m", end="", flush=True)
                        # Process normal content
                        else:
                            collected_content.append(content_piece)
                            print(content_piece, end="", flush=True)
                            
                break  # If successful, break out of retry loop
            except (httpx.RemoteProtocolError, httpx.ReadTimeout) as e:
                retry_count += 1
                if retry_count >= max_retries:
                    logger.error(f"Streaming transmission failed, retried {retry_count} times: {e}")
                    raise
                logger.warning(f"Streaming transmission interrupted, retrying ({retry_count}/{max_retries}): {e}")
                time.sleep(2 ** retry_count)  # Exponential backoff
                continue
        
        print()  # Print newline
        
        # Merge all content (excluding thinking process)
        full_content = "".join(collected_content)
        return full_content.strip() if full_content else ""
        
    except openai.RateLimitError as e:
        logger.warning(f"OpenAI API rate limit: {e}")
        raise
    except openai.AuthenticationError as e:
        logger.error(f"OpenAI API authentication failed: {e}")
        raise
    except openai.APIError as e:
        logger.error(f"OpenAI API error: {e}")
        raise
    except httpx.TimeoutException as e:
        logger.error(f"Request timeout: {e}")
        raise
    except Exception as e:
        logger.error(f"GPT-4o call failed: {e}")
        raise