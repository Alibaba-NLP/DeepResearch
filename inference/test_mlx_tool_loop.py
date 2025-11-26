#!/usr/bin/env python3
"""
Diagnostic test for MLX tool response injection.

This script tests the complete tool call loop:
1. Send a question to MLX
2. Model generates <tool_call>...</tool_call>
3. We parse and execute the tool
4. We inject <tool_response>...</tool_response>
5. Model continues with the tool response

Usage:
    python test_mlx_tool_loop.py
"""

import os
import sys
import json
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from openai import OpenAI
import json5

# Import tools
sys.path.insert(0, os.path.dirname(__file__))
from tool_search import Search
from prompt import SYSTEM_PROMPT

TOOL_MAP: Dict[str, Any] = {"search": Search()}


def today_date() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def parse_tool_call(content: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Extract tool name and arguments from model output."""
    if "<tool_call>" not in content or "</tool_call>" not in content:
        return None, None
    
    tool_call_str = content.split("<tool_call>")[1].split("</tool_call>")[0].strip()
    
    try:
        tool_call = json5.loads(tool_call_str)
        name = tool_call.get("name") if isinstance(tool_call, dict) else None
        args = tool_call.get("arguments", {}) if isinstance(tool_call, dict) else {}
        return name, args
    except Exception as e:
        print(f"Failed to parse tool call JSON: {e}")
        print(f"Raw tool call: {tool_call_str}")
        return None, None


def execute_tool(name: str, args: Dict[str, Any]) -> str:
    """Execute a tool and return the result."""
    if name not in TOOL_MAP:
        return f"Error: Tool '{name}' not found. Available: {list(TOOL_MAP.keys())}"
    
    args["params"] = args
    return TOOL_MAP[name].call(args)


def test_tool_loop():
    """Test the complete tool call loop."""
    print("=" * 60)
    print("MLX Tool Response Injection Diagnostic Test")
    print("=" * 60)
    
    # Connect to MLX server
    client = OpenAI(
        api_key="mlx-local",
        base_url="http://127.0.0.1:8080/v1",
        timeout=300.0,
    )
    
    # Verify connection
    try:
        models = client.models.list()
        print(f"Connected to MLX server. Model: {models.data[0].id}")
    except Exception as e:
        print(f"ERROR: Cannot connect to MLX server: {e}")
        print("Make sure the MLX server is running:")
        print("  mlx_lm.server --model abalogh/Tongyi-DeepResearch-30B-A3B-4bit --port 8080")
        return
    
    # Build messages
    system_prompt = SYSTEM_PROMPT + str(today_date())
    question = "What are the latest developments in quantum computing in 2024?"
    
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]
    
    print(f"\nQuestion: {question}")
    print("-" * 60)
    
    max_rounds = 5
    for round_num in range(1, max_rounds + 1):
        print(f"\n--- Round {round_num} ---")
        print(f"Messages count: {len(messages)}")
        print(f"Last message role: {messages[-1]['role']}")
        
        # Call MLX
        response = client.chat.completions.create(
            model=models.data[0].id,
            messages=messages,  # type: ignore
            stop=["\n<tool_response>", "<tool_response>"],
            temperature=0.85,
            top_p=0.95,
            max_tokens=8192,
        )
        
        raw_content = response.choices[0].message.content
        finish_reason = response.choices[0].finish_reason
        content = raw_content.strip() if raw_content else ""
        
        print(f"\nFinish reason: {finish_reason}")
        
        # Clean up if <tool_response> leaked
        if "<tool_response>" in content:
            content = content[:content.find("<tool_response>")]
        
        print(f"\nModel output ({len(content)} chars):")
        print("-" * 40)
        print(content[:1000] + "..." if len(content) > 1000 else content)
        print("-" * 40)
        
        # Add assistant message
        messages.append({"role": "assistant", "content": content})
        
        # Check for final answer
        if "<answer>" in content and "</answer>" in content:
            answer = content.split("<answer>")[1].split("</answer>")[0]
            print(f"\nFINAL ANSWER: {answer}")
            print(f"Total rounds: {round_num}")
            break
        
        # Check for tool call
        tool_name, tool_args = parse_tool_call(content)
        
        if tool_name and tool_args is not None:
            print(f"\nTool call detected: {tool_name}")
            print(f"Arguments: {json.dumps(tool_args, indent=2)}")
            
            # Execute tool
            result = execute_tool(tool_name, tool_args)
            print(f"\nTool result ({len(result)} chars):")
            print(result[:500] + "..." if len(result) > 500 else result)
            
            # Inject tool response
            tool_response = f"<tool_response>\n{result}\n</tool_response>"
            messages.append({"role": "user", "content": tool_response})
            print(f"\nInjected tool_response as user message")
        else:
            print("\nNo tool call detected in output")
            if round_num < max_rounds:
                print("Model may be stuck - no tool call and no answer")
    
    # Print final message history
    print("\n" + "=" * 60)
    print("FULL MESSAGE HISTORY")
    print("=" * 60)
    for i, msg in enumerate(messages):
        role = msg["role"]
        content = msg["content"]
        preview = content[:200] + "..." if len(content) > 200 else content
        print(f"\n[{i}] {role.upper()}:")
        print(preview)


if __name__ == "__main__":
    test_tool_loop()
