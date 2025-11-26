#!/usr/bin/env python3
"""
Quick test script to verify MLX server connection and basic inference.
Run this after starting the MLX server to verify everything works.

Usage:
    # Terminal 1: Start MLX server
    mlx_lm.server --model abalogh/Tongyi-DeepResearch-30B-A3B-4bit --port 8080
    
    # Terminal 2: Run this test
    python test_mlx_connection.py
"""

import sys
from openai import OpenAI

MLX_HOST = "127.0.0.1"
MLX_PORT = 8080


def test_connection():
    """Test basic connection to MLX server."""
    print(f"Testing connection to MLX server at {MLX_HOST}:{MLX_PORT}...")
    
    client = OpenAI(
        api_key="mlx-local",
        base_url=f"http://{MLX_HOST}:{MLX_PORT}/v1",
        timeout=60.0,
    )
    
    # Test 1: List models
    print("\n1. Listing available models...")
    try:
        models = client.models.list()
        available = [m.id for m in models.data]
        print(f"   Available models: {available}")
    except Exception as e:
        print(f"   FAILED: {e}")
        return False
    
    # Test 2: Simple completion
    print("\n2. Testing simple completion...")
    try:
        response = client.chat.completions.create(
            model=available[0] if available else "default",
            messages=[
                {"role": "user", "content": "What is 2+2? Answer with just the number."}
            ],
            max_tokens=10,
            temperature=0.1,
        )
        answer = response.choices[0].message.content
        print(f"   Response: {answer}")
    except Exception as e:
        print(f"   FAILED: {e}")
        return False
    
    # Test 3: Test with system prompt (like DeepResearch uses)
    print("\n3. Testing with system prompt...")
    try:
        response = client.chat.completions.create(
            model=available[0] if available else "default",
            messages=[
                {"role": "system", "content": "You are a helpful research assistant. Think step by step."},
                {"role": "user", "content": "What is the capital of Japan?"}
            ],
            max_tokens=100,
            temperature=0.7,
        )
        answer = response.choices[0].message.content or ""
        print(f"   Response: {answer[:200]}..." if len(answer) > 200 else f"   Response: {answer}")
    except Exception as e:
        print(f"   FAILED: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("All tests passed! MLX server is working correctly.")
    print("You can now run: bash inference/run_mlx_infer.sh")
    print("=" * 50)
    return True


if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1)
