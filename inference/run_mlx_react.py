"""
MLX React Agent Runner for Apple Silicon

This script runs DeepResearch using Apple's MLX framework instead of vLLM/CUDA.
It connects to the MLX-lm server which provides an OpenAI-compatible API.

Usage:
    python run_mlx_react.py --dataset eval_data/test.jsonl --output ./outputs
"""

import argparse
import json
import os
import time
import threading
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from tqdm import tqdm

import json5
from openai import OpenAI, APIError, APIConnectionError, APITimeoutError

from prompt import SYSTEM_PROMPT

# Tool registry - import tools with fallbacks for compatibility
TOOL_MAP = {}

try:
    from tool_search import Search
    TOOL_MAP['search'] = Search()
except ImportError as e:
    print(f"Warning: Could not import Search tool: {e}")

try:
    from tool_visit import Visit
    TOOL_MAP['visit'] = Visit()
except ImportError as e:
    print(f"Warning: Could not import Visit tool: {e}")

try:
    from tool_scholar import Scholar
    TOOL_MAP['google_scholar'] = Scholar()
except ImportError as e:
    print(f"Warning: Could not import Scholar tool: {e}")

try:
    from tool_file import FileParser
    TOOL_MAP['parse_file'] = FileParser()
except ImportError as e:
    print(f"Warning: Could not import FileParser tool: {e}")

try:
    from tool_python import PythonInterpreter
    TOOL_MAP['PythonInterpreter'] = PythonInterpreter()
except (ImportError, Exception) as e:
    print(f"Warning: Could not import PythonInterpreter tool: {e}")

print(f"Loaded tools: {list(TOOL_MAP.keys())}")

MAX_LLM_CALL_PER_RUN = int(os.getenv('MAX_LLM_CALL_PER_RUN', 100))


def today_date():
    return datetime.now().strftime("%Y-%m-%d")


class MLXReactAgent:
    """
    React agent that uses MLX-lm server for inference on Apple Silicon.
    
    The MLX server provides an OpenAI-compatible API at /v1/chat/completions,
    making it a drop-in replacement for vLLM/sglang servers.
    """
    
    def __init__(self, model: str, mlx_host: str = "127.0.0.1", mlx_port: int = 8080,
                 temperature: float = 0.85, top_p: float = 0.95, 
                 presence_penalty: float = 1.1, max_tokens: int = 10000):
        self.model = model
        self.mlx_host = mlx_host
        self.mlx_port = mlx_port
        self.temperature = temperature
        self.top_p = top_p
        self.presence_penalty = presence_penalty
        self.max_tokens = max_tokens
        
        # Create OpenAI client pointing to MLX server
        self.client = OpenAI(
            api_key="mlx-local",  # MLX server doesn't require auth
            base_url=f"http://{mlx_host}:{mlx_port}/v1",
            timeout=600.0,
        )
        
        # Verify connection
        self._verify_connection()
    
    def _verify_connection(self):
        """Verify that the MLX server is running and accessible."""
        try:
            models = self.client.models.list()
            available = [m.id for m in models.data]
            print(f"MLX server connected. Available models: {available}")
        except Exception as e:
            raise ConnectionError(f"Cannot connect to MLX server at {self.mlx_host}:{self.mlx_port}: {e}")
    
    def call_server(self, messages: list, max_tries: int = 10) -> str:
        """Call the MLX server with exponential backoff retry."""
        base_sleep = 1
        
        for attempt in range(max_tries):
            try:
                print(f"--- MLX call attempt {attempt + 1}/{max_tries} ---")
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    stop=["\n<tool_response>", "<tool_response>"],
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens,
                    # Note: MLX-lm may not support all parameters
                    # presence_penalty=self.presence_penalty,
                )
                
                content = response.choices[0].message.content
                
                if content and content.strip():
                    print("--- MLX call successful ---")
                    return content.strip()
                
                print(f"Warning: Attempt {attempt + 1} received empty response")
                
            except (APIError, APIConnectionError, APITimeoutError) as e:
                print(f"API error on attempt {attempt + 1}: {e}")
            except Exception as e:
                print(f"Unexpected error on attempt {attempt + 1}: {e}")
            
            if attempt < max_tries - 1:
                sleep_time = min(base_sleep * (2 ** attempt) + random.uniform(0, 1), 30)
                print(f"Retrying in {sleep_time:.2f}s...")
                time.sleep(sleep_time)
        
        return "MLX server error - all retries exhausted"
    
    def estimate_tokens(self, messages: list) -> int:
        """
        Rough token estimation without loading a full tokenizer.
        MLX models typically use ~4 chars per token for English text.
        """
        total_chars = sum(len(m.get("content", "")) for m in messages)
        return total_chars // 4
    
    def custom_call_tool(self, tool_name: str, tool_args: dict) -> str:
        """Execute a tool and return the result."""
        if tool_name not in TOOL_MAP:
            return f"Error: Tool {tool_name} not found"
        
        tool_args["params"] = tool_args
        
        if "python" in tool_name.lower():
            return TOOL_MAP['PythonInterpreter'].call(tool_args)
        
        if tool_name == "parse_file":
            import asyncio
            params = {"files": tool_args["files"]}
            result = asyncio.run(TOOL_MAP[tool_name].call(params, file_root_path="./eval_data/file_corpus"))
            return str(result) if not isinstance(result, str) else result
        
        return TOOL_MAP[tool_name].call(tool_args)
    
    def run(self, data: dict) -> dict:
        """
        Run the react agent loop for a single question.
        
        Args:
            data: Dict with 'item' containing 'question' and optionally 'answer'
            
        Returns:
            Dict with question, answer, messages, prediction, and termination status
        """
        # Extract question
        item = data['item']
        question = item.get('question', '')
        if not question:
            try:
                raw_msg = item['messages'][1]["content"]
                question = raw_msg.split("User:")[1].strip() if "User:" in raw_msg else raw_msg
            except Exception as e:
                print(f"Failed to extract question: {e}")
                return {"question": "", "error": "Could not extract question"}
        
        answer = item.get('answer', '')
        start_time = time.time()
        
        # Build initial messages
        system_prompt = SYSTEM_PROMPT + str(today_date())
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
        
        num_calls_remaining = MAX_LLM_CALL_PER_RUN
        round_num = 0
        max_context_tokens = 110 * 1024  # ~110K tokens max
        timeout_minutes = 150  # 2.5 hours
        
        while num_calls_remaining > 0:
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > timeout_minutes * 60:
                return {
                    "question": question,
                    "answer": answer,
                    "messages": messages,
                    "prediction": "No answer found after timeout",
                    "termination": "timeout"
                }
            
            round_num += 1
            num_calls_remaining -= 1
            
            # Call MLX server
            content = self.call_server(messages)
            print(f"Round {round_num}: {content[:200]}..." if len(content) > 200 else f"Round {round_num}: {content}")
            
            # Clean up response
            if '<tool_response>' in content:
                content = content[:content.find('<tool_response>')]
            
            messages.append({"role": "assistant", "content": content.strip()})
            
            # Check for tool calls
            if '<tool_call>' in content and '</tool_call>' in content:
                tool_call_str = content.split('<tool_call>')[1].split('</tool_call>')[0]
                
                try:
                    if "python" in tool_call_str.lower():
                        try:
                            code = content.split('<tool_call>')[1].split('</tool_call>')[0]
                            code = code.split('<code>')[1].split('</code>')[0].strip()
                            result = TOOL_MAP['PythonInterpreter'].call(code)
                        except:
                            result = "[Python Interpreter Error]: Formatting error."
                    else:
                        tool_call = json5.loads(tool_call_str)
                        tool_name = tool_call.get('name', '')
                        tool_args = tool_call.get('arguments', {})
                        result = self.custom_call_tool(tool_name, tool_args)
                except Exception as e:
                    result = f'Error: Tool call is not valid JSON. Must contain "name" and "arguments" fields. Error: {e}'
                
                result = f"<tool_response>\n{result}\n</tool_response>"
                messages.append({"role": "user", "content": result})
            
            # Check for final answer
            if '<answer>' in content and '</answer>' in content:
                prediction = content.split('<answer>')[1].split('</answer>')[0]
                return {
                    "question": question,
                    "answer": answer,
                    "messages": messages,
                    "prediction": prediction,
                    "termination": "answer"
                }
            
            # Check token limit
            token_count = self.estimate_tokens(messages)
            print(f"Round {round_num}, estimated tokens: {token_count}")
            
            if token_count > max_context_tokens:
                print(f"Token limit exceeded: {token_count} > {max_context_tokens}")
                
                # Force final answer
                messages[-1]['content'] = (
                    "You have reached the maximum context length. Stop making tool calls and "
                    "provide your best answer based on all information above in this format:\n"
                    "<think>your final thinking</think>\n<answer>your answer</answer>"
                )
                
                content = self.call_server(messages)
                messages.append({"role": "assistant", "content": content.strip()})
                
                if '<answer>' in content and '</answer>' in content:
                    prediction = content.split('<answer>')[1].split('</answer>')[0]
                    termination = "token_limit_answer"
                else:
                    prediction = content
                    termination = "token_limit_format_error"
                
                return {
                    "question": question,
                    "answer": answer,
                    "messages": messages,
                    "prediction": prediction,
                    "termination": termination
                }
            
            if num_calls_remaining <= 0:
                messages[-1]['content'] = "Maximum LLM calls reached."
        
        # No answer found
        if '<answer>' in messages[-1]['content']:
            prediction = messages[-1]['content'].split('<answer>')[1].split('</answer>')[0]
            termination = "answer"
        else:
            prediction = "No answer found."
            termination = "calls_exhausted"
        
        return {
            "question": question,
            "answer": answer,
            "messages": messages,
            "prediction": prediction,
            "termination": termination
        }


def main():
    parser = argparse.ArgumentParser(description="Run DeepResearch with MLX on Apple Silicon")
    parser.add_argument("--model", type=str, default="abalogh/Tongyi-DeepResearch-30B-A3B-4bit",
                        help="Model name (should match MLX server)")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to input dataset (JSON or JSONL)")
    parser.add_argument("--output", type=str, default="./outputs",
                        help="Output directory")
    parser.add_argument("--mlx_host", type=str, default="127.0.0.1",
                        help="MLX server host")
    parser.add_argument("--mlx_port", type=int, default=8080,
                        help="MLX server port")
    parser.add_argument("--temperature", type=float, default=0.85,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Top-p sampling")
    parser.add_argument("--presence_penalty", type=float, default=1.1,
                        help="Presence penalty")
    parser.add_argument("--max_workers", type=int, default=1,
                        help="Number of parallel workers (keep at 1 for MLX)")
    parser.add_argument("--roll_out_count", type=int, default=1,
                        help="Number of rollouts per question")
    args = parser.parse_args()
    
    # Setup output directory
    model_name = os.path.basename(args.model.rstrip('/'))
    model_dir = os.path.join(args.output, f"{model_name}_mlx")
    dataset_name = os.path.splitext(os.path.basename(args.dataset))[0]
    output_dir = os.path.join(model_dir, dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 50)
    print("DeepResearch MLX Inference")
    print("=" * 50)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {output_dir}")
    print(f"MLX Server: http://{args.mlx_host}:{args.mlx_port}")
    print(f"Temperature: {args.temperature}")
    print(f"Rollouts: {args.roll_out_count}")
    print("=" * 50)
    
    # Load dataset
    try:
        if args.dataset.endswith(".json"):
            with open(args.dataset, "r", encoding="utf-8") as f:
                items = json.load(f)
        elif args.dataset.endswith(".jsonl"):
            with open(args.dataset, "r", encoding="utf-8") as f:
                items = [json.loads(line) for line in f]
        else:
            raise ValueError("Dataset must be .json or .jsonl")
    except FileNotFoundError:
        print(f"Error: Dataset not found at {args.dataset}")
        return
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    print(f"Loaded {len(items)} items from dataset")
    
    # Initialize agent
    try:
        agent = MLXReactAgent(
            model=args.model,
            mlx_host=args.mlx_host,
            mlx_port=args.mlx_port,
            temperature=args.temperature,
            top_p=args.top_p,
            presence_penalty=args.presence_penalty,
        )
    except ConnectionError as e:
        print(f"Error: {e}")
        print("Make sure the MLX server is running:")
        print(f"  mlx_lm.server --model {args.model} --port {args.mlx_port}")
        return
    
    # Setup output files per rollout
    output_files = {
        i: os.path.join(output_dir, f"iter{i}.jsonl") 
        for i in range(1, args.roll_out_count + 1)
    }
    
    # Load already processed questions
    processed_per_rollout = {}
    for rollout_idx in range(1, args.roll_out_count + 1):
        processed = set()
        output_file = output_files[rollout_idx]
        if os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if "question" in data and "error" not in data:
                            processed.add(data["question"].strip())
                    except json.JSONDecodeError:
                        pass
        processed_per_rollout[rollout_idx] = processed
        print(f"Rollout {rollout_idx}: {len(processed)} already processed")
    
    # Build task list
    tasks = []
    for rollout_idx in range(1, args.roll_out_count + 1):
        processed = processed_per_rollout[rollout_idx]
        for item in items:
            question = item.get("question", "").strip()
            if not question:
                try:
                    user_msg = item["messages"][1]["content"]
                    question = user_msg.split("User:")[1].strip() if "User:" in user_msg else user_msg
                    item["question"] = question
                except:
                    continue
            
            if question and question not in processed:
                tasks.append({
                    "item": item.copy(),
                    "rollout_idx": rollout_idx,
                })
    
    print(f"Tasks to run: {len(tasks)}")
    
    if not tasks:
        print("All tasks already completed!")
        return
    
    # Run tasks
    # Note: MLX is single-threaded on GPU, so max_workers=1 is recommended
    write_locks = {i: threading.Lock() for i in range(1, args.roll_out_count + 1)}
    
    for task in tqdm(tasks, desc="Processing"):
        rollout_idx = task["rollout_idx"]
        output_file = output_files[rollout_idx]
        
        try:
            result = agent.run(task)
            result["rollout_idx"] = rollout_idx
            
            with write_locks[rollout_idx]:
                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    
        except Exception as e:
            print(f"Error processing task: {e}")
            error_result = {
                "question": task["item"].get("question", ""),
                "answer": task["item"].get("answer", ""),
                "rollout_idx": rollout_idx,
                "error": str(e),
                "messages": [],
                "prediction": "[Failed]"
            }
            with write_locks[rollout_idx]:
                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(error_result, ensure_ascii=False) + "\n")
    
    print("\nInference complete!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
