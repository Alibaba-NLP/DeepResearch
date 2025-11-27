"""
MLX React Agent Runner for Apple Silicon

This script runs DeepResearch using Apple's MLX framework instead of vLLM/CUDA.
Uses native MLX Python API with proper chat template handling.

Requirements:
    pip install mlx-lm python-dotenv requests json5 tqdm qwen-agent

Usage:
    python run_mlx_react.py --dataset eval_data/test.jsonl --output ./outputs
"""

import argparse
import json
import os
import signal
import sys
import time
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

# Load environment variables before other imports
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from tqdm import tqdm
import json5
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

from prompt import SYSTEM_PROMPT

# Disable tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Tool registry - import tools with fallbacks
TOOL_MAP: Dict[str, Any] = {}

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

# Graceful shutdown flag
shutdown_requested = False


def signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    global shutdown_requested
    if shutdown_requested:
        print("\nForce quit...")
        sys.exit(1)
    shutdown_requested = True
    print("\nShutdown requested. Finishing current task...")


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def today_date() -> str:
    return datetime.now().strftime("%Y-%m-%d")


class MLXReactAgent:
    """
    React agent using native MLX Python API for inference on Apple Silicon.
    
    Uses the model's built-in chat template for proper formatting.
    """
    
    def __init__(self, model_path: str, temperature: float = 0.85, 
                 top_p: float = 0.95, max_tokens: int = 8192):
        self.model_path = model_path
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        
        print(f"Loading model: {model_path}")
        self.model, self.tokenizer = load(model_path)
        print(f"Model loaded successfully (memory: {self._get_memory_usage():.1f} GB)")
    
    def _get_memory_usage(self) -> float:
        """Get current GPU memory usage in GB."""
        try:
            import mlx.core as mx
            # Use new API (mlx >= 0.24) or fall back to deprecated
            if hasattr(mx, 'get_active_memory'):
                return mx.get_active_memory() / (1024**3)
            return mx.metal.get_active_memory() / (1024**3)
        except Exception:
            return 0.0
    
    def build_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Build prompt using tokenizer's chat template.
        Falls back to manual Qwen format if template unavailable.
        """
        # Try using tokenizer's built-in chat template
        if hasattr(self.tokenizer, 'apply_chat_template'):
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                return prompt
            except Exception as e:
                print(f"Warning: apply_chat_template failed, using manual format: {e}")
        
        # Fallback: Manual Qwen/ChatML format
        prompt_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        prompt_parts.append("<|im_start|>assistant\n")
        return "\n".join(prompt_parts)
    
    def count_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Count tokens using the actual tokenizer."""
        prompt = self.build_prompt(messages)
        tokens = self.tokenizer.encode(prompt)
        return len(tokens)
    
    def generate_response(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None) -> str:
        """Generate response using native MLX API."""
        prompt = self.build_prompt(messages)
        tokens = max_tokens or self.max_tokens
        
        sampler = make_sampler(temp=self.temperature, top_p=self.top_p)
        
        response = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=tokens,
            sampler=sampler,
            verbose=False,
        )
        
        # Clean up response - remove trailing tokens
        if "<|im_end|>" in response:
            response = response.split("<|im_end|>")[0]
        if "<tool_response>" in response:
            response = response.split("<tool_response>")[0]
        
        return response.strip()
    
    def execute_tool(self, tool_name: str, tool_args: Dict[str, Any], timeout: int = 120) -> str:
        """Execute a tool with timeout protection."""
        if tool_name not in TOOL_MAP:
            return f"Error: Tool '{tool_name}' not found. Available: {list(TOOL_MAP.keys())}"
        
        # Copy args to avoid mutation
        args = dict(tool_args)
        result = ""
        error = None
        
        def run_tool():
            nonlocal result, error
            try:
                if "python" in tool_name.lower():
                    result = str(TOOL_MAP['PythonInterpreter'].call(args))
                elif tool_name == "parse_file":
                    import asyncio
                    params = {"files": args.get("files", [])}
                    r = asyncio.run(TOOL_MAP[tool_name].call(params, file_root_path="./eval_data/file_corpus"))
                    result = str(r) if not isinstance(r, str) else r
                else:
                    result = str(TOOL_MAP[tool_name].call(args))
            except Exception as e:
                error = str(e)
        
        thread = threading.Thread(target=run_tool)
        thread.start()
        thread.join(timeout=timeout)
        
        if thread.is_alive():
            return f"Error: Tool '{tool_name}' timed out after {timeout}s"
        
        if error:
            return f"Error executing tool '{tool_name}': {error}"
        
        return result
    
    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the react agent loop for a single question.
        
        Args:
            data: Dict with 'item' containing 'question' and optionally 'answer'
            
        Returns:
            Dict with question, answer, messages, prediction, and termination status
        """
        global shutdown_requested
        
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
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
        
        num_calls_remaining = MAX_LLM_CALL_PER_RUN
        round_num = 0
        max_context_tokens = 100 * 1024  # 100K tokens (conservative for 128K model)
        timeout_minutes = 120  # 2 hours
        consecutive_errors = 0
        last_tool_call = ""  # For loop detection
        
        while num_calls_remaining > 0:
            # Check for shutdown
            if shutdown_requested:
                return {
                    "question": question,
                    "answer": answer,
                    "messages": messages,
                    "prediction": "Interrupted by user",
                    "termination": "interrupted"
                }
            
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
            
            print(f"--- Round {round_num} (calls left: {num_calls_remaining}) ---")
            
            # Inject reminder at round 5 to encourage conclusion
            if round_num == 5:
                messages.append({
                    "role": "user",
                    "content": "REMINDER: You have made several searches. If you have enough information to answer the question, please provide your final answer now using <answer></answer> tags. Only continue searching if absolutely necessary."
                })
            
            # Generate response
            content = self.generate_response(messages)
            
            preview = content[:200] + "..." if len(content) > 200 else content
            print(f"Response: {preview}")
            
            messages.append({"role": "assistant", "content": content})
            
            # Check for tool calls
            if '<tool_call>' in content and '</tool_call>' in content:
                tool_call_str = content.split('<tool_call>')[1].split('</tool_call>')[0]
                
                # Loop detection: check if same tool call as last time
                if tool_call_str.strip() == last_tool_call:
                    print("Warning: Detected repeated tool call, forcing answer...")
                    messages.append({
                        "role": "user",
                        "content": "You are repeating the same action. Stop and provide your final answer NOW based on available information.\n<answer>your answer</answer>"
                    })
                    content = self.generate_response(messages, max_tokens=2048)
                    messages.append({"role": "assistant", "content": content})
                    if '<answer>' in content and '</answer>' in content:
                        prediction = content.split('<answer>')[1].split('</answer>')[0]
                    else:
                        prediction = content
                    return {
                        "question": question,
                        "answer": answer,
                        "messages": messages,
                        "prediction": prediction.strip(),
                        "termination": "loop_detected"
                    }
                
                last_tool_call = tool_call_str.strip()
                
                try:
                    # Handle Python interpreter specially
                    if "python" in tool_call_str.lower() and "<code>" in content:
                        code = content.split('<code>')[1].split('</code>')[0].strip()
                        result = self.execute_tool('PythonInterpreter', {"code": code})
                    else:
                        tool_call = json5.loads(tool_call_str.strip())
                        tool_name = tool_call.get('name', '')
                        tool_args = tool_call.get('arguments', {})
                        print(f"Tool: {tool_name} | Args: {json.dumps(tool_args)[:100]}...")
                        result = self.execute_tool(tool_name, tool_args)
                except json.JSONDecodeError as e:
                    result = f'Error: Invalid JSON in tool call. {e}'
                except Exception as e:
                    result = f'Error: Tool call failed. {e}'
                
                # Track consecutive errors
                if result.startswith('Error:'):
                    consecutive_errors += 1
                    if consecutive_errors >= 3:
                        print(f"Warning: {consecutive_errors} consecutive errors, forcing answer...")
                        messages.append({
                            "role": "user",
                            "content": f"Multiple tool errors occurred. Please provide your best answer based on the information you have gathered so far.\n<answer>your answer</answer>"
                        })
                        content = self.generate_response(messages, max_tokens=2048)
                        messages.append({"role": "assistant", "content": content})
                        if '<answer>' in content and '</answer>' in content:
                            prediction = content.split('<answer>')[1].split('</answer>')[0]
                        else:
                            prediction = content
                        return {
                            "question": question,
                            "answer": answer,
                            "messages": messages,
                            "prediction": prediction.strip(),
                            "termination": "consecutive_errors"
                        }
                else:
                    consecutive_errors = 0  # Reset on success
                
                result_preview = result[:200] + "..." if len(result) > 200 else result
                print(f"Result: {result_preview}")
                
                tool_response = f"<tool_response>\n{result}\n</tool_response>"
                messages.append({"role": "user", "content": tool_response})
            
            # Check for final answer
            if '<answer>' in content and '</answer>' in content:
                prediction = content.split('<answer>')[1].split('</answer>')[0]
                elapsed_mins = (time.time() - start_time) / 60
                print(f"Answer found in {elapsed_mins:.1f} minutes")
                return {
                    "question": question,
                    "answer": answer,
                    "messages": messages,
                    "prediction": prediction.strip(),
                    "termination": "answer"
                }
            
            # Check token limit
            token_count = self.count_tokens(messages)
            print(f"Tokens: {token_count:,}")
            
            if token_count > max_context_tokens:
                print(f"Token limit exceeded: {token_count:,} > {max_context_tokens:,}")
                
                # Force final answer
                messages.append({
                    "role": "user",
                    "content": "IMPORTANT: You have reached the maximum context length. "
                               "Stop making tool calls. Provide your final answer NOW based on all information above.\n"
                               "Format: <think>final reasoning</think>\n<answer>your answer</answer>"
                })
                
                content = self.generate_response(messages, max_tokens=2048)
                messages.append({"role": "assistant", "content": content})
                
                if '<answer>' in content and '</answer>' in content:
                    prediction = content.split('<answer>')[1].split('</answer>')[0]
                    termination = "token_limit_answer"
                else:
                    prediction = content
                    termination = "token_limit_no_answer"
                
                return {
                    "question": question,
                    "answer": answer,
                    "messages": messages,
                    "prediction": prediction.strip(),
                    "termination": termination
                }
        
        # Max calls reached - try to get final answer
        print("Max LLM calls reached, requesting final answer...")
        messages.append({
            "role": "user", 
            "content": "Maximum iterations reached. Provide your final answer NOW.\n"
                       "<answer>your answer</answer>"
        })
        
        content = self.generate_response(messages, max_tokens=2048)
        messages.append({"role": "assistant", "content": content})
        
        if '<answer>' in content and '</answer>' in content:
            prediction = content.split('<answer>')[1].split('</answer>')[0]
            termination = "max_calls_answer"
        else:
            prediction = content if content else "No answer found."
            termination = "max_calls_no_answer"
        
        return {
            "question": question,
            "answer": answer,
            "messages": messages,
            "prediction": prediction.strip(),
            "termination": termination
        }


def main():
    parser = argparse.ArgumentParser(
        description="Run DeepResearch with MLX on Apple Silicon",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model", type=str, default="abalogh/Tongyi-DeepResearch-30B-A3B-4bit",
                        help="Model path or HuggingFace model ID")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to input dataset (JSON or JSONL)")
    parser.add_argument("--output", type=str, default="./outputs",
                        help="Output directory")
    parser.add_argument("--temperature", type=float, default=0.85,
                        help="Sampling temperature (0.0-2.0)")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Top-p (nucleus) sampling (0.0-1.0)")
    parser.add_argument("--max_tokens", type=int, default=8192,
                        help="Maximum tokens per generation")
    parser.add_argument("--roll_out_count", type=int, default=1,
                        help="Number of rollouts per question")
    args = parser.parse_args()
    
    # Validate args
    if not 0.0 <= args.temperature <= 2.0:
        print("Warning: temperature should be between 0.0 and 2.0")
    if not 0.0 <= args.top_p <= 1.0:
        print("Warning: top_p should be between 0.0 and 1.0")
    
    # Setup output directory
    model_name = os.path.basename(args.model.rstrip('/'))
    model_dir = os.path.join(args.output, f"{model_name}_mlx")
    dataset_name = os.path.splitext(os.path.basename(args.dataset))[0]
    output_dir = os.path.join(model_dir, dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("DeepResearch MLX Inference (Apple Silicon)")
    print("=" * 60)
    print(f"Model:       {args.model}")
    print(f"Dataset:     {args.dataset}")
    print(f"Output:      {output_dir}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-P:       {args.top_p}")
    print(f"Max Tokens:  {args.max_tokens}")
    print(f"Rollouts:    {args.roll_out_count}")
    print("=" * 60)
    
    # Load dataset
    try:
        if args.dataset.endswith(".json"):
            with open(args.dataset, "r", encoding="utf-8") as f:
                items = json.load(f)
                if isinstance(items, dict):
                    items = [items]
        elif args.dataset.endswith(".jsonl"):
            with open(args.dataset, "r", encoding="utf-8") as f:
                items = [json.loads(line) for line in f if line.strip()]
        else:
            print("Error: Dataset must be .json or .jsonl")
            return 1
    except FileNotFoundError:
        print(f"Error: Dataset not found at {args.dataset}")
        return 1
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in dataset: {e}")
        return 1
    
    print(f"Loaded {len(items)} items from dataset")
    
    if not items:
        print("Error: No items in dataset")
        return 1
    
    # Initialize agent
    try:
        agent = MLXReactAgent(
            model_path=args.model,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return 1
    
    # Setup output files per rollout
    output_files = {
        i: os.path.join(output_dir, f"iter{i}.jsonl") 
        for i in range(1, args.roll_out_count + 1)
    }
    
    # Load already processed questions
    processed_per_rollout: Dict[int, set] = {}
    for rollout_idx in range(1, args.roll_out_count + 1):
        processed: set = set()
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
        if processed:
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
                except Exception:
                    continue
            
            if question and question not in processed:
                tasks.append({
                    "item": item.copy(),
                    "rollout_idx": rollout_idx,
                })
    
    print(f"Tasks to run: {len(tasks)}")
    
    if not tasks:
        print("All tasks already completed!")
        return 0
    
    # Run tasks
    write_lock = threading.Lock()
    completed = 0
    failed = 0
    
    for task in tqdm(tasks, desc="Processing", disable=shutdown_requested):
        if shutdown_requested:
            print(f"\nStopped early. Completed: {completed}, Failed: {failed}")
            break
        
        rollout_idx = task["rollout_idx"]
        output_file = output_files[rollout_idx]
        
        try:
            result = agent.run(task)
            result["rollout_idx"] = rollout_idx
            result["elapsed_time"] = time.time()
            
            with write_lock:
                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
            
            completed += 1
            
        except Exception as e:
            failed += 1
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
            
            error_result = {
                "question": task["item"].get("question", ""),
                "answer": task["item"].get("answer", ""),
                "rollout_idx": rollout_idx,
                "error": str(e),
                "messages": [],
                "prediction": "[Failed]"
            }
            with write_lock:
                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(error_result, ensure_ascii=False) + "\n")
    
    print("\n" + "=" * 60)
    print("Inference Complete")
    print("=" * 60)
    print(f"Completed: {completed}")
    print(f"Failed:    {failed}")
    print(f"Output:    {output_dir}")
    print("=" * 60)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
