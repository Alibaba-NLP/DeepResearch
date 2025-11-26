"""
MLX React Agent Runner for Apple Silicon

This script runs DeepResearch using Apple's MLX framework instead of vLLM/CUDA.
Uses native MLX Python API with proper chat template handling.

Usage:
    python run_mlx_react.py --dataset eval_data/test.jsonl --output ./outputs
"""

import argparse
import json
import os
import time
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

# Load environment variables
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from tqdm import tqdm
import json5
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

from prompt import SYSTEM_PROMPT

# Tool registry - import tools with fallbacks for compatibility
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


def today_date() -> str:
    return datetime.now().strftime("%Y-%m-%d")


class MLXReactAgent:
    """
    React agent that uses native MLX Python API for inference on Apple Silicon.
    
    Uses the model's chat template directly for proper tool-calling format.
    """
    
    def __init__(self, model_path: str, temperature: float = 0.85, 
                 top_p: float = 0.95, max_tokens: int = 8192):
        self.model_path = model_path
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        
        print(f"Loading model: {model_path}")
        self.model, self.tokenizer = load(model_path)
        print("Model loaded successfully")
    
    def build_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Build prompt using the Qwen chat template format.
        Format: <|im_start|>role\ncontent<|im_end|>
        """
        prompt_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        
        # Add assistant start token for generation
        prompt_parts.append("<|im_start|>assistant\n")
        return "\n".join(prompt_parts)
    
    def generate_response(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None) -> str:
        """Generate response using native MLX API."""
        prompt = self.build_prompt(messages)
        tokens = max_tokens or self.max_tokens
        
        # Create sampler with temperature and top_p
        sampler = make_sampler(temp=self.temperature, top_p=self.top_p)
        
        # Generate with sampler
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
    
    def estimate_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Rough token estimation."""
        total_chars = sum(len(m.get("content", "")) for m in messages)
        return total_chars // 4
    
    def custom_call_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        """Execute a tool and return the result."""
        if tool_name not in TOOL_MAP:
            return f"Error: Tool {tool_name} not found. Available: {list(TOOL_MAP.keys())}"
        
        tool_args["params"] = tool_args
        
        if "python" in tool_name.lower():
            return str(TOOL_MAP['PythonInterpreter'].call(tool_args))
        
        if tool_name == "parse_file":
            import asyncio
            params = {"files": tool_args["files"]}
            result = asyncio.run(TOOL_MAP[tool_name].call(params, file_root_path="./eval_data/file_corpus"))
            return str(result) if not isinstance(result, str) else result
        
        return str(TOOL_MAP[tool_name].call(tool_args))
    
    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
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
        messages: List[Dict[str, str]] = [
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
            
            print(f"--- Round {round_num} ---")
            
            # Generate response
            content = self.generate_response(messages)
            
            preview = content[:200] + "..." if len(content) > 200 else content
            print(f"Response: {preview}")
            
            messages.append({"role": "assistant", "content": content})
            
            # Check for tool calls
            if '<tool_call>' in content and '</tool_call>' in content:
                tool_call_str = content.split('<tool_call>')[1].split('</tool_call>')[0]
                
                try:
                    if "python" in tool_call_str.lower() and "<code>" in content:
                        try:
                            code = content.split('<tool_call>')[1].split('</tool_call>')[0]
                            code = code.split('<code>')[1].split('</code>')[0].strip()
                            result = str(TOOL_MAP['PythonInterpreter'].call(code))
                        except Exception:
                            result = "[Python Interpreter Error]: Formatting error."
                    else:
                        tool_call = json5.loads(tool_call_str)
                        tool_name = tool_call.get('name', '')
                        tool_args = tool_call.get('arguments', {})
                        print(f"Tool call: {tool_name} with args: {tool_args}")
                        result = self.custom_call_tool(tool_name, tool_args)
                except Exception as e:
                    result = f'Error: Tool call is not valid JSON. Must contain "name" and "arguments" fields. Error: {e}'
                
                print(f"Tool result preview: {result[:200]}..." if len(result) > 200 else f"Tool result: {result}")
                
                tool_response = f"<tool_response>\n{result}\n</tool_response>"
                messages.append({"role": "user", "content": tool_response})
            
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
            print(f"Estimated tokens: {token_count}")
            
            if token_count > max_context_tokens:
                print(f"Token limit exceeded: {token_count} > {max_context_tokens}")
                
                # Force final answer
                messages.append({
                    "role": "user",
                    "content": "You have reached the maximum context length. Stop making tool calls and "
                               "provide your best answer based on all information above in this format:\n"
                               "<think>your final thinking</think>\n<answer>your answer</answer>"
                })
                
                content = self.generate_response(messages)
                messages.append({"role": "assistant", "content": content})
                
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
                messages.append({
                    "role": "user", 
                    "content": "Maximum LLM calls reached. Please provide your final answer now."
                })
        
        # No answer found
        last_content = messages[-1].get('content', '')
        if '<answer>' in last_content:
            prediction = last_content.split('<answer>')[1].split('</answer>')[0]
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
                        help="Model path or HuggingFace model ID")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to input dataset (JSON or JSONL)")
    parser.add_argument("--output", type=str, default="./outputs",
                        help="Output directory")
    parser.add_argument("--temperature", type=float, default=0.85,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Top-p sampling")
    parser.add_argument("--max_tokens", type=int, default=8192,
                        help="Maximum tokens per generation")
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
    print("DeepResearch MLX Inference (Native API)")
    print("=" * 50)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {output_dir}")
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
    agent = MLXReactAgent(
        model_path=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )
    
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
        return
    
    # Run tasks
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
            with write_locks[rollout_idx]:
                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(error_result, ensure_ascii=False) + "\n")
    
    print("\nInference complete!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
