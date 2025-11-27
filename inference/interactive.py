#!/usr/bin/env python3
"""
Interactive CLI for DeepResearch on Apple Silicon (MLX)

Usage:
    python interactive.py [--model MODEL_PATH]
    
Example:
    python interactive.py
    python interactive.py --model abalogh/Tongyi-DeepResearch-30B-A3B-4bit
"""

import argparse
import json
import os
import sys
import time

# Load environment variables first
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# Disable tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Optional: rich for better formatting
try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None


def print_header():
    """Print welcome header."""
    header = """
╔══════════════════════════════════════════════════════════════╗
║           DeepResearch - Interactive Mode (MLX)              ║
║                  Apple Silicon Optimized                     ║
╚══════════════════════════════════════════════════════════════╝
"""
    if RICH_AVAILABLE:
        console.print(header, style="bold blue")
    else:
        print(header)


def print_help():
    """Print help information."""
    help_text = """
Commands:
  /help     - Show this help message
  /quit     - Exit the program (or Ctrl+C)
  /clear    - Clear conversation history (start fresh)
  /status   - Show model and memory status
  
Just type your research question to begin!

Examples:
  > What is the current population of Tokyo?
  > Who won the 2024 Nobel Prize in Physics?
  > Explain the mechanism of CRISPR-Cas9 gene editing
"""
    if RICH_AVAILABLE:
        console.print(Panel(help_text, title="Help", border_style="green"))
    else:
        print(help_text)


def format_answer(answer: str):
    """Format the answer for display."""
    if RICH_AVAILABLE:
        console.print("\n")
        console.print(Panel(Markdown(answer), title="[bold green]Answer[/]", border_style="green"))
    else:
        print("\n" + "=" * 60)
        print("ANSWER:")
        print("=" * 60)
        print(answer)
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Interactive DeepResearch CLI")
    parser.add_argument("--model", type=str, 
                        default="abalogh/Tongyi-DeepResearch-30B-A3B-4bit",
                        help="Model path or HuggingFace ID")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--max_tokens", type=int, default=4096,
                        help="Max tokens per generation")
    parser.add_argument("--max_rounds", type=int, default=15,
                        help="Max research rounds per question")
    args = parser.parse_args()

    print_header()
    
    # Set max rounds via environment
    os.environ['MAX_LLM_CALL_PER_RUN'] = str(args.max_rounds)
    
    # Import agent after setting environment
    print("Loading model (this may take a minute)...")
    
    try:
        from run_mlx_react import MLXReactAgent, TOOL_MAP
    except ImportError as e:
        print(f"Error importing agent: {e}")
        print("Make sure you're running from the inference directory.")
        return 1
    
    if RICH_AVAILABLE:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            progress.add_task("Loading MLX model...", total=None)
            agent = MLXReactAgent(
                model_path=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens
            )
    else:
        agent = MLXReactAgent(
            model_path=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )
    
    print(f"\nTools available: {list(TOOL_MAP.keys())}")
    print(f"Max rounds per question: {args.max_rounds}")
    print_help()
    
    while True:
        try:
            # Get user input
            if RICH_AVAILABLE:
                query = console.input("\n[bold cyan]Research Query>[/] ").strip()
            else:
                query = input("\nResearch Query> ").strip()
            
            # Handle commands
            if not query:
                continue
            
            if query.lower() in ('/quit', '/exit', '/q'):
                print("Goodbye!")
                break
            
            if query.lower() == '/help':
                print_help()
                continue
            
            if query.lower() == '/clear':
                print("Ready for a new question.")
                continue
            
            if query.lower() == '/status':
                try:
                    import mlx.core as mx
                    mem_gb = mx.metal.get_active_memory() / (1024**3)
                    print(f"Model: {args.model}")
                    print(f"GPU Memory: {mem_gb:.1f} GB")
                except Exception:
                    print(f"Model: {args.model}")
                continue
            
            if query.startswith('/'):
                print(f"Unknown command: {query}. Type /help for available commands.")
                continue
            
            # Run research
            print("\nResearching...\n")
            start = time.time()
            
            data = {'item': {'question': query, 'answer': ''}}
            result = agent.run(data)
            
            elapsed = time.time() - start
            
            # Display result
            prediction = result.get('prediction', 'No answer found.')
            termination = result.get('termination', 'unknown')
            num_rounds = len([m for m in result.get('messages', []) if m.get('role') == 'assistant'])
            
            format_answer(prediction)
            
            if RICH_AVAILABLE:
                console.print(f"[dim]Completed in {elapsed:.1f}s | {num_rounds} rounds | Termination: {termination}[/]")
            else:
                print(f"\nCompleted in {elapsed:.1f}s | {num_rounds} rounds | Termination: {termination}")
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Type /quit to exit or continue with a new question.")
            continue
        except EOFError:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
