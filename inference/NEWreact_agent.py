import json
from pyexpat.errors import messages
import json5
import os
from typing import Dict, Iterator, List, Literal, Optional, Tuple, Union
from qwen_agent.llm.schema import Message
from qwen_agent.utils.utils import build_text_completion_prompt
from openai import OpenAI, APIError, APIConnectionError, APITimeoutError
from transformers import AutoTokenizer 
from datetime import datetime
from qwen_agent.agents.fncall_agent import FnCallAgent
from qwen_agent.llm import BaseChatModel
from qwen_agent.llm.schema import ASSISTANT, DEFAULT_SYSTEM_MESSAGE, Message
from qwen_agent.settings import MAX_LLM_CALL_PER_RUN
from qwen_agent.tools import BaseTool
from qwen_agent.utils.utils import format_as_text_message, merge_generate_cfgs
from prompt import *
import time
import asyncio
import hashlib
from dotenv import load_dotenv

from tool_file import *
from tool_scholar import *
from tool_python import *
from tool_search import *
from tool_visit import *

from tool_scholarLOCAL import *
from tool_searchLOCAL import *
from tool_visitLOCAL import *
import re

OBS_START = '<tool_response>'
OBS_END = '\n</tool_response>'

MAX_LLM_CALL_PER_RUN = int(os.getenv('MAX_LLM_CALL_PER_RUN', 100))

TOOL_CLASS = [
    #FileParser(),
    Scholar(),
    Visit(),
    Search(),
    #PythonInterpreter(),
]
TOOL_MAP = {tool.name: tool for tool in TOOL_CLASS}

import random
import datetime


def today_date():
    return datetime.date.today().strftime("%Y-%m-%d")

class SummaryMemory:
    """
    Rolling, chunked memory the orchestrator can inject as a single system summary.
    - Keeps the most recent `max_chunks` summaries.
    - Provides a stable, structured text via as_text() for LLM consumption.
    """
    def __init__(self, max_chunks: int = 6):
        self.chunks: List[str] = []
        self.max_chunks = max_chunks

    def add(self, text: str):
        text = (text or "").strip()
        if not text:
            return
        self.chunks.append(text)
        if len(self.chunks) > self.max_chunks:
            # Drop the oldest summaries to bound memory
            self.chunks = self.chunks[-self.max_chunks:]

    def as_text(self) -> str:
        if not self.chunks:
            return "(empty)"
        # Prefix each chunk with a stable [S#] marker for LLM robustness
        return "\n".join(f"[S{i+1}] {c}" for i, c in enumerate(self.chunks))

    def clear(self):
        self.chunks.clear()
        
class MultiTurnReactAgent(FnCallAgent):
    def __init__(self,
                 function_list: Optional[List[Union[str, Dict, BaseTool]]] = None,
                 llm: Optional[Union[Dict, BaseChatModel]] = None,
                 **kwargs):

        self.llm_generate_cfg = llm["generate_cfg"]
        self.llm_local_path = llm["model"]
        
        # -----------------------------
        # NEW: Orchestrator summarizer
        # -----------------------------
        self.summary_memory = SummaryMemory(max_chunks=6)
        # Summarize every N rounds (after assistant + optional tool_response)
        self.summarize_every_n_rounds: int = int(os.getenv("ORCH_SUMMARY_EVERY_N_ROUNDS", "10"))
        # Start summarizing/pruning long before the hard cap (performance + safety)
        self.soft_token_cap: int = int(os.getenv("ORCH_SOFT_TOKEN_CAP", str(80_000)))
        # Large tool outputs (by char length) trigger immediate summarization
        self.tool_response_size_threshold: int = int(os.getenv("ORCH_TOOL_RESP_LEN_TRIGGER", "20000"))
        # For exact token counting with HF, cache the tokenizer (performance)
        self._tokenizer = None
        self.initial_user_question: Optional[str] = None
        self.raw_messages: List[Dict] = [] # copy of all messages w/o summaries

        # Track whether a visit actually occurred (may be useful downstream)
        self.did_visit: bool = False

    def sanity_check_output(self, content):
        return "<think>" in content and "</think>" in content
    
    def call_server(self,
                    msgs,
                    planning_port,
                    max_tries: int = 10,
                    override_stop: Optional[List[str]] = None):        # Prefer agent-specific env; fall back to generic if needed
        openai_api_key  = os.getenv("DR_API_KEY")  or os.getenv("API_KEY")
        openai_api_base = os.getenv("DR_API_BASE") or os.getenv("API_BASE") or "https://openrouter.ai/api/v1"
        resolved_model  = os.getenv("DR_MODEL_NAME") or os.getenv("MODEL_NAME") or "alibaba/tongyi-deepresearch-30b-a3b"

        # If self.model is "api" or empty, use the real model id from env
        use_model = resolved_model if str(getattr(self, "model", "")).strip().lower() in ("", "api") else self.model

        if not openai_api_key:
            # Fail fast with a clear message; caller handles the string appropriately.
            return "ERROR: Missing API key. Set DR_API_KEY or API_KEY."

        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
            timeout=600.0,
        )

        # Use default tool stop sequences unless explicitly overridden
        stop_sequences = ["\n<tool_response>", "<tool_response>"] if (override_stop is None) else override_stop

        base_sleep_time = 120
        last_exception_str = None
        for attempt in range(max_tries):
            try:
                print(f"--- Attempting to call the service, try {attempt + 1}/{max_tries} ---")
                chat_response = client.chat.completions.create(
                    model=use_model,  # <--- use the resolved model
                    messages=msgs,
                    stop=stop_sequences if stop_sequences else None,
                    temperature=self.llm_generate_cfg.get('temperature', 0.6),
                    top_p=self.llm_generate_cfg.get('top_p', 0.95),
                    logprobs=True,
                    max_tokens=10000,
                    presence_penalty=self.llm_generate_cfg.get('presence_penalty', 1.1)
                )
                content = (chat_response.choices[0].message.content or "").strip()

                # Optional: prepend reasoning if provider returns it
                try:
                    reasoning_text = getattr(chat_response.choices[0].message, "reasoning", None)
                    if isinstance(reasoning_text, str) and reasoning_text.strip():
                        content = "<think>\n" + reasoning_text.strip() + "\n</think>" + content
                except Exception:
                    pass

                if content:
                    print("--- Service call successful, received a valid response ---")
                    return content
                else:
                    print(f"Warning: Attempt {attempt + 1} received an empty response.")

            except (APIError, APIConnectionError, APITimeoutError) as e:
                print(f"Error: Attempt {attempt + 1} failed with an API or network error: {e}")
            except Exception as e:
                print(f"Error: Attempt {attempt + 1} failed with an unexpected error: {e}")

            if attempt < max_tries - 1:
                sleep_time = min(base_sleep_time * (2 ** attempt) + random.uniform(0, 1), 300)
                print(f"Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            else:
                print("Error: All retry attempts have been exhausted. The call has failed.")

        return f"vllm server error!!! ({last_exception_str or 'unknown error'})"



    def count_tokens(self, messages):
        # >>> minimal guard for API mode (no local HF tokenizer) <<<
        if str(getattr(self, "llm_local_path", "")).strip().lower() == "api":
            try:
                text = "\n".join(m.get("content", "") for m in messages if isinstance(m, dict))
            except Exception:
                text = str(messages)
            # cheap approx: ~1 token per 4 chars
            return max(1, len(text) // 4)

        # Cache tokenizer for performance
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.llm_local_path)

        
        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        tokens = tokenizer(full_prompt, return_tensors="pt")
        token_count = int(tokens["input_ids"].shape[-1])
        
        return token_count

    def _cleanup_after_run(self):
        """Reset per-question orchestrator state to avoid leakage across tasks."""
        try:
            if hasattr(self, "summary_memory") and self.summary_memory:
                self.summary_memory.clear()
        except Exception:
            pass
        self.initial_user_question = None
        self.raw_messages = []

    # ======================================================
    # NEW: Summarize current conversation for orchestration
    # ======================================================
    def summarize_messages(self, messages: List[Dict]) -> str:
        """
        Ask the model to compress the conversation so far into a structured, compact summary.
        - Uses a side-call (no tool stops) so it won't be truncated unexpectedly.
        - Clips to the last N messages to control cost.
        - Robust to unexpected outputs; returns plain text on error.
        """
        try:
            # Keep the prompt deterministic and tool-free
            summary_system = (
                "You compress context for downstream agents. Never call tools or return the XML tags used in the main loop. "
                "Use only the provided materials; do not browse or fetch external content. "
                "The raw system prompt and raw question are provided separately—do NOT restate or paraphrase either. "
                "Do not include hidden reasoning or chain-of-thought."
            )

            summary_instruction = (
                "Create a compact one-shot summary from ONLY the supplied context (messages, tool outputs, prior notes). "
                "Do NOT browse or invent facts.\n"
                "Use this schema:\n"
                "1) Task\n"
                "2) Key findings so far\n"
                "3) Tool evidence (short bullets; end each with a handle if provided; if none, write 'None')\n"
                "4) Decisions/assumptions (mark each; add Confidence High/Med/Low if present)\n"
                "5) Open questions / next actions\n"
                "Rules: Be concise, no repetition, no new facts, no tool calls. Prefer bullets; keep each bullet ≤ 1 line. "
                "If a required section would be empty, write 'None'. Output exactly the sections above—no preamble or extras. "
                "Do NOT use <tool_call>, <tool_response>, <answer>, or <think> tags."
            )
            # Clip from RAW history and SKIP messages[0] (system) and [1] (initial question)
            base = self.raw_messages if self.raw_messages else messages
            tail_k = 12  # last 12 messages are usually enough
            body = base[2:] if len(base) > 2 else []
            clipped = body[-tail_k:] if len(body) > tail_k else body[:]
            mini_msgs = [
                {"role": "system", "content": summary_system},
                {"role": "user", "content": summary_instruction},
                # Serialize the clipped messages for stable summarization. We avoid including binary/tool blobs.
                {"role": "user", "content": json.dumps(clipped, ensure_ascii=False)}
            ]

            # IMPORTANT: override_stop=[] disables the default tool-response stops for this side-call
            content = self.call_server(
                msgs=mini_msgs,
                planning_port=None,
                max_tries=3,
                override_stop=[]
            )

            # Defensive post-processing: strip our loop control tags if any leaked in
            text = (content or "").strip()
            for tag in ("<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>",
                        "<answer>", "</answer>", "<think>", "</think>"):
                text = text.replace(tag, "")
            text = text.strip()
            # Keep it non-empty
            return text or "(empty summarization)"

        except Exception as e:
            print(f"[Summary] summarize_messages error: {e}")
            return "(summarization failed)"
    
    # 
    def _run(self, data: str, model: str, **kwargs) -> List[List[Message]]:
        self.model=model
        try:
            question = data['item']['question']
        except: 
            raw_msg = data['item']['messages'][1]["content"] 
            question = raw_msg.split("User:")[1].strip() if "UserI needed a question id, w:" in raw_msg else raw_msg 
        
        self.external_question_id = str(data['item'].get('question_id', '') or '')

        start_time = time.time()
        planning_port = data['planning_port']
        answer = data['item']['answer']
        self.user_prompt = question
        system_prompt = SYSTEM_PROMPT
        cur_date = today_date()
        system_prompt = system_prompt + str(cur_date)
        self.initial_user_question = question
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]        
        self.raw_messages = list(messages)
        
        num_llm_calls_available = MAX_LLM_CALL_PER_RUN
        round = 0
        while num_llm_calls_available > 0:
            # Check whether time is reached
            if time.time() - start_time > 150 * 60:  # 150 minutes in seconds
                prediction = 'No answer found after 2h30mins'
                termination = 'No answer found after 2h30mins'
                result = {
                    "question": question,
                    "answer": answer,
                    "messages": messages,
                    "prediction": prediction,
                    "termination": termination,
                    "handoff_summary": self.summary_memory.as_text(),
                }
                self._cleanup_after_run()
                return result
            round += 1
            num_llm_calls_available -= 1
            content = self.call_server(messages, planning_port)
            print(f'Round {round}: {content}')
            if not isinstance(content, str) or not content:
                # Defensive: empty or non-string response
                content = "ERROR: Empty response from LLM."
            if '<tool_response>' in content:
                pos = content.find('<tool_response>')
                content = content[:pos]
            messages.append({"role": "assistant", "content": content.strip()})
            
            just_ran_tool = False
            last_tool_response_len = 0
            
            # ============================
            # Multi-<tool_call> integration
            # ============================
            if '<tool_call>' in content and '</tool_call>' in content:
                # find all tool_call payloads (multiline-safe)
                calls = re.findall(r"<tool_call>(.*?)</tool_call>", content, flags=re.DOTALL)
                aggregated_outputs: List[str] = []

                for raw_call in calls:
                    try:
                        is_python = ("python" in raw_call.lower())
                        has_code_tags = ("<code>" in raw_call and "</code>" in raw_call)

                        if is_python and has_code_tags:
                            # Python inline path
                            try:
                                code_raw = raw_call.split("<code>", 1)[1].split("</code>", 1)[0].strip()
                                result = TOOL_MAP['PythonInterpreter'].call(code_raw)
                            except Exception as e:
                                result = f"[Python Interpreter Error]: Formatting error. ({e})"
                        elif is_python and not has_code_tags:
                            # Preserve prior behavior: treat as formatting error if python is indicated but no code tags
                            result = "[Python Interpreter Error]: Formatting error."
                        else:
                            # JSON tool call path (normal)
                            call_obj = None
                            try:
                                call_obj = json5.loads(raw_call)
                            except Exception as e:
                                result = f'Error: Tool call is not a valid JSON. Tool call must contain a valid "name" and "arguments" field. ({e})'
                            if call_obj is not None:
                                tool_name = call_obj.get('name', '')
                                tool_args = call_obj.get('arguments', {})
                                try:
                                    result = self.custom_call_tool(tool_name, tool_args)
                                    # Only flip did_visit if we actually executed a 'visit' successfully
                                    if tool_name and isinstance(tool_name, str) and tool_name.lower() == "visit" and result is not None:
                                        self.did_visit = True
                                except Exception as tool_exec_err:
                                    result = f"Error: Tool call execution failed: {tool_exec_err}"

                    except Exception as e:
                        # Fallback safeguard per call
                        result = f"Error: Tool call execution failed: {e}"

                    # stringify tool output
                    tool_out_str = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
                    aggregated_outputs.append(tool_out_str)

                # Feed *all* tool outputs back at once (clear delimiter between items)
                tool_response_str = "<tool_response>\n" + "\n\n---\n\n".join(aggregated_outputs) + "\n</tool_response>"
                #messages.append({"role": "user", "content": tool_response_str})
                msg_tool = {"role": "user", "content": tool_response_str}
                messages.append(msg_tool)
                self.raw_messages.append(msg_tool)  # keep raw
                just_ran_tool = True
                last_tool_response_len = len(tool_response_str)
            # ============================
            # /Multi-<tool_call> integration
            # ============================
                
            if '<answer>' in content and '</answer>' in content:
                termination = 'answer'
                break
            if num_llm_calls_available <= 0 and '<answer>' not in content:
                #messages.append({"role": "assistant", "content": 'Sorry, the number of llm calls exceeds the limit.'})
                msg_budget = {"role": "assistant", "content": 'Sorry, the number of llm calls exceeds the limit.'}
                messages.append(msg_budget)
                self.raw_messages.append(msg_budget)  # keep raw                
            max_tokens = 110 * 1024
            token_count = self.count_tokens(messages)
            print(f"round: {round}, token count: {token_count}")

            # Trigger summarization:
            # - every N rounds
            # - or when crossing soft token cap
            # - or when the last tool response is very large (heuristic)
            should_summarize = False
            try:
                if (round % max(1, self.summarize_every_n_rounds) == 0):
                    print("Summarization triggered by round count.\n")
                    should_summarize = True
                if token_count > self.soft_token_cap:
                    print("Summarization triggered by soft token cap.\n")
                    should_summarize = True
                if just_ran_tool and last_tool_response_len >= self.tool_response_size_threshold:
                    print("Summarization triggered by large tool response.\n")
                    should_summarize = True
            except Exception:
                # Fallback if any unexpected type issues
                print("Summarization triggered by exception fallback.\n")
                should_summarize = True

            if should_summarize:
                try:
                    summary_text = self.summarize_messages(messages)
                    self.summary_memory.add(summary_text)
                    rolled = self.summary_memory.as_text()
                    print("[ORCH SUMMARY]\n{}\n[Length: {} chars]".format(rolled, len(rolled)))
                    print("[/ORCH SUMMARY]\n")

                    # Inject a single orchestrator summary as a high-priority system message
                    summary_msg = {
                        "role": "system",
                        "content": "<orchestrator_summary>\n" + rolled + "\n</orchestrator_summary>"
                    }

                    # REPLACEMENT STRATEGY (no pinned question):
                    # keep:
                    #  - messages[0] = system prompt
                    #  - messages[1] = initial user question
                    #  - latest summary as a system message
                    #  - optional short tail from RAW history (skip first two)
                    K = 3  # retain ~last K pairs as tail
                    head = [messages[0], messages[1]]
                    raw_body = self.raw_messages[2:] if len(self.raw_messages) > 2 else []
                    tail_len = min(len(raw_body), 2 * K)
                    tail = raw_body[-tail_len:] if tail_len > 0 else []
                    messages = head + [summary_msg] + tail

                    # Recount tokens post-prune to log effect (optional)
                    new_tc = self.count_tokens(messages)
                    print(f"[Summary] Injected summary. Token count reduced: {token_count} -> {new_tc}")

                except Exception as e:
                    print(f"[Summary] skipped due to error: {e}")
            # ------------------------------------------------------------
            # HARD CAP HANDLING: if we exceed hard cap, force a final answer
            # ------------------------------------------------------------
            token_count = self.count_tokens(messages)
            if token_count > max_tokens:
                print(f"Token quantity exceeds the limit: {token_count} > {max_tokens}")

                # Instruct the model to finalize now. We keep existing context and ask for a structured finalization.
                #messages.append({
                msg_finalize = {
                    "role": "assistant",
                    "content": "You have now reached the maximum context length you can handle. "
                               "You should stop making tool calls and, based on all the information above, "
                               "think again and provide what you consider the most likely answer in the following format:"
                               "<think>your final thinking</think>\n<answer>your answer</answer>"
                }#)
                messages.append(msg_finalize)
                self.raw_messages.append(msg_finalize)  # keep raw
                content = self.call_server(messages, planning_port)
                msg_assistant = {"role": "assistant", "content": content.strip()}
                messages.append(msg_assistant)
                self.raw_messages.append(msg_assistant)  # keep raw
                if '<answer>' in (content or "") and '</answer>' in (content or ""):
                    prediction = messages[-1]['content'].split('<answer>')[1].split('</answer>')[0]
                    termination = 'generate an answer as token limit reached'
                else:
                    prediction = messages[-1]['content']
                    termination = 'format error: generate an answer as token limit reached'
                result = {
                    "question": question,
                    "answer": answer,
                    "messages": messages,
                    "prediction": prediction,
                    "termination": termination,
                    "handoff_summary": self.summary_memory.as_text(),
                }
                self._cleanup_after_run()
                return result

        if '<answer>' in messages[-1]['content']:
            prediction = messages[-1]['content'].split('<answer>')[1].split('</answer>')[0]
            termination = 'answer'
        else:
            prediction = 'No answer found.'
            termination = 'answer not found'
            if num_llm_calls_available == 0:
                termination = 'exceed available llm calls'
        result = {
            "question": question,
            "answer": answer,
            "messages": messages,
            "prediction": prediction,
            "termination": termination,
            "handoff_summary": self.summary_memory.as_text(),
        }
        self._cleanup_after_run()
        return result

    def custom_call_tool(self, tool_name: str, tool_args: dict, **kwargs):
        if tool_name in TOOL_MAP:
            # make a copy so we don’t mutate a shared object
            tool_args = dict(tool_args or {})

            # Always inject question_id for visit & scholar if missing/falsey
            if tool_name.lower() in ("visit", "google_scholar"):
                qi = tool_args.get("question_id")
                if not qi:  # covers None, "", 0, etc.
                    qi = self.external_question_id or self.question_id
                    tool_args["question_id"] = str(qi)
                print(f"[agent] {tool_name} → question_id={tool_args['question_id']}")

            # (You don’t need 'params' = tool_args; that self-reference is messy. Remove it.)
            # tool_args["params"] = tool_args  # <- delete this line

            raw_result = TOOL_MAP[tool_name].call(tool_args, **kwargs)
            return raw_result
        else:
            return f"Error: Tool {tool_name} not found"