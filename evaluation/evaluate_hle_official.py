from openai import OpenAI
import os
from pydantic import BaseModel
from typing import Literal, Optional
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
import re
from collections import Counter, defaultdict
from transformers import AutoTokenizer
import argparse



# ---------------- Verbose logging helpers ----------------
VERBOSE = False
_log_lock = threading.Lock()
def vlog(msg: str, pad: int = 2, border: bool = False, width: int = 80):
    """
    Verbose logger.
    - pad:    how many blank lines to add before and after (default: 2)
    - border: if True, wraps the message with a separator line
    - width:  separator width when border=True
    """
    if not VERBOSE:
        return
    with _log_lock:
        text = str(msg)
        if border:
            sep = "=" * width
            out = ("\n" * pad) + f"{sep}\n{text}\n{sep}" + ("\n" * pad)
        else:
            out = ("\n" * pad) + text + ("\n" * pad)
        try:
            tqdm.write(out)
        except Exception:
            print(out, flush=True)

def _mask_key(k: str, head: int = 6, tail: int = 4) -> str:
    if not k:
        return "<empty>"
    if len(k) <= head + tail:
        return k[0] + "*" * (len(k) - 2) + k[-1]
    return f"{k[:head]}...{k[-tail:]}"

# --------------- Robust env handling ---------------------
def _clean(s: Optional[str]) -> str:
    return (s or "").replace("\ufeff", "").replace("\r", "").strip()

def _first_env(*names: str, default: str = "") -> str:
    for n in names:
        v = os.getenv(n)
        if v and _clean(v):
            return _clean(v)
    return _clean(default)

# Use a valid model id for the judge:
JUDGE_MODEL = _first_env("JUDGE_MODEL", default="gpt-5-nano")
MAX_WORKERS = 20

API_KEY  = _first_env("API_KEY", "OPENAI_API_KEY")
BASE_URL = _first_env("API_BASE", "BASE_URL", "OPENAI_BASE_URL", default="https://api.openai.com/v1")

def _assert_api_ready():
    if not API_KEY:
        raise RuntimeError("No API key found. Set API_KEY or OPENAI_API_KEY (and ensure no CRLF).")
    if not BASE_URL:
        raise RuntimeError("No base URL found. Set API_BASE or BASE_URL or OPENAI_BASE_URL.")

def _preflight():
    """Optional: quick connectivity test."""
    _assert_api_ready()
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    try:
        class _R(BaseModel):
            extracted_final_answer: str
            reasoning: str
            correct: Literal["yes", "no"]
            confidence: int
            strict: Optional[bool] = None

        r = client.beta.chat.completions.parse(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": "[question]: 2+2\n[response]: 4\n[correct_answer]: 4"}],
            response_format=_R,
            max_completion_tokens=64,
            timeout=20
        )
        return True, f"preflight ok: model={JUDGE_MODEL}, correct={r.choices[0].message.parsed.correct}"
    except Exception as e:
        return False, f"preflight failed: {e}"

def load_jsonl(fp):
    with open(fp, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

def write_jsonl(data, fp):
    with open(fp, 'w', encoding='utf-8') as f:
        f.write('\n'.join(json.dumps(line, ensure_ascii=False) for line in data) + '\n')

thread_local = threading.local()

def get_client():
    if not hasattr(thread_local, 'client'):
        _assert_api_ready()
        vlog(f"[cfg] BASE_URL={BASE_URL}")
        vlog(f"[cfg] JUDGE_MODEL={JUDGE_MODEL}")
        vlog(f"[cfg] API_KEY={_mask_key(API_KEY)}")
        thread_local.client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    return thread_local.client

JUDGE_PROMPT = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0|\\%| and 100|\\%| from [response]. Put 100 if there is no confidence score available."""

class ExtractedAnswer(BaseModel):
    extracted_final_answer: str
    reasoning: str
    correct: Literal["yes", "no"]
    confidence: int
    strict: Optional[bool] = None  # optional

def extract_answer(question, correct_answer, response, item_idx=None):
    client = get_client()
    prompt = JUDGE_PROMPT.format(question=question, correct_answer=correct_answer, response=response)

    try:
        response_obj = client.beta.chat.completions.parse(
            model=JUDGE_MODEL,
            max_completion_tokens=8192,  
            messages=[{"role": "user", "content": prompt}],
            response_format=ExtractedAnswer,
            timeout=60.0
        )
        content = response_obj.choices[0].message.parsed
        if VERBOSE:
            usage = getattr(response_obj, "usage", None)
            vlog(f"[judge_result] extracted_final_answer={content.extracted_final_answer!r}")
            vlog(f"[judge_result] correct={content.correct}, confidence={content.confidence}")
            vlog(f"[judge_result] reasoning={content.reasoning}")
            if usage: vlog(f"[judge_usage] {usage}")
        return {
            "correct_answer": correct_answer,
            "model_answer": content.extracted_final_answer,
            "reasoning": content.reasoning,
            "correct": content.correct,
            "confidence": content.confidence
        }
    except Exception as e:
        vlog(f"[judge_error] cap={8192}: {e}")
        return None


def extract_response(messages):
    ANSWER_TAG = "answer"
    def get_answers(text: str):
        pattern = r"<{TAG}>(.*?)</{TAG}>".format(TAG=ANSWER_TAG)
        match = re.search(pattern, text, re.DOTALL)
        if match:
            answer_output = match.group(1).strip()
            return answer_output, 1
        return text, 0
    response, success_flag = get_answers(messages['records'][-1]['content'])
    return response, success_flag

def process_item(item, tokenizer, idx):
    response = item['prediction']
    question, correct_answer = item['question'], item['answer']
    token_usage = item.get('usage', '')
    tool_usage = Counter()

    judge_result = extract_answer(question, correct_answer, response, item_idx=idx)
    if not judge_result:
        if VERBOSE:
            vlog(f"[item {idx}] judge_result=None → acc=0")
        return {
            'acc': 0, 'turns': 0, 'token_usage': token_usage, 'tool_usage': tool_usage,
            'item': item, 'context_length': len(tokenizer.encode('')), 'dollars_o4mini': 0,
            'is_answer': 1,
        }

    acc = 1 if judge_result['correct'] in ('y', 'yes', 'true', 'positive') else 0
    if VERBOSE:
        vlog(f"[item {idx}] acc={acc} (correct={judge_result['correct']})")
    report = {
        'acc': acc,
        'turns': 0,
        'token_usage': token_usage,
        'tool_usage': tool_usage,
        'item': item,
        'context_length': len(tokenizer.encode('')),
        'dollars_o4mini': 0,
        'is_answer': 1,
    }
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_fp', type=str, default="")
    parser.add_argument('--repeat_times', type=int, default=1)
    parser.add_argument('--tokenizer_path', type=str, default='gpt2')
    parser.add_argument('--preflight', action='store_true', help='test API key/base & judge model, then exit')
    parser.add_argument('-v', '--verbose', action='store_true', help='print detailed input/judge/output info')
    args = parser.parse_args()

    VERBOSE = bool(args.verbose)

    if args.preflight:
        ok, msg = _preflight()
        print(msg)
        raise SystemExit(0 if ok else 1)

    input_fp = args.input_fp
    d = load_jsonl(input_fp) * args.repeat_times

    if VERBOSE:
        vlog(f"[start] input_fp={input_fp}, items={len(d)}")
        vlog(f"[start] tokenizer_path={args.tokenizer_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    res = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_idx = {}
        for idx, item in enumerate(d):
            future = executor.submit(process_item, item, tokenizer, idx)
            future_to_idx[future] = idx
        for future in tqdm(as_completed(future_to_idx), total=len(d), desc="Processing"):
            try:
                result = future.result()
                if result is not None:
                    res.append(result)
            except Exception as e:
                print(f"Task failed with error: {e}")

    metrics = [i['acc'] for i in res]
    metrics = sum(metrics)/len(res) if res else 0
    is_answer_rate = (len([i for i in res if i['is_answer']]) / len(res)) if res else 0

    completion_tokens = 0
    prompt_tokens = 0
    for i in res:
        if i['token_usage']:
            completion_tokens += i['token_usage'].get('completion_tokens', 0)
            prompt_tokens += i['token_usage'].get('prompt_tokens', 0)
    n = max(1, len(res))
    avg_completion_tokens = completion_tokens / n
    avg_prompt_tokens = prompt_tokens / n

    acc_under_turns, turns_dist, context_length_under_turns, dollars_under_turn = Counter(), Counter(), Counter(), Counter()
    for i in res:
        for turn in range(31):
            if i['turns'] <= turn and i['acc'] == 1:
                acc_under_turns[turn] += 1
        turns_dist[i['turns']] += 1
        context_length_under_turns[i['turns']] += i['context_length']
        dollars_under_turn[i['turns']] += i['dollars_o4mini']
    for i in context_length_under_turns:
        context_length_under_turns[i] /= max(1, turns_dist[i])
    context_length_under_turns = sorted(context_length_under_turns.items(), key=lambda x: x[0])
    for i in dollars_under_turn:
        dollars_under_turn[i] /= max(1, turns_dist[i])
    dollars_under_turn = sorted(dollars_under_turn.items(), key=lambda x: x[0])

    token_usage_under_turns = Counter()
    for i in res:
        for turn in range(31):
            if i['turns'] <= turn and i['acc'] == 1:
                acc_under_turns[turn] += 1

    tool_usage = defaultdict(list)
    tool_usage_correct = defaultdict(list)
    for i in res:
        cur_usage = {'google_search': 0, 'google_scholar': 0, 'Visit': 0, 'PythonInterpreter': 0}
        for tool_use in i['tool_usage']:
            if tool_use in cur_usage:
                cur_usage[tool_use] += i['tool_usage'][tool_use]
        for tool_use in cur_usage:
            tool_usage[tool_use].append(cur_usage[tool_use])
            if i['acc'] == 1:
                tool_usage_correct[tool_use].append(cur_usage[tool_use])
    tool_usage = {k: (sum(v) / n) for k, v in tool_usage.items()}
    correct_num = sum(1 for r in res if r['acc'] == 1)
    tool_usage_correct = {k: (sum(v) / max(1, correct_num)) for k, v in tool_usage_correct.items()}

    avg_turns = (sum([i['turns'] for i in res]) / n) if res else 0
    avg_turns_correct = (sum([i['turns'] for i in res if i['acc'] == 1]) / max(1, correct_num)) if res else 0

    report = {
        'evaluated_nums': len(d),
        'valid_nums': len(res),
        'metrics': metrics * 100,
        'judge_model': JUDGE_MODEL,
        'avg_prompt_tokens': avg_prompt_tokens,
        'avg_completion_tokens': avg_completion_tokens,
        'avg_dollars_o4mini': avg_prompt_tokens/1_000_000 * 1.1 + avg_completion_tokens/1_000_000 * 4.4,
        'avg_dollars_claude': avg_prompt_tokens/1_000_000 * 3 + avg_completion_tokens/1_000_000 * 15,
        'tool_usage': tool_usage,
        'tool_usage_correct': tool_usage_correct,
        'is_answer_rate': is_answer_rate,
        'repeat_times': args.repeat_times,
        "avg_turns": avg_turns,
        "avg_turns_correct": avg_turns_correct,
        "turns_dist": turns_dist
    }

    print(report)

    # Output paths (suffixes added to FINAL component only)
    import os as _os
    root, ext = _os.path.splitext(input_fp)
    if ext.lower() != ".jsonl":
        root = input_fp
    eval_details_path = root + '.eval_details.jsonl'
    report_path       = root + '.report.json'
    _os.makedirs(_os.path.dirname(eval_details_path) or ".", exist_ok=True)

    if VERBOSE:
        vlog(f"[write] eval_details_path={eval_details_path}")
        vlog(f"[write] report_path={report_path}")

    write_jsonl(res, eval_details_path)
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4)
