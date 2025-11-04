from openai import OpenAI
import os, json, re, argparse, time
from tqdm import tqdm
from collections import Counter, defaultdict
from transformers import AutoTokenizer
from dotenv import load_dotenv

load_dotenv()

# ---------- tiny env helpers ----------
def _clean(s): return (s or "").replace("\ufeff", "").replace("\r", "").strip()
def _first_env(*names, default=""):
    for n in names:
        v = os.getenv(n)
        if v and _clean(v): return _clean(v)
    return _clean(default)

# ---------- config (sequential, simple) ----------
BASE_URL   = _first_env("API_BASE", "BASE_URL", "OPENAI_BASE_URL", default="http://128.114.59.123:8666/v1")
API_KEY    = _first_env("API_KEY", "OPENAI_API_KEY", default="EMPTY")  # vLLM usually accepts "EMPTY"
JUDGE_MODEL = _first_env("JUDGE_MODEL", default="Qwen/Qwen3-4B-Instruct-2507-FP8")
MAX_TOKENS  = int(_first_env("JUDGE_MAX_TOKENS", default="1024"))
TIMEOUT_SEC = int(_first_env("JUDGE_TIMEOUT_SEC", default="45"))
TRIM_TAIL_CHARS = int(_first_env("TRIM_TAIL_CHARS", default="8000"))

JUDGE_PROMPT = """You are a strict grader. Compare [response] to [correct_answer] for the given [question].
Return ONLY a compact JSON object with exactly these keys:
{{"extracted_final_answer": string, "reasoning": string, "correct": "yes"|"no", "confidence": integer}}

Rules:
- extracted_final_answer = the final exact answer you extract from [response], or "None" if none.
- correct = "yes" only if extracted_final_answer matches [correct_answer] (allow tiny numeric rounding), else "no".
- confidence = an integer 0-100 (100 if no explicit confidence in [response]).
Do NOT output any text before or after the JSON.

[question]: {question}

[response]: {response}

[correct_answer]: {correct_answer}
"""


# ---------- tiny utils ----------
def load_jsonl(fp):
    with open(fp, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def write_jsonl(rows, fp):
    with open(fp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

_json_obj_re = re.compile(r'(\{[\s\S]*\})')
def try_json(text: str):
    text = (text or "").strip()
    if not text: return None
    try:
        return json.loads(text)
    except Exception:
        m = _json_obj_re.search(text)
        if m:
            try: return json.loads(m.group(1))
            except Exception: return None
    return None

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()

def short_circuit(prediction: str, answer: str) -> bool:
    return norm(prediction) == norm(answer)

def trim_tail(s: str, n: int) -> str:
    s = s or ""
    return s if len(s) <= n else s[-n:]

# ---------- openai client ----------
def get_client():
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)

def judge_one(client, question: str, correct_answer: str, response_text: str):
    """
    Single, synchronous call. Bullet-proof parsing.
    """
    prompt = JUDGE_PROMPT.format(question=question, correct_answer=correct_answer, response=response_text)
    try:
        r = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_TOKENS,
            timeout=TIMEOUT_SEC
        )
        txt = r.choices[0].message.content or ""
    except Exception as e:
        # On any error, return a conservative 'no' with the error in reasoning
        return {
            "extracted_final_answer": "None",
            "reasoning": f"judge error: {e}",
            "correct": "no",
            "confidence": 0
        }

    data = try_json(txt) or {}
    # Guarantee schema, no KeyErrors
    data.setdefault("extracted_final_answer", "None")
    data.setdefault("reasoning", txt)
    data.setdefault("correct", "no")
    data.setdefault("confidence", 50)
    # Coerce types
    if str(data["correct"]).lower() not in ("yes", "no"):
        data["correct"] = "no"
    try:
        data["confidence"] = int(str(data["confidence"]).strip("% ").split()[0])
    except Exception:
        data["confidence"] = 50
    data["extracted_final_answer"] = str(data["extracted_final_answer"])
    data["reasoning"] = str(data["reasoning"])
    return data

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_fp", type=str, required=True)
    ap.add_argument("--repeat_times", type=int, default=1)
    ap.add_argument("--tokenizer_path", type=str, default="gpt2")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    data = load_jsonl(args.input_fp) * max(1, args.repeat_times)
    print(f"[start] input_fp={args.input_fp}, items={len(data)}")
    print(f"[start] BASE_URL={BASE_URL}")
    print(f"[start] JUDGE_MODEL={JUDGE_MODEL}  TOKENS={MAX_TOKENS}  TIMEOUT={TIMEOUT_SEC}s")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    client = get_client()

    results = []
    for idx, item in enumerate(tqdm(data, desc="Judging (sequential)")):
        # extract fields (be tolerant about keys)
        question = item.get("question", "")
        correct_answer = item.get("answer", "")
        prediction = item.get("prediction") or item.get("response") or ""

        # short-circuit exact normalized match, no API call
        if short_circuit(prediction, correct_answer):
            acc = 1
            judge = {
                "extracted_final_answer": prediction,
                "reasoning": "normalized exact match",
                "correct": "yes",
                "confidence": 100
            }
        else:
            trimmed = trim_tail(str(prediction), TRIM_TAIL_CHARS)
            judge = judge_one(client, question, correct_answer, trimmed)
            acc = 1 if str(judge.get("correct", "no")).lower() == "yes" else 0

        row = {
            "acc": acc,
            "turns": 0,
            "token_usage": item.get("usage", {}),
            "tool_usage": Counter(),  # keep shape
            "item": item,
            "context_length": len(tokenizer.encode("")),
            "dollars_o4mini": 0.0,
            "is_answer": 1
        }
        results.append(row)

    # ---- summary report (same fields your downstream expects) ----
    n = max(1, len(results))
    metrics = sum(r["acc"] for r in results) / n
    is_answer_rate = len([r for r in results if r["is_answer"]]) / n
    completion_tokens = sum((r.get("token_usage", {}) or {}).get("completion_tokens", 0) for r in results)
    prompt_tokens = sum((r.get("token_usage", {}) or {}).get("prompt_tokens", 0) for r in results)
    avg_completion_tokens = completion_tokens / n
    avg_prompt_tokens = prompt_tokens / n

    turns_dist = Counter(r["turns"] for r in results)
    context_length_under_turns = defaultdict(float)
    dollars_under_turn = defaultdict(float)
    for r in results:
        context_length_under_turns[r["turns"]] += r["context_length"]
        dollars_under_turn[r["turns"]] += r["dollars_o4mini"]
    for k in list(context_length_under_turns.keys()):
        context_length_under_turns[k] /= max(1, turns_dist[k])

    tool_usage = defaultdict(list)
    tool_usage_correct = defaultdict(list)
    correct_num = sum(1 for r in results if r["acc"] == 1)
    for r in results:
        tu = {"google_search": 0, "google_scholar": 0, "Visit": 0, "PythonInterpreter": 0}
        for k in (r.get("tool_usage") or {}):
            if k in tu:
                tu[k] += r["tool_usage"][k]
        for k in tu:
            tool_usage[k].append(tu[k])
            if r["acc"] == 1:
                tool_usage_correct[k].append(tu[k])
    tool_usage = {k: (sum(v) / n) for k, v in tool_usage.items()}
    tool_usage_correct = {k: (sum(v) / max(1, correct_num)) for k, v in tool_usage_correct.items()}

    avg_turns = sum(r["turns"] for r in results) / n
    avg_turns_correct = (sum(r["turns"] for r in results if r["acc"] == 1) / max(1, correct_num)) if results else 0.0

    report = {
        "evaluated_nums": len(data),
        "valid_nums": len(results),
        "metrics": metrics * 100,
        "judge_model": JUDGE_MODEL,
        "avg_prompt_tokens": avg_prompt_tokens,
        "avg_completion_tokens": avg_completion_tokens,
        "avg_dollars_o4mini": avg_prompt_tokens/1_000_000 * 1.1 + avg_completion_tokens/1_000_000 * 4.4,
        "avg_dollars_claude": avg_prompt_tokens/1_000_000 * 3 + avg_completion_tokens/1_000_000 * 15,
        "tool_usage": tool_usage,
        "tool_usage_correct": tool_usage_correct,
        "is_answer_rate": is_answer_rate,
        "repeat_times": 1,
        "avg_turns": avg_turns,
        "avg_turns_correct": avg_turns_correct,
        "turns_dist": turns_dist
    }

    # ---- write outputs with same naming ----
    root, ext = os.path.splitext(args.input_fp)
    if ext.lower() != ".jsonl":
        root = args.input_fp
    eval_details_path = root + ".eval_details.jsonl"
    report_path       = root + ".report.json"
    os.makedirs(os.path.dirname(eval_details_path) or ".", exist_ok=True)
    write_jsonl(results, eval_details_path)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print(f"[done] wrote {eval_details_path}")
    print(f"[done] wrote {report_path}")

if __name__ == "__main__":
    main()
