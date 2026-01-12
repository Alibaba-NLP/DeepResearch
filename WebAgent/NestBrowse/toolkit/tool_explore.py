import os
import json5
from utils import count_tokens, call_llm
from prompts import *


async def process_response(raw_response, goal, summary_model, tokenizer, sem):
    limit = int(os.getenv("MAX_SUMMARY_SHARD_LEN"))
    record = []
    raw_response_shard = []

    if count_tokens(raw_response, tokenizer) > limit:
        tokens = tokenizer.encode(raw_response)
        for i in range(0, len(tokens), limit):
            chunk_tokens = tokens[i:i+limit]
            chunk_text = tokenizer.decode(chunk_tokens)
            raw_response_shard.append(chunk_text)
    else:
        raw_response_shard.append(raw_response)
    
    for i, raw_resp in enumerate(raw_response_shard):
        if i == 0:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_SUMMARY_OURS},
                {"role": "user", "content": SUMMARY_PROMPT.format(raw_response=raw_resp, goal=goal)}
            ]
        else:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_SUMMARY_OURS},
                {"role": "user", "content": SUMMARY_PROMPT_INCREMENTAL.format(raw_response=raw_resp, goal=goal, existing_evidence=evidence, existing_summary=summary)}
            ]

        response = await call_llm(sem, messages, int(os.getenv("MAX_SINGLE_GEN_TOKENS")), summary_model, mode="summary")
        messages.append({"role": "assistant", "content": response})

        record.append({"messages": messages})

        processed_response_json = response.split("</think>")[-1].split('<useful_info>')[-1].split('</useful_info>')[0].strip()
        processed_response_json = json5.loads(processed_response_json)

        evidence = processed_response_json["evidence"]
        summary = processed_response_json["summary"]
            
    processed_response = "Evidence in page: \n" + str(evidence) + "\n\n" + "Summary: \n" + str(summary)
    processed_response = processed_response.strip()
    
    return processed_response, record