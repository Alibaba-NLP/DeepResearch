### run search

export SCRAPER_API_KEY="SCRAPER_API_KEY"
export SERPER_KEY_ID="SERPER_KEY_ID"
export DASHSCOPE_API_KEY="DASHSCOPE_API_KEY"
export DASHSCOPE_API_BASE="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
export SUMMARY_MODEL_PATH="SAME AS the SUMMARY_MODEL_PATH in run_vllm_search.sh"


python run_search_outline.py \
    --model "qwen3-235b-a22b-instruct-2507" \
    --output_path YOUR_OUTPUT_OUTLINE_PATH \
    --dataset sample \
    --temperature 0.6 \
    --top_p 0.95 \
    --max_workers 1  \
    --if_infer True