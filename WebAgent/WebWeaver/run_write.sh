

export SCRAPER_API_KEY="SCRAPER_API_KEY"
export SERPER_KEY_ID="SERPER_KEY_ID"
export DASHSCOPE_API_KEY="DASHSCOPE_API_KEY"
export DASHSCOPE_API_BASE="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

export INFER_MODEL_PATH="qwen3-235b-a22b-instruct-2507"


## write, YOUR_OUTLINE_PATH should be the same as the one you used in run_search.sh 
python run_write_outline.py \
    --model $INFER_MODEL_PATH \
    --outline_path YOUR_OUTLINE_PATH \
    --output_path YOUR_ANSWER_PATH \
    --temperature 0.7 \
    --top_p 0.95 \
    --max_workers 1 \
    --write_pattern "multi_turn" \
    --if_infer True