export API_KEY=$1
export BASE_URL=$2
export JUDGE_MODEL=$3
export MAX_WORKERS=$4

python evaluation/evaluate_hle_official.py \
    --input_fp YOUR_PREDICTIONS_FILE.jsonl \