export JINA_API_KEY="your_jina_api_key"
export SEARCH_API_URL="your_search_api_url"
export GOOGLE_SEARCH_KEY="your_google_search_key"

export SUMMARY_MODEL_PATH="/path/Qwen2.5-72B-Instruct"
export MAX_LENGTH=$((1024 * 31 - 500))

cd src

# The arguments are the model path, the dataset name, and the location of the prediction file.
# bash run.sh <model_path> <dataset> <output_path>

# Dataset names (strictly match the following names):
# - gaia
# - browsecomp_zh (Full set, 289 Cases)
# - browsecomp_en (Full set, 1266 Cases)
# - xbench-deepsearch

bash run.sh websailor_3b gaia output_path