############## Evaluation Parameters ################
export MODEL_PATH=$1 # Model path

# Dataset names (strictly match the following names):
# - gaia
# - browsecomp_zh (Full set, 289 Cases)
# - browsecomp_en (Full set, 1266 Cases)
export DATASET=$2 
export OUTPUT_PATH=$3 # Output path for prediction results


######################################
##### 0. System Configuration   #####
######################################
# Search Tool 
export GOOGLE_SEARCH_KEY="Your Google Search API Key" 

# Visit Tool 
export JINA_API_KEYS="Your Jina API Key"
export SUMMARY_API_KEY="EMPTY"
export SUMMARY_API_BASE="http://localhost:8001/v1"
export SUMMARY_MODEL_NAME="/path/to/your/Qwen3-30B-A3B-Instruct-2507"

# ReSum Tool 
export RESUM_TOOL_NAME="/path/to/your/ReSum-Tool-30B-A3B" 
export RESUM_TOOL_URL="http://localhost:6002/v1/chat/completions" 


######################################
### 1. Start server (background)   ###
######################################
# Server - default localhost:6001
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve $MODEL_PATH --host 0.0.0.0 --port 6001 --tensor-parallel-size 4 & 

# Summary model 
CUDA_VISIBLE_DEVICES=4,5 vllm serve "/path/to/your/Qwen3-30B-A3B-Instruct-2507" --host 0.0.0.0 --port 8001 --tensor-parallel-size 2 & 

# Resum model 
CUDA_VISIBLE_DEVICES=6,7 vllm serve "/path/to/your/ReSum-Tool-30B-A3B" --host 0.0.0.0 --port 6002 --tensor-parallel-size 2 & 


#####################################
#### 2. Wait for server ready     ###
#####################################
timeout=2400
sleep_interval=30

check_port() {
    local port=$1
    local model_name=$2
    local start_time=$(date +%s)
    
    echo "Wait for $port ($model_name) to start..."
    while true; do
        if netstat -tuln 2>/dev/null | grep -q ":$port " || ss -tuln 2>/dev/null | grep -q ":$port "; then
            if curl -s --connect-timeout 10 --max-time 5 http://localhost:$port/v1/chat/completions > /dev/null 2>&1; then
                echo "Port $port ($model_name) is ready."
                return 0
            fi
        fi
    
        current_time=$(date +%s)
        elapsed=$((current_time - start_time))
        if [ $elapsed -gt $timeout ]; then
            echo "Error: start port $port ($model_name) timeout ($timeout seconds)!"
            return 1
        fi
        echo "Port $port ($model_name) is not started, waiting $sleep_interval seconds to retry... (elapsed ${elapsed} seconds)"
        sleep $sleep_interval
    done
}

check_port 6001 "Infer Model" || exit 1 
check_port 8001 "Summary Model" || exit 1  
check_port 6002 "ReSum Model" || exit 1  

echo "All vLLM services are ready!"


#####################################
### 3. Start inference           ####
#####################################
echo "==== Starting inference... ===="

# Todo: Activate inference conda environment

export RESUM=True
# Other settings 
# export MAX_CONTEXT=32
# export MAX_LLM_CALL_PER_RUN=80 

python3 -u main.py \
        --dataset $DATASET \
        --output $OUTPUT_PATH \
        --max_workers 40 \
        --model $MODEL_PATH 

echo "==== Inference completed! ===="
exit 0 
