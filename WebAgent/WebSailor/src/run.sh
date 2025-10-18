##############Evaluation Parameters################
export MODEL_PATH=$1 # Model path

# Dataset names (strictly match the following names):
# - gaia
# - browsecomp_zh (Full set, 289 Cases)
# - browsecomp_en (Full set, 1266 Cases)
# - xbench-deepsearch
export DATASET=$2 
export OUTPUT_PATH=$3 # Output path for prediction results

export TEMPERATURE=0.6 # LLM generation parameter, fixed at 0.6


######################################
### 1. Start server (background)   ###
######################################

# Activate server environment

export GPUS_PER_NODE=${MLP_WORKER_GPU:-${KUBERNETES_CONTAINER_RESOURCE_GPU:-8}}
export NNODES=${MLP_WORKER_NUM:-${WORLD_SIZE:-1}}
export NODE_RANK=${MLP_WORKER_RACK_RANK_INDEX:-${MLP_ROLE_INDEX:-${RANK:-0}}}
export MASTER_ADDR=${MLP_WORKER_0_HOST:-${MASTER_ADDR:-127.0.0.1}}
export MASTER_PORT=${MLP_WORKER_0_PORT:-${MASTER_PORT:-1234}}

# Optional dependency installation
apt update
apt install tmux -y
pip install nvitop

echo "==== Starting Original Model SGLang Server (Port 6001)... ===="
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m sglang.launch_server \
    --model-path $MODEL_PATH --host 0.0.0.0 --tp 2 --port 6001 &

ORIGINAL_SERVER_PID=$!

echo "==== Starting Summary Model SGLang Server (Port 6002)... ===="
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m sglang.launch_server \
    --model-path $SUMMARY_MODEL_PATH --host 0.0.0.0 --tp 4 --port 6002 &

SUMMARY_SERVER_PID=$!

#####################################
### 2. Wait for server ports ready###
#####################################

# sleep 500
timeout=3000
start_time=$(date +%s)
server1_ready=false
server2_ready=false

while true; do
    # Check Local Model
    if ! $server1_ready && curl -s http://localhost:6001/v1/chat/completions > /dev/null; then
        echo -e "\nLocal model (port 6001) is ready!"
        server1_ready=true
    fi
    
    # Check Summary Model
    if ! $server2_ready && curl -s http://localhost:6002/v1/chat/completions > /dev/null; then
        echo -e "\nSummary model (port 6002) is ready!"
        server2_ready=true
    fi
    
    # If both servers are ready, exit loop
    if $server1_ready && $server2_ready; then
        echo "Both servers are ready for inference!"
        break
    fi
    
    # Check if timeout
    current_time=$(date +%s)
    elapsed=$((current_time - start_time))
    if [ $elapsed -gt $timeout ]; then
        echo -e "\nWarning: Server startup timeout after ${timeout} seconds"
        if ! $server1_ready; then
            echo "First server (port 6001) failed to start"
        fi
        if ! $server2_ready; then
            echo "Second server (port 6002) failed to start"
        fi
        break
    fi
    
    printf 'Waiting for servers to start .....'
    sleep 10
done

if $server1_ready && $server2_ready; then
    echo "Proceeding with both servers..."
else
    echo "Proceeding with available servers..."
fi


#####################################
### 3. Start inference           ####
#####################################

echo "==== Starting inference... ===="
# Activate inference conda environment
export QWEN_DOC_PARSER_USE_IDP=false
export QWEN_IDP_ENABLE_CSI=false
export NLP_WEB_SEARCH_ONLY_CACHE=false
export NLP_WEB_SEARCH_ENABLE_READPAGE=false
export NLP_WEB_SEARCH_ENABLE_SFILTER=false
export QWEN_SEARCH_ENABLE_CSI=false
export SPECIAL_CODE_MODE=false

export MAX_WORKERS=20

python -u run_multi_react.py --dataset "$DATASET" --output "$OUTPUT_PATH" --max_workers $MAX_WORKERS --model $MODEL_PATH --temperature $TEMPERATURE 


#####################################
### 4. Start evaluation          ####
#####################################

SUMMARY_PATH="${OUTPUT_PATH}/${DATASET}_summary.jsonl"
export MODEL_NAME=$(basename ${MODEL_PATH}) 
PREDICTION_PATH="${OUTPUT_PATH}/${MODEL_NAME}_sglang/${DATASET}"

echo "Evaluating predictions in $PREDICTION_PATH"
python evaluate.py --input_folder ${PREDICTION_PATH} --restore_result_path ${SUMMARY_PATH} --dataset ${DATASET}
