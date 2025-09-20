#!/bin/bash

# VLLM Process Management
VLLM_MANAGER_PID=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_FILE="$SCRIPT_DIR/vllm_processes.json"

# Cleanup function
cleanup_vllm() {
    echo "Cleaning up VLLM processes..."

    # First try to kill processes using stored PIDs
    if [ -f "$SCRIPT_DIR/vllm_pids.txt" ]; then
        echo "Using stored PIDs for cleanup..."
        STORED_PIDS=$(cat "$SCRIPT_DIR/vllm_pids.txt" 2>/dev/null || echo "")
        if [ -n "$STORED_PIDS" ]; then
            for pid in $STORED_PIDS; do
                if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
                    echo "Killing PID $pid..."
                    kill -TERM "$pid" 2>/dev/null || true
                fi
            done

            # Wait a bit for graceful shutdown
            sleep 3

            # Force kill any remaining
            for pid in $STORED_PIDS; do
                if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
                    echo "Force killing PID $pid..."
                    kill -KILL "$pid" 2>/dev/null || true
                fi
            done
        fi
        rm -f "$SCRIPT_DIR/vllm_pids.txt"
    fi

    # Fallback: Use Python manager if available for comprehensive cleanup
    if [ -f "$SCRIPT_DIR/vllm_manager.py" ]; then
        python3 "$SCRIPT_DIR/vllm_manager.py" cleanup --state-file "$STATE_FILE" 2>/dev/null || true
    fi

    # Final fallback: Manual process killing
    echo "Final cleanup check..."
    pkill -f "vllm serve" 2>/dev/null || true
    pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true

    # Clean up state files
    rm -f "$STATE_FILE" "$SCRIPT_DIR/vllm_pids.txt"
}

# Set up signal traps
trap 'echo "Received SIGINT, cleaning up..."; cleanup_vllm; exit 130' INT
trap 'echo "Received SIGTERM, cleaning up..."; cleanup_vllm; exit 143' TERM
trap 'cleanup_vllm' EXIT

export TORCHDYNAMO_VERBOSE=1
export TORCHDYNAMO_DISABLE=1
export NCCL_IB_TC=16
export NCCL_IB_SL=5
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth
export NCCL_DEBUG=INFO
export NCCL_IB_HCA=mlx5
export NCCL_IB_TIMEOUT=22
export NCCL_IB_QPS_PER_CONNECTION=8
export NCCL_MIN_NCHANNELS=4
export NCCL_NET_PLUGIN=none
export GLOO_SOCKET_IFNAME=eth0
export QWEN_DOC_PARSER_USE_IDP=false
export QWEN_IDP_ENABLE_CSI=false
export NLP_WEB_SEARCH_ONLY_CACHE=false
export NLP_WEB_SEARCH_ENABLE_READPAGE=false
export NLP_WEB_SEARCH_ENABLE_SFILTER=false
export QWEN_SEARCH_ENABLE_CSI=false
export SPECIAL_CODE_MODE=false
export PYTHONDONTWRITEBYTECODE=1

##############hyperparams################
export MODEL_PATH=/your/model/path
export DATASET=your_dataset_name
export OUTPUT_PATH=/your/output/path
export ROLLOUT_COUNT=3 # eval avg@3
export TEMPERATURE=0.85
export PRESENCE_PENALTY=1.1
export MAX_WORKERS=30


## serper key for search&google scholar
## https://serper.dev/
export SERPER_KEY_ID=your_key

## jina key for read page
## https://jina.ai/
export JINA_API_KEYS=your_key

## summary model api for page summary in visit tool
## https://platform.openai.com/
export API_KEY=your_key
export API_BASE=your_api_base
export SUMMARY_MODEL_NAME=your_summary_model_name

## dashscope key for file parser
## https://dashscope.aliyun.com/
export DASHSCOPE_API_KEY=your_key  # support：qwen-omni-turbo，qwen-plus-latest
export DASHSCOPE_API_BASE=your_api_base
export VIDEO_MODEL_NAME=your_video_model_name
export VIDEO_ANALYSIS_MODEL_NAME=your_analysis_model_name

# code sandbox ip for python interperter
# example for ENDPOINTS_STRING "http://22.16.67.220:8080,http://22.16.78.153:8080,http://22.17.10.216:8080,http://22.14.58.9:8080,http://22.16.14.3:8080,http://22.17.26.164:8080,http://22.16.245.207:8080"
# we use sandbox_fusion: https://github.com/bytedance/SandboxFusion
ENDPOINTS_STRING="your_sandbox_endpoint"
export SANDBOX_FUSION_ENDPOINT="$ENDPOINTS_STRING"
export TORCH_COMPILE_CACHE_DIR="./cache"

# IDP service is used for file parsing. If set to false, rule-based parsing is used. You can add an IDP key and set USE_IDP=True to use a more powerful file parsing tool.
# https://help.aliyun.com/zh/document-mind/developer-reference/use-idp-llm-to-complete-document-summary
export USE_IDP=False
export IDP_KEY_ID=your_idp_key_id
export IDP_KEY_SECRET=your_idp_key_secret

######################################
### 1. start server           ###
######################################

# You can customize the VLLM server startup by:
# 1. Commenting out GPU lines you don't have
# 2. Modifying VLLM parameters as needed
# 3. Changing ports if required
# The cleanup will work regardless of how you start the servers

echo "Starting VLLM servers..."

# Store PIDs for cleanup tracking
VLLM_PIDS=()

# GPU 0 - Port 6001
CUDA_VISIBLE_DEVICES=0 vllm serve $MODEL_PATH --host 0.0.0.0 --port 6001 --disable-log-requests &
VLLM_PIDS+=($!)

# GPU 1 - Port 6002
CUDA_VISIBLE_DEVICES=1 vllm serve $MODEL_PATH --host 0.0.0.0 --port 6002 --disable-log-requests &
VLLM_PIDS+=($!)

# GPU 2 - Port 6003
CUDA_VISIBLE_DEVICES=2 vllm serve $MODEL_PATH --host 0.0.0.0 --port 6003 --disable-log-requests &
VLLM_PIDS+=($!)

# GPU 3 - Port 6004
CUDA_VISIBLE_DEVICES=3 vllm serve $MODEL_PATH --host 0.0.0.0 --port 6004 --disable-log-requests &
VLLM_PIDS+=($!)

# GPU 4 - Port 6005
CUDA_VISIBLE_DEVICES=4 vllm serve $MODEL_PATH --host 0.0.0.0 --port 6005 --disable-log-requests &
VLLM_PIDS+=($!)

# GPU 5 - Port 6006
CUDA_VISIBLE_DEVICES=5 vllm serve $MODEL_PATH --host 0.0.0.0 --port 6006 --disable-log-requests &
VLLM_PIDS+=($!)

# GPU 6 - Port 6007
CUDA_VISIBLE_DEVICES=6 vllm serve $MODEL_PATH --host 0.0.0.0 --port 6007 --disable-log-requests &
VLLM_PIDS+=($!)

# GPU 7 - Port 6008
CUDA_VISIBLE_DEVICES=7 vllm serve $MODEL_PATH --host 0.0.0.0 --port 6008 --disable-log-requests &
VLLM_PIDS+=($!)

# Save PIDs to state file for cleanup
echo "Started VLLM processes with PIDs: ${VLLM_PIDS[@]}"
echo "${VLLM_PIDS[@]}" > "$SCRIPT_DIR/vllm_pids.txt"

#######################################################
### 2. Waiting for the server port to be ready  ###
######################################################

timeout=6000
start_time=$(date +%s)

main_ports=(6001 6002 6003 6004 6005 6006 6007 6008)
echo "Mode: All ports used as main model"

declare -A server_status
for port in "${main_ports[@]}"; do
    server_status[$port]=false
done

echo "Waiting for servers to start..."

while true; do
    all_ready=true

    for port in "${main_ports[@]}"; do
        if [ "${server_status[$port]}" = "false" ]; then
            if curl -s -f http://localhost:$port/v1/models > /dev/null 2>&1; then
                echo "Main model server (port $port) is ready!"
                server_status[$port]=true
            else
                all_ready=false
            fi
        fi
    done

    if [ "$all_ready" = "true" ]; then
        echo "All servers are ready for inference!"
        break
    fi

    current_time=$(date +%s)
    elapsed=$((current_time - start_time))
    if [ $elapsed -gt $timeout ]; then
        echo -e "\nError: Server startup timeout after ${timeout} seconds"

        for port in "${main_ports[@]}"; do
            if [ "${server_status[$port]}" = "false" ]; then
                echo "Main model server (port $port) failed to start"
            fi
        done

        exit 1
    fi

    printf 'Waiting for servers to start .....'
    sleep 10
done

failed_servers=()
for port in "${main_ports[@]}"; do
    if [ "${server_status[$port]}" = "false" ]; then
        failed_servers+=($port)
    fi
done

if [ ${#failed_servers[@]} -gt 0 ]; then
    echo "Error: The following servers failed to start: ${failed_servers[*]}"
    exit 1
else
    echo "All required servers are running successfully!"
fi

#####################################
### 3. start infer               ####
#####################################

echo "==== start infer... ===="


cd "$( dirname -- "${BASH_SOURCE[0]}" )"

python -u run_multi_react.py --dataset "$DATASET" --output "$OUTPUT_PATH" --max_workers $MAX_WORKERS --model $MODEL_PATH --temperature $TEMPERATURE --presence_penalty $PRESENCE_PENALTY --total_splits ${WORLD_SIZE:-1} --worker_split $((${RANK:-0} + 1)) --roll_out_count $ROLLOUT_COUNT
