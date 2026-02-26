

#!/bin/bash

SUMMARY_MODEL_PATH="SUMMARY_MODEL_PATH"

export LLM_URL="http://localhost:6002/v1/completions"
export LLM_AUTH="EMPTY"

export GPUS_PER_NODE=${MLP_WORKER_GPU:-${KUBERNETES_CONTAINER_RESOURCE_GPU:-8}}
export NNODES=${MLP_WORKER_NUM:-${WORLD_SIZE:-1}}
export NODE_RANK=${MLP_WORKER_RACK_RANK_INDEX:-${MLP_ROLE_INDEX:-${RANK:-0}}}
export MASTER_ADDR=${MLP_WORKER_0_HOST:-${MASTER_ADDR:-127.0.0.1}}
export MASTER_PORT=${MLP_WORKER_0_PORT:-${MASTER_PORT:-1234}}

unset SETUPTOOLS_USE_DISTUTILS || true
export SETUPTOOLS_USE_DISTUTILS=local 

export SUMMARY_MODEL_PATH=$SUMMARY_MODEL_PATH

# echo "==== 启动原模型 VLLM Server (端口6001)... ===="
# CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve $INFER_MODEL_PATH --host 0.0.0.0 --port 6001 --tensor-parallel-size 4  & 

echo "==== 启动 vLLM (6002) ===="
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve $SUMMARY_MODEL_PATH --host 0.0.0.0 --port 6002 --tensor-parallel-size 8  &


timeout=1800
start_time=$(date +%s)
server1_ready=true
server2_ready=false

while true; do
    # 检查Local Model
    if ! $server1_ready && curl -s http://localhost:6001/v1/chat/completions > /dev/null; then
        echo -e "\nLocal model (port 6001) is ready!"
        server1_ready=true
    fi
    
    # 检查Summary Model
    if ! $server2_ready && curl -s http://localhost:6002/v1/chat/completions > /dev/null; then
        echo -e "\nSummary model (port 6002) is ready!"
        server2_ready=true
    fi
    
    # 如果两个服务器都准备好了，退出循环
    if $server1_ready && $server2_ready; then
        echo "Both servers are ready for inference!"
        break
    fi
    
    # 检查是否超时
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
    sleep 3
done

if ! $server1_ready || ! $server2_ready; then
    echo "Error: Some servers are not ready"
    exit 1
else
    echo "Proceeding with available servers..."
fi

source ~/.bashrc

sleep 60


