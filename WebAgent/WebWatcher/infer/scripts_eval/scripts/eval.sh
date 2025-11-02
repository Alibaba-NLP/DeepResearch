date=$(date +%Y%m%d)

######################################
### 1. 启动 server （后台）         ###
######################################


PROJECT_NAME=${date}
benchmark='hle' 
EXPERIMENT_NAME='1030'
MODEL_PATH=pretrain_model/webwatcher7b
SUMMERY_MODEL_PATH=pretrain_model/qwen2.5_vl_72b
# benchmark=$1
# EXPERIMENT_NAME=$2
# MODEL_PATH=$3
# SUMMERY_MODEL_PATH=$4

SAVE_PATH=scripts_eval/results/${PROJECT_NAME}_${benchmark}
SAVE_FILE=scripts_eval/results/${PROJECT_NAME}_${benchmark}/${EXPERIMENT_NAME}.jsonl

if [ ! -d "$SAVE_PATH" ]; then
    echo "目录 $SAVE_PATH 不存在，正在创建..."
    mkdir -p "$SAVE_PATH"
fi

# search config
echo "" | sudo tee -a /etc/hosts
echo "==== 启动模型 vllm (端口8001)... ===="
vllm serve $MODEL_PATH --port 8001 --host 0.0.0.0 --limit-mm-per-prompt '{"image": 100}' --served-model-name $MODEL_PATH --max-num-batched-tokens 32768 --max-num-seqs 128 --tensor-parallel-size 1 > ${SAVE_PATH}/${EXPERIMENT_NAME}_vllm.log 2>&1 & vllm_pid=$!
# CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve $MODEL_PATH --port 8001 --host 0.0.0.0 --limit-mm-per-prompt '{"image": 100}' --served-model-name $MODEL_PATH --max-num-batched-tokens 32768 --max-num-seqs 128 --tensor-parallel-size 1 > ${SAVE_PATH}/${EXPERIMENT_NAME}_vllm.log 2>&1 & vllm_pid=$!

# echo "==== 启动summery model vllm (端口6002)... ===="
# CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve $SUMMERY_MODEL_PATH --port 6002 --host 0.0.0.0 --served-model-name $SUMMERY_MODEL_PATH --max-num-batched-tokens 32768 --max-num-seqs 128 --tensor-parallel-size 1  & summery_pid=$!

#####################################
### 2. 等待 server 端口 ready     ###
#####################################

timeout=120000
start_time=$(date +%s)
server1_ready=false
server2_ready=false

while true; do
    if ! $server1_ready && curl -s http://localhost:8001/v1/chat/completions > /dev/null; then
        echo -e "\nLocal model (port 8001) is ready!"
        server1_ready=true
        # TODO: delete break
        break
    fi

        # Check Summary Model
    # if ! $server2_ready && curl -s http://localhost:6002/v1/chat/completions > /dev/null; then
    #     echo -e "\nSummary model (port 6002) is ready!"
    #     server2_ready=true
    # fi
    
    # # If both servers are ready, exit loop
    # if $server1_ready && $server2_ready; then
    #     echo "Both servers are ready for inference!"
    #     break
    # fi
    
    current_time=$(date +%s)
    elapsed=$((current_time - start_time))
    if [ $elapsed -gt $timeout ]; then
        echo -e "Warning: Server startup timeout after ${timeout} seconds"
        if ! $server1_ready; then
            echo "Vllm server failed to start"
            exit 1
        fi
    fi
    
    printf 'Waiting for servers to start .....'
    sleep 10
done

#####################################
### 3. 启动 infer                ####
#####################################

echo "==== 启动 infer... ===="

export VLLM_MODEL=$MODEL_PATH

if [ "$benchmark" = "mmsearch" ]; then
  export IMAGE_DIR=scripts_eval/images/mmsearch
  echo "已设置 IMAGE_DIR 为 mmsearch 路径"
elif [ "$benchmark" = "hle" ]; then
  export IMAGE_DIR=scripts_eval/images/hle
  echo "已设置 IMAGE_DIR 为 hle 路径"
elif [ "$benchmark" = "livevqa" ]; then
  export IMAGE_DIR=scripts_eval/images/livevqa
  echo "已设置 IMAGE_DIR 为 livevqa 路径"
elif [ "$benchmark" = "infoseek" ]; then
  export IMAGE_DIR=scripts_eval/images/infoseek
  echo "已设置 IMAGE_DIR 为 infoseek 路径"
elif [ "$benchmark" = "simplevqa" ]; then
  export IMAGE_DIR=scripts_eval/images/simplevqa
  echo "已设置 IMAGE_DIR 为 simplevqa 路径"
elif [ "$benchmark" = "gaia" ]; then
  export IMAGE_DIR=scripts_eval/images/gaia
  echo "已设置 IMAGE_DIR 为 gaia 路径"
elif [ "$benchmark" = "bc_vl_v1" ]; then
  export IMAGE_DIR=scripts_eval/images/bc_vl_v1
  echo "已设置 IMAGE_DIR 为 bc_vl_v1 路径"
elif [ "$benchmark" = "bc_vl_v2" ]; then
  export IMAGE_DIR=scripts_eval/images/bc_vl_v2
  echo "已设置 IMAGE_DIR 为 bc-vl-v2 路径"
else
  echo "警告: 未知的 benchmark 值 '$benchmark'. 未设置 IMAGE_DIR."
fi

export IMG_SEARCH_KEY="6fd1f6920c314a1f47b6ee5b5d2494274ab6d386"
export JINA_API_KEY='jina_b5f95c8238764e49972e57909ed2f77fmV1N1mLXZ-rUNQQOJCb60KduO3Ym'
# export DASHSCOPE_API=$7
export TEXT_SEARCH_KEY="6fd1f6920c314a1f47b6ee5b5d2494274ab6d386"

# export IMG_SEARCH_KEY=$5
# export JINA_API_KEY=$6
# export DASHSCOPE_API=$7
# export TEXT_SEARCH_KEY=$8 
# export ALIBABA_CLOUD_ACCESS_KEY_ID=$9
# export ALIBABA_CLOUD_ACCESS_KEY_SECRET=$10

# pip uninstall qwen-agent
# pip install -e vl_search_r1/qwen-agent-o1_search --no-deps
# pip install "qwen-agent[code_interpreter]"

HOST_LINE="203.119.169.238   idealab.alibaba-inc.com"
if ! grep -q "idealab.alibaba-inc.com" /etc/hosts; then
    echo "$HOST_LINE" | sudo tee -a /etc/hosts
    echo "Added host mapping: $HOST_LINE"
else
    echo "Host mapping already exists, skipped."
fi


# for i in 1 2 3
# do
#     SAVE_FILE=${SAVE_PATH}/${EXPERIMENT_NAME}_round${i}.jsonl
#     [ -s "$SAVE_FILE" ] && > "$SAVE_FILE"
#     python scripts_eval/agent_eval.py \
#         --output_file $SAVE_FILE \
#         --eval_data $benchmark
# done

SAVE_FILE=${SAVE_PATH}/${EXPERIMENT_NAME}.jsonl
python scripts_eval/agent_eval.py \
    --output_file $SAVE_FILE \
    --eval_data $benchmark

# echo "==== 关闭服务... ===="
if kill ${vllm_pid}; then
    echo "成功关闭VLLM服务 (PID: ${vllm_pid})"
else
    echo "警告：未能关闭VLLM服务 (PID: ${vllm_pid})，可能已被关闭或不存在。"
fi

# if kill ${summery_pid}; then
#     echo "成功关闭VLLM服务 (PID: ${summery_pid})"
# else
#     echo "警告：未能关闭VLLM服务 (PID: ${summery_pid})，可能已被关闭或不存在。"
# fi
