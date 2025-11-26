#!/bin/bash
# MLX Inference Script for Apple Silicon (M1/M2/M3/M4)
# This script runs DeepResearch using Apple's MLX framework instead of vLLM/CUDA

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."
ENV_FILE="$PROJECT_ROOT/.env"
VENV_PATH="$PROJECT_ROOT/.venv"

# Activate virtual environment if it exists
if [ -d "$VENV_PATH" ]; then
    echo "Activating virtual environment..."
    source "$VENV_PATH/bin/activate"
else
    echo "Warning: No .venv found at $VENV_PATH"
    echo "Make sure mlx-lm is installed: pip install mlx-lm"
fi

# Load environment variables
if [ ! -f "$ENV_FILE" ]; then
    echo "Error: .env file not found at $ENV_FILE"
    echo "Please copy .env.example to .env and configure your settings"
    exit 1
fi

echo "Loading environment variables..."
set -a
source "$ENV_FILE"
set +a

# MLX-specific configuration
MLX_MODEL="${MLX_MODEL:-abalogh/Tongyi-DeepResearch-30B-A3B-4bit}"
MLX_PORT="${MLX_PORT:-8080}"
MLX_HOST="${MLX_HOST:-127.0.0.1}"

# Default inference parameters for MLX
TEMPERATURE="${TEMPERATURE:-0.85}"
MAX_TOKENS="${MAX_TOKENS:-10000}"

echo "============================================"
echo "DeepResearch MLX Inference (Apple Silicon)"
echo "============================================"
echo "Model: $MLX_MODEL"
echo "Server: http://$MLX_HOST:$MLX_PORT"
echo "Temperature: $TEMPERATURE"
echo "============================================"

# Check if mlx-lm is installed
if ! command -v mlx_lm.server &> /dev/null; then
    echo "Error: mlx-lm not installed. Install with: pip install mlx-lm"
    exit 1
fi

######################################
### 1. Start MLX Server            ###
######################################

echo "Starting MLX server..."
echo "Note: First run will download the model (~17GB for 4-bit version)"

# Kill any existing MLX server on the port
lsof -ti:$MLX_PORT | xargs kill -9 2>/dev/null || true

# Start MLX server in background
mlx_lm.server \
    --model "$MLX_MODEL" \
    --host "$MLX_HOST" \
    --port "$MLX_PORT" \
    --temp "$TEMPERATURE" \
    --max-tokens "$MAX_TOKENS" \
    --trust-remote-code \
    --log-level INFO \
    --use-default-chat-template &

MLX_PID=$!
echo "MLX server started with PID: $MLX_PID"

# Trap to cleanup on exit
cleanup() {
    echo "Shutting down MLX server..."
    kill $MLX_PID 2>/dev/null || true
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

######################################
### 2. Wait for server to be ready ###
######################################

echo "Waiting for MLX server to be ready..."
timeout=600  # 10 minutes (model download may take time)
start_time=$(date +%s)

while true; do
    if curl -s -f "http://$MLX_HOST:$MLX_PORT/v1/models" > /dev/null 2>&1; then
        echo "MLX server is ready!"
        break
    fi
    
    current_time=$(date +%s)
    elapsed=$((current_time - start_time))
    
    if [ $elapsed -gt $timeout ]; then
        echo "Error: MLX server startup timeout after ${timeout} seconds"
        exit 1
    fi
    
    echo -n "."
    sleep 5
done

######################################
### 3. Run Inference               ###
######################################

echo ""
echo "==== Starting inference ===="

cd "$SCRIPT_DIR"

# Use MLX-specific react agent script
python -u run_mlx_react.py \
    --dataset "${DATASET:-$PROJECT_ROOT/eval_data/sample_questions.jsonl}" \
    --output "${OUTPUT_PATH:-./outputs}" \
    --max_workers "${MAX_WORKERS:-1}" \
    --model "$MLX_MODEL" \
    --mlx_port "$MLX_PORT" \
    --temperature "$TEMPERATURE" \
    --presence_penalty "${PRESENCE_PENALTY:-1.1}" \
    --roll_out_count "${ROLLOUT_COUNT:-1}"

echo "Inference complete!"
