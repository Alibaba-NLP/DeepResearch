#!/bin/bash
# MLX Inference Script for Apple Silicon (M1/M2/M3/M4)
# This script runs DeepResearch using Apple's MLX framework instead of vLLM/CUDA
#
# Uses native MLX Python API (no separate server needed)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."
ENV_FILE="$PROJECT_ROOT/.env"
VENV_PATH="$PROJECT_ROOT/venv"

# Activate virtual environment if it exists
if [ -d "$VENV_PATH" ]; then
    echo "Activating virtual environment..."
    source "$VENV_PATH/bin/activate"
else
    echo "Warning: No venv found at $VENV_PATH"
    echo "Create one with: python3 -m venv $VENV_PATH"
    echo "Then install: pip install mlx-lm python-dotenv requests json5 tqdm qwen-agent"
    exit 1
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

# Default inference parameters
TEMPERATURE="${TEMPERATURE:-0.85}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
TOP_P="${TOP_P:-0.95}"

echo "============================================"
echo "DeepResearch MLX Inference (Apple Silicon)"
echo "============================================"
echo "Model: $MLX_MODEL"
echo "Temperature: $TEMPERATURE"
echo "Top-P: $TOP_P"
echo "Max Tokens: $MAX_TOKENS"
echo "============================================"

# Check if mlx-lm is installed
python -c "import mlx_lm" 2>/dev/null || {
    echo "Error: mlx-lm not installed. Install with: pip install mlx-lm"
    exit 1
}

# Disable tokenizer parallelism warning
export TOKENIZERS_PARALLELISM=false

# Run inference using native MLX API (no server needed)
cd "$SCRIPT_DIR"

python -u run_mlx_react.py \
    --dataset "${DATASET:-$PROJECT_ROOT/eval_data/sample_questions.jsonl}" \
    --output "${OUTPUT_PATH:-./outputs}" \
    --model "$MLX_MODEL" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --max_tokens "$MAX_TOKENS" \
    --roll_out_count "${ROLLOUT_COUNT:-1}"

echo "Inference complete!"
