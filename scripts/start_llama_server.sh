#!/bin/bash
# =============================================================================
# DeepResearch Local Server - llama.cpp with Metal Acceleration
# =============================================================================
#
# This script starts the llama.cpp server optimized for the DeepResearch
# ReAct agent workflow on Apple Silicon.
#
# The server provides:
#   - OpenAI-compatible API at http://localhost:8080/v1/chat/completions
#   - Built-in Web UI at http://localhost:8080 (chat interface!)
#   - Metal (GPU) acceleration for fast inference
#   - Model loaded once and kept resident in memory
#
# Usage:
#   ./scripts/start_llama_server.sh              # Start with defaults
#   ./scripts/start_llama_server.sh --ctx 16384  # Custom context size
#   ./scripts/start_llama_server.sh --no-webui   # API only, no web UI
#
# Access:
#   - Web UI:  http://localhost:8080
#   - API:     http://localhost:8080/v1/chat/completions
#   - CLI:     python inference/interactive_llamacpp.py
#
# =============================================================================

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LLAMA_SERVER="$PROJECT_DIR/llama.cpp/build/bin/llama-server"
MODEL_PATH="$PROJECT_DIR/models/gguf/Alibaba-NLP_Tongyi-DeepResearch-30B-A3B-Q4_K_M.gguf"

# Default settings (optimized for Apple Silicon with 32GB RAM)
PORT=${PORT:-8080}
HOST=${HOST:-127.0.0.1}
CTX_SIZE=${CTX_SIZE:-16384}       # 16K context (use --ctx 32768 for longer sessions)
GPU_LAYERS=${GPU_LAYERS:-99}      # Offload all layers to Metal
THREADS=${THREADS:-8}             # CPU threads for non-GPU ops
PARALLEL=${PARALLEL:-1}           # Parallel request slots
BATCH_SIZE=${BATCH_SIZE:-512}     # Batch size for prompt processing
WEBUI=${WEBUI:-true}              # Enable web UI by default
MLOCK=${MLOCK:-false}             # Don't lock model in RAM (saves memory for other apps)

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "============================================================"
echo "  DeepResearch Local Server (llama.cpp + Metal)"
echo "============================================================"
echo -e "${NC}"

# Check if llama-server exists
if [ ! -f "$LLAMA_SERVER" ]; then
    echo -e "${RED}Error: llama-server not found at $LLAMA_SERVER${NC}"
    echo ""
    echo "Please build llama.cpp first:"
    echo "  cd $PROJECT_DIR/llama.cpp"
    echo "  cmake -B build -DLLAMA_METAL=ON -DCMAKE_BUILD_TYPE=Release"
    echo "  cmake --build build --config Release"
    exit 1
fi

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${RED}Error: Model not found at $MODEL_PATH${NC}"
    echo ""
    echo "Please download the model first:"
    echo "  cd $PROJECT_DIR/models/gguf"
    echo "  curl -L -C - -o Alibaba-NLP_Tongyi-DeepResearch-30B-A3B-Q4_K_M.gguf \\"
    echo "    'https://huggingface.co/bartowski/Alibaba-NLP_Tongyi-DeepResearch-30B-A3B-GGUF/resolve/main/Alibaba-NLP_Tongyi-DeepResearch-30B-A3B-Q4_K_M.gguf'"
    exit 1
fi

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        --ctx)
            CTX_SIZE="$2"
            shift 2
            ;;
        --threads)
            THREADS="$2"
            shift 2
            ;;
        --parallel)
            PARALLEL="$2"
            shift 2
            ;;
        --no-webui)
            WEBUI=false
            shift
            ;;
        --webui)
            WEBUI=true
            shift
            ;;
        --mlock)
            MLOCK=true
            shift
            ;;
        --low-memory)
            # Low memory mode: smaller context, no mlock
            CTX_SIZE=8192
            MLOCK=false
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --port N      Port number (default: 8080)"
            echo "  --ctx N       Context size (default: 16384)"
            echo "  --threads N   CPU threads (default: 8)"
            echo "  --parallel N  Parallel requests (default: 1)"
            echo "  --webui       Enable web UI (default)"
            echo "  --no-webui    Disable web UI, API only"
            echo "  --mlock       Lock model in RAM (uses more memory but faster)"
            echo "  --low-memory  Low memory mode: 8K context, no mlock"
            echo "  -h, --help    Show this help"
            echo ""
            echo "Access points:"
            echo "  Web UI:  http://127.0.0.1:PORT"
            echo "  API:     http://127.0.0.1:PORT/v1/chat/completions"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo -e "${GREEN}Configuration:${NC}"
echo "  Model:     $(basename "$MODEL_PATH")"
echo "  Size:      $(du -h "$MODEL_PATH" | cut -f1)"
echo "  Context:   $CTX_SIZE tokens"
echo "  GPU:       Metal (all $GPU_LAYERS layers)"
echo "  Threads:   $THREADS"
echo "  Parallel:  $PARALLEL slots"
echo "  Mlock:     $MLOCK"
echo "  Web UI:    $WEBUI"
echo "  Endpoint:  http://$HOST:$PORT"
echo ""

# Check for existing server on port
if lsof -i :$PORT > /dev/null 2>&1; then
    echo -e "${YELLOW}Warning: Port $PORT is already in use.${NC}"
    echo "Existing process:"
    lsof -i :$PORT | head -2
    echo ""
    read -p "Kill existing process and continue? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        lsof -t -i :$PORT | xargs kill -9 2>/dev/null || true
        sleep 1
    else
        echo "Aborting."
        exit 1
    fi
fi

echo -e "${YELLOW}Starting server...${NC}"
echo "(Model loading takes ~30-60 seconds)"
echo ""

# Build command arguments
SERVER_ARGS=(
    --model "$MODEL_PATH"
    --host "$HOST"
    --port "$PORT"
    --ctx-size "$CTX_SIZE"
    --n-gpu-layers "$GPU_LAYERS"
    --threads "$THREADS"
    --parallel "$PARALLEL"
    --batch-size "$BATCH_SIZE"
    --flash-attn auto
    --metrics
    --log-disable
    --alias deepresearch
)
# Note: --jinja is enabled by default in recent llama.cpp versions

# Add mlock if requested (uses more memory but may be faster)
if [ "$MLOCK" = "true" ]; then
    SERVER_ARGS+=(--mlock)
fi

# Add no-webui flag if requested
if [ "$WEBUI" = "false" ]; then
    SERVER_ARGS+=(--no-webui)
fi

# Start the server with optimized settings for DeepResearch
exec "$LLAMA_SERVER" "${SERVER_ARGS[@]}" 2>&1 | while read -r line; do
    # Colorize output
    if [[ $line == *"error"* ]] || [[ $line == *"Error"* ]]; then
        echo -e "${RED}$line${NC}"
    elif [[ $line == *"listening"* ]] || [[ $line == *"ready"* ]]; then
        echo -e "${GREEN}$line${NC}"
        echo ""
        echo -e "${GREEN}============================================================${NC}"
        echo -e "${GREEN}  Server ready!${NC}"
        echo -e "${GREEN}============================================================${NC}"
        if [ "$WEBUI" = "true" ]; then
            echo ""
            echo -e "${GREEN}  Web UI:  http://$HOST:$PORT${NC}"
            echo "           Open in your browser for a chat interface!"
        fi
        echo ""
        echo -e "${GREEN}  API:     http://$HOST:$PORT/v1/chat/completions${NC}"
        echo ""
        echo "Run DeepResearch CLI:"
        echo "  python inference/interactive_llamacpp.py"
        echo ""
        echo "Test API:"
        echo "  curl http://$HOST:$PORT/v1/chat/completions \\"
        echo "    -H 'Content-Type: application/json' \\"
        echo "    -d '{\"model\": \"deepresearch\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}]}'"
        echo ""
        echo "Press Ctrl+C to stop the server."
    elif [[ $line == *"warning"* ]] || [[ $line == *"Warning"* ]]; then
        echo -e "${YELLOW}$line${NC}"
    else
        echo "$line"
    fi
done
