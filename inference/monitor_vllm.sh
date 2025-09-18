#!/bin/bash

# VLLM Process Monitor and Cleanup Script
# This script helps monitor VLLM processes and provides quick cleanup options

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_FILE="$SCRIPT_DIR/vllm_processes.json"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}===========================================${NC}"
    echo -e "${BLUE}         VLLM Process Monitor${NC}"
    echo -e "${BLUE}===========================================${NC}"
}

print_usage() {
    cat << EOF
Usage: $0 [COMMAND]

Commands:
    status      Show status of managed VLLM processes
    list        List all VLLM processes (managed and orphaned)
    cleanup     Clean up all VLLM processes
    kill        Force kill all VLLM processes
    gpu         Show GPU memory usage
    ports       Show which ports are in use
    help        Show this help message

Examples:
    $0 status       # Show managed process status
    $0 list         # List all VLLM processes
    $0 cleanup      # Clean shutdown of all processes
    $0 kill         # Force kill all VLLM processes
    $0 gpu          # Show GPU memory usage
EOF
}

show_status() {
    print_header
    echo "Managed VLLM Process Status:"
    echo "----------------------------"

    if [ -f "$SCRIPT_DIR/vllm_manager.py" ]; then
        python3 "$SCRIPT_DIR/vllm_manager.py" status --state-file "$STATE_FILE"
    else
        echo -e "${RED}Error: vllm_manager.py not found${NC}"
        return 1
    fi
}

list_all_processes() {
    print_header
    echo "All VLLM Processes:"
    echo "-------------------"

    echo -e "${YELLOW}Searching for VLLM processes...${NC}"

    # Find all VLLM related processes
    VLLM_PIDS=$(pgrep -f "vllm" 2>/dev/null || true)

    if [ -z "$VLLM_PIDS" ]; then
        echo -e "${GREEN}No VLLM processes found${NC}"
        return 0
    fi

    echo -e "PID\tCOMMAND"
    echo -e "---\t-------"

    for pid in $VLLM_PIDS; do
        if [ -e /proc/$pid ]; then
            cmd=$(ps -p $pid -o command= 2>/dev/null | cut -c1-80)
            echo -e "$pid\t$cmd"
        fi
    done

    echo
    echo -e "${YELLOW}Total VLLM processes: $(echo $VLLM_PIDS | wc -w)${NC}"
}

cleanup_processes() {
    print_header
    echo "Cleaning up VLLM processes..."
    echo "------------------------------"

    if [ -f "$SCRIPT_DIR/vllm_manager.py" ]; then
        python3 "$SCRIPT_DIR/vllm_manager.py" cleanup --state-file "$STATE_FILE"
    else
        echo -e "${YELLOW}Warning: vllm_manager.py not found, using fallback method${NC}"
        force_kill_processes
    fi
}

force_kill_processes() {
    print_header
    echo "Force killing VLLM processes..."
    echo "-------------------------------"

    # Find all VLLM processes
    VLLM_PIDS=$(pgrep -f "vllm" 2>/dev/null || true)

    if [ -z "$VLLM_PIDS" ]; then
        echo -e "${GREEN}No VLLM processes found${NC}"
        return 0
    fi

    echo -e "${YELLOW}Found VLLM processes: $VLLM_PIDS${NC}"

    # Kill processes
    for pid in $VLLM_PIDS; do
        if [ -e /proc/$pid ]; then
            echo "Killing process $pid..."
            kill -TERM $pid 2>/dev/null || true
        fi
    done

    # Wait a bit for graceful shutdown
    sleep 5

    # Force kill any remaining processes
    REMAINING_PIDS=$(pgrep -f "vllm" 2>/dev/null || true)
    if [ -n "$REMAINING_PIDS" ]; then
        echo -e "${YELLOW}Force killing remaining processes: $REMAINING_PIDS${NC}"
        for pid in $REMAINING_PIDS; do
            if [ -e /proc/$pid ]; then
                kill -KILL $pid 2>/dev/null || true
            fi
        done
    fi

    # Clean up state file
    rm -f "$STATE_FILE"

    echo -e "${GREEN}Cleanup completed${NC}"
}

show_gpu_usage() {
    print_header
    echo "GPU Memory Usage:"
    echo "-----------------"

    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | \
        awk -F', ' 'BEGIN {printf "%-5s %-20s %-10s %-10s %-8s\n", "GPU", "Name", "Used(MB)", "Total(MB)", "Usage%"}
                   {printf "%-5s %-20s %-10s %-10s %-8s\n", $1, substr($2,1,20), $3, $4, $5}'
    else
        echo -e "${RED}Error: nvidia-smi not found${NC}"
        return 1
    fi

    echo
    echo "VLLM processes using GPU:"
    echo "-------------------------"

    VLLM_PIDS=$(pgrep -f "vllm" 2>/dev/null || true)
    if [ -n "$VLLM_PIDS" ]; then
        for pid in $VLLM_PIDS; do
            if [ -e /proc/$pid ]; then
                gpu_mem=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | grep "^$pid," || true)
                if [ -n "$gpu_mem" ]; then
                    mem=$(echo "$gpu_mem" | cut -d',' -f2)
                    echo "PID $pid: ${mem}MB"
                fi
            fi
        done
    else
        echo -e "${GREEN}No VLLM processes found${NC}"
    fi
}

show_ports() {
    print_header
    echo "Port Usage (6001-6008):"
    echo "------------------------"

    for port in {6001..6008}; do
        if netstat -tuln 2>/dev/null | grep ":$port " >/dev/null; then
            pid=$(lsof -ti:$port 2>/dev/null || echo "unknown")
            if [ "$pid" != "unknown" ] && [ -n "$pid" ]; then
                cmd=$(ps -p $pid -o command= 2>/dev/null | cut -c1-40 || echo "unknown")
                echo -e "${RED}Port $port: OCCUPIED (PID: $pid) - $cmd${NC}"
            else
                echo -e "${RED}Port $port: OCCUPIED${NC}"
            fi
        else
            echo -e "${GREEN}Port $port: FREE${NC}"
        fi
    done

    echo
    echo "All listening ports on localhost:"
    echo "----------------------------------"
    netstat -tuln 2>/dev/null | grep "127.0.0.1" | head -20
}

interactive_menu() {
    while true; do
        print_header
        echo "Select an option:"
        echo "1) Show status"
        echo "2) List all processes"
        echo "3) Show GPU usage"
        echo "4) Show port usage"
        echo "5) Cleanup processes"
        echo "6) Force kill processes"
        echo "7) Exit"
        echo
        read -p "Enter choice [1-7]: " choice

        case $choice in
            1) show_status ;;
            2) list_all_processes ;;
            3) show_gpu_usage ;;
            4) show_ports ;;
            5) cleanup_processes ;;
            6) force_kill_processes ;;
            7) echo "Goodbye!"; exit 0 ;;
            *) echo -e "${RED}Invalid option. Please try again.${NC}" ;;
        esac

        echo
        read -p "Press Enter to continue..."
        clear
    done
}

# Main script logic
case "${1:-}" in
    status)
        show_status
        ;;
    list)
        list_all_processes
        ;;
    cleanup)
        cleanup_processes
        ;;
    kill)
        force_kill_processes
        ;;
    gpu)
        show_gpu_usage
        ;;
    ports)
        show_ports
        ;;
    help|--help|-h)
        print_usage
        ;;
    "")
        interactive_menu
        ;;
    *)
        echo -e "${RED}Error: Unknown command '$1'${NC}"
        echo
        print_usage
        exit 1
        ;;
esac