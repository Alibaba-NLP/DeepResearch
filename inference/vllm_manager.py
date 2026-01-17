#!/usr/bin/env python3
"""
VLLM Server Process Manager

This utility manages VLLM server processes for the DeepResearch inference pipeline.
It provides robust process lifecycle management with proper cleanup on exit.
"""

import os
import sys
import json
import signal
import subprocess
import time
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import psutil
import atexit

class VLLMManager:
    def __init__(self, state_file: str = "vllm_processes.json"):
        self.state_file = Path(state_file)
        self.processes: Dict[int, subprocess.Popen] = {}
        self.ports = [6001, 6002, 6003, 6004, 6005, 6006, 6007, 6008]

        # Register cleanup on exit
        atexit.register(self.cleanup_all)

        # Handle signals
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        print(f"\nReceived signal {signum}, cleaning up VLLM processes...")
        self.cleanup_all()
        sys.exit(0)

    def load_state(self) -> Dict:
        """Load process state from file."""
        if not self.state_file.exists():
            return {}

        try:
            with open(self.state_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load state file: {e}")
            return {}

    def save_state(self, state: Dict):
        """Save process state to file."""
        try:
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except IOError as e:
            print(f"Warning: Could not save state file: {e}")

    def is_port_in_use(self, port: int) -> bool:
        """Check if a port is already in use."""
        try:
            # Check if any process is using this port
            for conn in psutil.net_connections():
                if conn.laddr.port == port:
                    return True
            return False
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            return False

    def find_vllm_processes(self) -> List[psutil.Process]:
        """Find all running VLLM processes."""
        vllm_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline_str = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                if cmdline_str and ('vllm serve' in cmdline_str or 'vllm.entrypoints.openai.api_server' in cmdline_str):
                    vllm_processes.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return vllm_processes

    def start_servers(self, model_path: str, gpu_devices: List[int] = None) -> bool:
        """Start VLLM servers on specified GPUs."""
        if gpu_devices is None:
            gpu_devices = list(range(8))  # Default to GPUs 0-7

        if len(gpu_devices) > len(self.ports):
            print(f"Warning: More GPUs ({len(gpu_devices)}) than ports ({len(self.ports)})")
            gpu_devices = gpu_devices[:len(self.ports)]

        print(f"Starting VLLM servers on {len(gpu_devices)} GPUs...")

        state = self.load_state()
        started_processes = {}

        for i, (gpu_id, port) in enumerate(zip(gpu_devices, self.ports)):
            if self.is_port_in_use(port):
                print(f"Port {port} is already in use, skipping GPU {gpu_id}")
                continue

            print(f"Starting VLLM server on GPU {gpu_id}, port {port}...")

            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

            cmd = [
                'vllm', 'serve', model_path,
                '--host', '0.0.0.0',
                '--port', str(port),
                '--disable-log-requests'
            ]

            try:
                # Start process in new process group
                process = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    preexec_fn=os.setsid
                )

                self.processes[port] = process
                started_processes[str(port)] = {
                    'pid': process.pid,
                    'gpu_id': gpu_id,
                    'model_path': model_path,
                    'started_at': time.time()
                }

                print(f"Started VLLM server on port {port} (PID: {process.pid})")

            except Exception as e:
                print(f"Failed to start VLLM server on GPU {gpu_id}: {e}")
                return False

        # Save state
        state.update(started_processes)
        self.save_state(state)

        return len(started_processes) > 0

    def wait_for_servers(self, timeout: int = 600) -> bool:
        """Wait for all servers to be ready."""
        print("Waiting for servers to start...")
        start_time = time.time()

        state = self.load_state()
        ports_to_check = [int(port) for port in state.keys()]

        ready_ports = set()

        while time.time() - start_time < timeout:
            all_ready = True

            for port in ports_to_check:
                if port in ready_ports:
                    continue

                try:
                    import requests
                    response = requests.get(f"http://localhost:{port}/v1/models", timeout=5)
                    if response.status_code == 200:
                        print(f"Server on port {port} is ready!")
                        ready_ports.add(port)
                    else:
                        all_ready = False
                except Exception:
                    all_ready = False

            if len(ready_ports) == len(ports_to_check):
                print("All servers are ready!")
                return True

            if not all_ready:
                print(".", end="", flush=True)
                time.sleep(10)

        print(f"\nTimeout: Only {len(ready_ports)}/{len(ports_to_check)} servers ready")
        return len(ready_ports) > 0

    def stop_servers(self, ports: List[int] = None) -> bool:
        """Stop VLLM servers on specified ports."""
        state = self.load_state()

        if ports is None:
            ports = [int(port) for port in state.keys()]

        success = True
        for port in ports:
            if not self._stop_server(port, state):
                success = False

        # Clean up state file
        remaining_state = {k: v for k, v in state.items() if int(k) not in ports}
        self.save_state(remaining_state)

        return success

    def _stop_server(self, port: int, state: Dict) -> bool:
        """Stop a specific VLLM server."""
        port_str = str(port)

        if port_str not in state:
            print(f"No server recorded on port {port}")
            return True

        pid = state[port_str]['pid']

        try:
            # First try graceful shutdown
            process = psutil.Process(pid)
            print(f"Stopping VLLM server on port {port} (PID: {pid})...")

            # Try SIGTERM first
            process.terminate()

            # Wait up to 30 seconds for graceful shutdown
            try:
                process.wait(timeout=30)
                print(f"Server on port {port} stopped gracefully")
                return True
            except psutil.TimeoutExpired:
                print(f"Server on port {port} didn't stop gracefully, using SIGKILL...")
                process.kill()
                process.wait(timeout=10)
                print(f"Server on port {port} killed")
                return True

        except psutil.NoSuchProcess:
            print(f"Process {pid} already dead")
            return True
        except Exception as e:
            print(f"Error stopping server on port {port}: {e}")
            return False

    def cleanup_all(self):
        """Clean up all VLLM processes."""
        print("Cleaning up all VLLM processes...")

        # First try to stop servers from state
        state = self.load_state()
        if state:
            self.stop_servers()

        # Then find and kill any remaining VLLM processes
        orphan_processes = self.find_vllm_processes()

        for proc in orphan_processes:
            try:
                print(f"Killing orphan VLLM process {proc.pid}")
                proc.terminate()
                proc.wait(timeout=10)
            except (psutil.NoSuchProcess, psutil.TimeoutExpired):
                try:
                    proc.kill()
                except psutil.NoSuchProcess:
                    pass
            except Exception as e:
                print(f"Error killing process {proc.pid}: {e}")

        # Clean up state file
        if self.state_file.exists():
            self.state_file.unlink()

        print("Cleanup completed")

    def status(self) -> Dict:
        """Get status of all managed servers."""
        state = self.load_state()
        status_info = {}

        for port_str, info in state.items():
            port = int(port_str)
            pid = info['pid']

            try:
                process = psutil.Process(pid)
                is_running = process.is_running()

                # Check if port is responsive
                port_responsive = False
                try:
                    import requests
                    response = requests.get(f"http://localhost:{port}/v1/models", timeout=2)
                    port_responsive = response.status_code == 200
                except Exception:
                    pass

                status_info[port] = {
                    'pid': pid,
                    'gpu_id': info.get('gpu_id', 'unknown'),
                    'model_path': info.get('model_path', 'unknown'),
                    'running': is_running,
                    'responsive': port_responsive,
                    'uptime': time.time() - info.get('started_at', 0)
                }
            except psutil.NoSuchProcess:
                status_info[port] = {
                    'pid': pid,
                    'gpu_id': info.get('gpu_id', 'unknown'),
                    'model_path': info.get('model_path', 'unknown'),
                    'running': False,
                    'responsive': False,
                    'uptime': 0
                }

        return status_info


def main():
    parser = argparse.ArgumentParser(description="VLLM Server Process Manager")
    parser.add_argument('command', choices=['start', 'stop', 'status', 'cleanup'],
                       help='Command to execute')
    parser.add_argument('--model', type=str, required=False,
                       help='Model path (required for start command)')
    parser.add_argument('--gpus', type=str, default='0,1,2,3,4,5,6,7',
                       help='Comma-separated list of GPU IDs')
    parser.add_argument('--ports', type=str,
                       help='Comma-separated list of ports to operate on')
    parser.add_argument('--timeout', type=int, default=600,
                       help='Timeout for server startup (seconds)')
    parser.add_argument('--state-file', type=str, default='vllm_processes.json',
                       help='State file path')

    args = parser.parse_args()

    manager = VLLMManager(args.state_file)

    if args.command == 'start':
        if not args.model:
            print("Error: --model is required for start command")
            sys.exit(1)

        gpu_devices = [int(x.strip()) for x in args.gpus.split(',')]

        if manager.start_servers(args.model, gpu_devices):
            if manager.wait_for_servers(args.timeout):
                print("All servers started successfully!")
                sys.exit(0)
            else:
                print("Some servers failed to start")
                sys.exit(1)
        else:
            print("Failed to start servers")
            sys.exit(1)

    elif args.command == 'stop':
        ports = None
        if args.ports:
            ports = [int(x.strip()) for x in args.ports.split(',')]

        if manager.stop_servers(ports):
            print("Servers stopped successfully")
            sys.exit(0)
        else:
            print("Some servers failed to stop")
            sys.exit(1)

    elif args.command == 'status':
        status_info = manager.status()
        if not status_info:
            print("No managed servers found")
        else:
            print("VLLM Server Status:")
            print("-" * 80)
            for port, info in status_info.items():
                status = "RUNNING" if info['running'] else "STOPPED"
                responsive = "RESPONSIVE" if info['responsive'] else "NOT RESPONSIVE"
                uptime_str = f"{info['uptime']:.1f}s" if info['uptime'] > 0 else "N/A"
                print(f"Port {port:4d}: {status:8s} | {responsive:14s} | "
                      f"PID {info['pid']:6d} | GPU {info['gpu_id']} | Uptime {uptime_str}")

    elif args.command == 'cleanup':
        manager.cleanup_all()
        print("Cleanup completed")


if __name__ == '__main__':
    main()