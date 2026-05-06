#!/usr/bin/env python3
"""
Start a vLLM server with GPU acceleration.
This script launches the server and keeps it running until interrupted.
"""

import subprocess
import sys

def main():
    """Start vLLM server with GPU acceleration."""

    model_name = "facebook/opt-125m"
    port = 8000

    print(f"Starting vLLM server with GPU acceleration...")
    print(f"Model: {model_name}")
    print(f"Port: {port}")
    print(f"API endpoint: http://localhost:{port}")
    print("\nMonitor GPU usage with: watch -n 1 nvidia-smi")
    print("Press Ctrl+C to stop the server\n")

    # Start vLLM server with GPU settings
    # --host 0.0.0.0 makes it accessible locally
    # --port specifies the port
    # --model specifies which model to use
    # --dtype auto lets vLLM choose optimal precision for GPU
    # --gpu-memory-utilization sets max GPU memory to use (0.9 = 90%)
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_name,
        "--host", "0.0.0.0",
        "--port", str(port),
        "--dtype", "auto",  # Auto-select optimal dtype for GPU
        "--gpu-memory-utilization", "0.9",  # Use up to 90% of GPU memory
    ]

    try:
        # Run the server process
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n\nServer stopped by user")
        sys.exit(0)
    except subprocess.CalledProcessError as e:
        print(f"\nError starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
