#!/usr/bin/env python3
"""
Start a vLLM server with a small model for CPU inference.
This script launches the server and keeps it running until interrupted.
"""

import subprocess
import sys

def main():
    """Start vLLM server with CPU-friendly configuration."""

    model_name = "facebook/opt-125m"
    port = 8000

    print(f"Starting vLLM server...")
    print(f"Model: {model_name}")
    print(f"Port: {port}")
    print(f"API endpoint: http://localhost:{port}")
    print("\nPress Ctrl+C to stop the server\n")

    # Start vLLM server with CPU-optimized settings
    # --host 0.0.0.0 makes it accessible locally
    # --port specifies the port
    # --model specifies which model to use
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_name,
        "--host", "0.0.0.0",
        "--port", str(port),
        "--dtype", "float32",  # Use float32 for CPU compatibility
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
