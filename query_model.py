#!/usr/bin/env python3
"""
Query a running vLLM server using the OpenAI-compatible API.
Make sure start_server.py is running before executing this script.
"""

import requests
import time
import sys

def check_server_health(base_url: str, max_retries: int = 3) -> bool:
    """Check if the vLLM server is responding."""
    for i in range(max_retries):
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            if i < max_retries - 1:
                print(f"Server not ready, retrying... ({i+1}/{max_retries})")
                time.sleep(2)
    return False

def query_model(base_url: str, prompt: str, max_tokens: int = 50) -> dict:
    """Send a completion request to the vLLM server."""

    url = f"{base_url}/v1/completions"

    payload = {
        "model": "facebook/opt-125m",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }

    print(f"Sending prompt: '{prompt}'")
    print("Waiting for response...\n")

    start_time = time.time()

    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()

        elapsed = time.time() - start_time

        result = response.json()
        return {
            "success": True,
            "text": result["choices"][0]["text"],
            "tokens": result["usage"]["completion_tokens"],
            "time": elapsed
        }

    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "error": str(e)
        }

def main():
    """Main query execution."""

    base_url = "http://localhost:8000"

    print("vLLM Query Tool")
    print("=" * 50)
    print()

    # Check if server is running
    print("Checking server health...")
    if not check_server_health(base_url):
        print("\n❌ Error: vLLM server is not responding.")
        print("   Make sure start_server.py is running in another terminal.")
        sys.exit(1)

    print("✓ Server is ready\n")

    # Example queries
    prompts = [
        "Once upon a time",
        "The future of AI is",
        "Python is a programming language that",
    ]

    for i, prompt in enumerate(prompts, 1):
        print(f"Query {i}/{len(prompts)}")
        print("-" * 50)

        result = query_model(base_url, prompt, max_tokens=30)

        if result["success"]:
            print(f"✓ Response: {result['text']}")
            print(f"  Tokens: {result['tokens']}")
            print(f"  Time: {result['time']:.2f}s")
        else:
            print(f"❌ Error: {result['error']}")

        print()

    print("=" * 50)
    print("Query complete!")
    print("\nTip: Modify the prompts list in this script to try your own queries.")

if __name__ == "__main__":
    main()
