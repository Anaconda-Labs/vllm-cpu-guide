#!/usr/bin/env python3
"""
Complete vLLM CPU tutorial script.
This script demonstrates the full workflow: checking setup, providing instructions,
and running a sample query (requires manual server start in between).
"""

import sys
import subprocess

def check_vllm_installation():
    """Verify that vLLM is properly installed."""
    print("Step 1: Checking vLLM installation")
    print("-" * 50)

    try:
        import vllm
        print(f"✓ vLLM is installed (version {vllm.__version__})")
        return True
    except ImportError:
        print("❌ vLLM is not installed")
        print("   Run: conda env create -f environment.yml")
        print("   Then: conda activate vllm-cpu-demo")
        return False

def check_dependencies():
    """Verify that required dependencies are installed."""
    print("\nStep 2: Checking dependencies")
    print("-" * 50)

    required = ["requests", "fastapi", "uvicorn"]
    all_present = True

    for package in required:
        try:
            __import__(package)
            print(f"✓ {package} is installed")
        except ImportError:
            print(f"❌ {package} is not installed")
            all_present = False

    return all_present

def print_instructions():
    """Print instructions for running the tutorial."""
    print("\n" + "=" * 50)
    print("TUTORIAL INSTRUCTIONS")
    print("=" * 50)
    print()
    print("✓ Environment setup is complete!")
    print()
    print("To complete the tutorial, follow these steps:")
    print()
    print("1. START THE SERVER (in this terminal):")
    print("   python start_server.py")
    print()
    print("2. WAIT FOR SERVER TO START")
    print("   Wait until you see: 'Uvicorn running on http://0.0.0.0:8000'")
    print("   This may take 1-2 minutes on first run (downloading model)")
    print()
    print("3. QUERY THE MODEL (open a NEW terminal):")
    print("   conda activate vllm-cpu-demo")
    print("   python query_model.py")
    print()
    print("4. OBSERVE THE RESULTS")
    print("   You should see generated text responses from the model")
    print()
    print("=" * 50)
    print("\nNote: This tutorial uses a small model (opt-125m) that runs")
    print("      efficiently on CPU without requiring a GPU.")
    print()

def main():
    """Run the tutorial setup and provide instructions."""

    print()
    print("=" * 50)
    print("vLLM CPU Tutorial - Setup Check")
    print("=" * 50)
    print()

    # Check installation
    vllm_ok = check_vllm_installation()
    if not vllm_ok:
        sys.exit(1)

    deps_ok = check_dependencies()
    if not deps_ok:
        print("\n❌ Missing dependencies. Please recreate the environment.")
        sys.exit(1)

    # Print tutorial instructions
    print_instructions()

if __name__ == "__main__":
    main()
