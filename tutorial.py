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
        print("   Then: conda activate vllm-gpu-demo")
        return False

def check_gpu():
    """Verify that GPU and CUDA are available."""
    print("\nStep 2: Checking GPU availability")
    print("-" * 50)

    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✓ CUDA is available")
            print(f"✓ GPU detected: {gpu_name}")
            print(f"✓ GPU memory: {gpu_memory:.1f} GB")
            return True
        else:
            print("❌ CUDA is not available")
            print("   Ensure you're on a GPU-enabled workstation")
            print("   Run: nvidia-smi to check GPU status")
            return False
    except ImportError:
        print("❌ PyTorch is not installed")
        return False

def check_dependencies():
    """Verify that required dependencies are installed."""
    print("\nStep 3: Checking dependencies")
    print("-" * 50)

    required = ["requests", "torch"]
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
    print("✓ GPU is ready for vLLM!")
    print()
    print("To complete the tutorial, follow these steps:")
    print()
    print("1. START THE SERVER (in this terminal):")
    print("   python start_server.py")
    print()
    print("2. WAIT FOR SERVER TO START")
    print("   Wait until you see: 'Uvicorn running on http://0.0.0.0:8000'")
    print("   First run may take 1-2 minutes (downloading model)")
    print()
    print("3. QUERY THE MODEL (open a NEW terminal):")
    print("   conda activate vllm-gpu-demo")
    print("   python query_model.py")
    print()
    print("4. OBSERVE GPU PERFORMANCE")
    print("   Watch GPU utilization with: watch -n 1 nvidia-smi")
    print("   You should see fast token generation (~100-200 tokens/sec)")
    print()
    print("=" * 50)
    print("\nNote: This tutorial uses GPU acceleration for fast inference.")
    print("      Model: opt-125m (~500MB GPU memory)")
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

    gpu_ok = check_gpu()
    if not gpu_ok:
        print("\n❌ GPU not available. This tutorial requires a GPU workstation.")
        print("   Use an Outerbounds GPU workstation or similar Linux GPU instance.")
        sys.exit(1)

    deps_ok = check_dependencies()
    if not deps_ok:
        print("\n❌ Missing dependencies. Please recreate the environment.")
        sys.exit(1)

    # Print tutorial instructions
    print_instructions()

if __name__ == "__main__":
    main()
